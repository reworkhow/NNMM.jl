#!/usr/bin/env julia
#=
NNMM Benchmark Script with Missing Omics Data

This script tests NNMM's ability to handle missing intermediate layer values,
which is one of its key features. Missing omics are imputed using HMC sampling.

Usage:
    julia --project=. benchmarks/benchmark_missing_omics.jl

Output:
    - Prints accuracy metrics with various missing data percentages
    - Compares performance with full vs missing omics data
=#

using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random
using Dates
using StatsBase  # For sample()

println("=" ^ 70)
println("NNMM Benchmark with Missing Omics Data")
println("=" ^ 70)
println("Date: ", Dates.now())
println()

# Configuration
const SEED = 42
const CHAIN_LENGTH = 500
const BURNIN = 100
const MISSING_PERCENTAGES = [0.0, 0.3, 0.5]  # 0%, 30%, 50% missing

# Load simulated dataset
println("Loading simulated dataset...")
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

n_individuals = nrow(pheno_df)
n_omics = 10

println("  Individuals: ", n_individuals)
println("  SNPs: 1000 (927 after MAF filtering)")
println("  Omics: ", n_omics)
println()

# Store results
results = DataFrame(
    missing_pct = Float64[],
    n_missing_cells = Int[],
    accuracy_total = Float64[],
    accuracy_direct = Float64[],
    accuracy_indirect = Float64[]
)

for missing_pct in MISSING_PERCENTAGES
    println("-" ^ 70)
    println("Running with $(Int(missing_pct * 100))% missing omics data...")
    println("-" ^ 70)
    
    Random.seed!(SEED)
    
    # Setup temp directory
    benchmark_dir = mktempdir()
    
    # Create omics data with missing values
    omics_cols = vcat([:ID], [Symbol("omic$i") for i in 1:n_omics])
    omics_df = copy(pheno_df[:, omics_cols])
    
    # Introduce missing values randomly
    n_missing_cells = 0
    if missing_pct > 0
        # Allow missing values in omics columns
        for col in [Symbol("omic$i") for i in 1:n_omics]
            allowmissing!(omics_df, col)
        end
        
        for col in [Symbol("omic$i") for i in 1:n_omics]
            # For each omics column, randomly set some values to missing
            n_to_miss = round(Int, n_individuals * missing_pct)
            miss_idx = sample(1:n_individuals, n_to_miss, replace=false)
            omics_df[miss_idx, col] .= missing
            n_missing_cells += n_to_miss
        end
        println("  Introduced $n_missing_cells missing cells ($(n_omics) omics × $(round(Int, n_individuals * missing_pct)) individuals)")
    else
        println("  No missing data (baseline)")
    end
    
    omics_path = joinpath(benchmark_dir, "omics.csv")
    CSV.write(omics_path, omics_df; missingstring="NA")
    
    pheno_out_df = pheno_df[:, [:ID, :trait1]]
    pheno_out_path = joinpath(benchmark_dir, "phenotypes.csv")
    CSV.write(pheno_out_path, pheno_out_df; missingstring="NA")
    
    # Define layers
    layers = [
        Layer(layer_name="geno", data_path=[geno_path]),
        Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
        Layer(layer_name="phenotypes", data_path=pheno_out_path, missing_value="NA")
    ]
    
    # Define equations
    equations = [
        Equation(
            from_layer_name="geno",
            to_layer_name="omics",
            equation="omics = intercept + geno",
            omics_name=["omic$i" for i in 1:n_omics],
            method="BayesC",
            estimatePi=true
        ),
        Equation(
            from_layer_name="omics",
            to_layer_name="phenotypes",
            equation="phenotypes = intercept + omics",
            phenotype_name=["trait1"],
            method="BayesC",
            activation_function="linear"
        )
    ]
    
    # Run NNMM
    println("  Running MCMC (chain_length=$CHAIN_LENGTH, burnin=$BURNIN)...")
    result = runNNMM(layers, equations;
        chain_length=CHAIN_LENGTH,
        burnin=BURNIN,
        printout_frequency=CHAIN_LENGTH + 1,
        output_folder=joinpath(benchmark_dir, "output")
    )
    
    # Get EBV results
    ebv_df = result["EBV_NonLinear"]
    ebv_df.ID = string.(ebv_df.ID)
    pheno_df_copy = copy(pheno_df)
    pheno_df_copy.ID = string.(pheno_df_copy.ID)
    
    # Merge with true breeding values
    merged_df = innerjoin(ebv_df, pheno_df_copy[:, [:ID, :genetic_total, :genetic_direct, :genetic_indirect]], on=:ID)
    
    # Calculate accuracy metrics
    accuracy_total = cor(merged_df.EBV, merged_df.genetic_total)
    accuracy_direct = cor(merged_df.EBV, merged_df.genetic_direct)
    accuracy_indirect = cor(merged_df.EBV, merged_df.genetic_indirect)
    
    push!(results, (missing_pct, n_missing_cells, accuracy_total, accuracy_direct, accuracy_indirect))
    
    println("  Accuracy: cor(EBV, genetic_total) = $(round(accuracy_total, digits=4))")
    println()
    
    # Cleanup
    rm(benchmark_dir, recursive=true)
end

# Print summary
println()
println("=" ^ 70)
println("BENCHMARK RESULTS SUMMARY - MISSING OMICS DATA")
println("=" ^ 70)
println()
println("Configuration:")
println("  Seed: $SEED")
println("  Chain length: $CHAIN_LENGTH")
println("  Burnin: $BURNIN")
println("  Method: BayesC (both layers)")
println("  Activation: linear")
println()
println("Dataset (simulated_omics_data):")
println("  Individuals: $n_individuals")
println("  SNPs: 1000 (927 after MAF filtering)")
println("  Omics: $n_omics")
println("  Target heritability: 0.5 (20% direct, 80% indirect)")
println()
println("ACCURACY METRICS BY MISSING PERCENTAGE:")
println()
println("┌─────────────┬───────────────┬─────────────────┬────────────────┬──────────────────┐")
println("│ Missing %   │ Missing Cells │ cor(EBV,total)  │ cor(EBV,direct)│ cor(EBV,indirect)│")
println("├─────────────┼───────────────┼─────────────────┼────────────────┼──────────────────┤")
for row in eachrow(results)
    pct_str = lpad("$(Int(row.missing_pct * 100))%", 5)
    cells_str = lpad(string(row.n_missing_cells), 10)
    total_str = lpad(string(round(row.accuracy_total, digits=4)), 10)
    direct_str = lpad(string(round(row.accuracy_direct, digits=4)), 10)
    indirect_str = lpad(string(round(row.accuracy_indirect, digits=4)), 10)
    println("│ $pct_str       │ $cells_str    │ $total_str      │ $direct_str     │ $indirect_str       │")
end
println("└─────────────┴───────────────┴─────────────────┴────────────────┴──────────────────┘")
println()

# Calculate degradation
if nrow(results) > 1
    baseline = results[1, :accuracy_total]
    println("Accuracy degradation from baseline (0% missing):")
    for row in eachrow(results)
        if row.missing_pct > 0
            degradation = (baseline - row.accuracy_total) / baseline * 100
            println("  $(Int(row.missing_pct * 100))% missing: $(round(degradation, digits=2))% reduction")
        end
    end
    println()
end

println("=" ^ 70)
println()
println("PyNNMM should show similar accuracy patterns with missing data.")
println("Key: NNMM uses HMC to impute missing omics values during MCMC.")
println()

