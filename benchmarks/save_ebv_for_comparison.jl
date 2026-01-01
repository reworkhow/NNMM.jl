#!/usr/bin/env julia
#=
Save EBVs from NNMM.jl for cross-package comparison with PyNNMM

Usage:
    julia --project=. benchmarks/save_ebv_for_comparison.jl
=#

using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random

println("=" ^ 70)
println("NNMM.jl - Saving EBVs for Cross-Package Comparison")
println("=" ^ 70)

# Configuration - matches PyNNMM benchmark
const SEED = 42
const CHAIN_LENGTH = 1000
const BURNIN = 200

Random.seed!(SEED)

# Load simulated dataset
println("Loading simulated dataset...")
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

println("  Individuals: ", nrow(pheno_df))
println("  SNPs: 1000")
println("  Omics: 10")
println()

# Setup temp directory
benchmark_dir = mktempdir()

# Create data files for NNMM
omics_cols = vcat([:ID], [Symbol("omic$i") for i in 1:10])
omics_df = pheno_df[:, omics_cols]
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
        omics_name=["omic$i" for i in 1:10],
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
println("Running MCMC (chain_length=$CHAIN_LENGTH, burnin=$BURNIN, seed=$SEED)...")
result = runNNMM(layers, equations;
    chain_length=CHAIN_LENGTH,
    burnin=BURNIN,
    printout_frequency=CHAIN_LENGTH + 1,
    output_folder=joinpath(benchmark_dir, "output")
)

# Get EBV results
ebv_df = result["EBV_NonLinear"]
ebv_df.ID = string.(ebv_df.ID)

# Get EPV results
epv_df = result["EPV_NonLinear"]
epv_df.ID = string.(epv_df.ID)

# Save EBVs to shared location
ebv_output_path = joinpath(@__DIR__, "ebv_julia.csv")
CSV.write(ebv_output_path, ebv_df)
println()
println("✅ EBVs saved to: $ebv_output_path")
println("   Individuals: $(nrow(ebv_df))")
println("   Mean EBV: $(round(mean(ebv_df.EBV), digits=4))")
println("   Std EBV:  $(round(std(ebv_df.EBV), digits=4))")

# Save EPVs to shared location
epv_output_path = joinpath(@__DIR__, "epv_julia.csv")
CSV.write(epv_output_path, epv_df)
println()
println("✅ EPVs saved to: $epv_output_path")
println("   Individuals: $(nrow(epv_df))")
println("   Mean EPV: $(round(mean(epv_df.EBV), digits=4))")
println("   Std EPV:  $(round(std(epv_df.EBV), digits=4))")

# Cleanup temp directory
rm(benchmark_dir, recursive=true)

println()
println("Now run PyNNMM benchmark and then compare_ebv.py")

