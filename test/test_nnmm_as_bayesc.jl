using Test
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random

@testset "NNMM as Traditional BayesC" begin
    #=
    This test demonstrates how NNMM generalizes traditional genomic prediction models.
    
    By using:
    - A single node in the middle layer that is COMPLETELY MISSING for all individuals
    - A LINEAR activation function
    
    The NNMM framework reduces to a standard single-trait BayesC model:
    
        Genotypes ──► [Missing Latent Node] ──► Phenotype
                          (sampled)           (linear)
    
    This is mathematically equivalent to:
        y = Xβ + Zα + e
    
    where α are the marker effects sampled using BayesC.
    =#
    
    # Setup test data directory
    data_dir = joinpath(@__DIR__, "fixtures", "bayesc_pattern")
    mkpath(data_dir)
    
    # Get genotype data
    geno_path = dataset("genotypes0.csv")
    Random.seed!(42)
    geno = NNMM.nnmm_get_genotypes(geno_path)
    nind = length(geno.obsID)
    
    @testset "Single-trait BayesC via NNMM" begin
        # Create middle layer with ONE completely missing node
        # All individuals have NA for this latent trait
        omics_df = DataFrame(
            ID = geno.obsID,
            latent = fill(missing, nind)  # Completely missing for all individuals
        )
        o_path = joinpath(data_dir, "latent_missing.csv")
        CSV.write(o_path, omics_df; missingstring="NA")
        
        # Verify the file has all missing values
        omics_check = CSV.read(o_path, DataFrame; missingstring="NA")
        @test all(ismissing, omics_check.latent)
        
        # Create phenotype data (simulated trait)
        # Simulate: y = intercept + genetic_value + noise
        genetic_value = randn(nind)
        noise = randn(nind) * 0.5
        y_values = 10.0 .+ genetic_value .+ noise
        
        pheno_df = DataFrame(
            ID = geno.obsID,
            y1 = y_values
        )
        y_path = joinpath(data_dir, "phenotypes.csv")
        CSV.write(y_path, pheno_df; missingstring="NA")
        
        # Define 3-layer network
        layers = [
            Layer(layer_name="geno", data_path=[geno_path]),
            Layer(layer_name="latent", data_path=o_path, missing_value="NA"),
            Layer(layer_name="phenotypes", data_path=y_path, missing_value="NA")
        ]
        
        # Define equations:
        # 1. Genotypes → Latent (BayesC marker regression)
        # 2. Latent → Phenotype (LINEAR activation = standard regression)
        equations = [
            Equation(
                from_layer_name="geno",
                to_layer_name="latent",
                equation="latent = intercept + geno",
                omics_name=["latent"],
                method="BayesC",
                estimatePi=true
            ),
            Equation(
                from_layer_name="latent",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + latent",
                phenotype_name=["y1"],
                method="BayesC",
                activation_function="linear"  # Key: linear activation
            )
        ]
        
        # Verify configuration
        @test equations[1].method == "BayesC"
        @test equations[2].activation_function == "linear"
        
        # Run NNMM (short chain for testing)
        result = runNNMM(layers, equations; chain_length=10, printout_frequency=100)
        
        # Verify results
        @test result !== nothing
        @test haskey(result, "EBV_y1")
        
        ebv_df = result["EBV_y1"]
        @test nrow(ebv_df) == nind
        @test :ID in propertynames(ebv_df)
        @test :EBV in propertynames(ebv_df)
        
        # EBV values should be finite and reasonable
        @test all(!isnan, ebv_df.EBV)
        @test all(!isinf, ebv_df.EBV)
        
        # EBVs should have some variance (not all identical)
        @test std(ebv_df.EBV) > 0
    end
    
    @testset "Multi-trait via multiple missing nodes" begin
        # This shows how to do multi-trait BayesC:
        # Use multiple missing nodes in the middle layer
        
        # Create middle layer with TWO completely missing nodes
        omics_df = DataFrame(
            ID = geno.obsID,
            latent1 = fill(missing, nind),
            latent2 = fill(missing, nind)
        )
        o_path = joinpath(data_dir, "latent_multi.csv")
        CSV.write(o_path, omics_df; missingstring="NA")
        
        # Create phenotype data with two traits
        pheno_df = DataFrame(
            ID = geno.obsID,
            y1 = randn(nind) .+ 10.0,
            y2 = randn(nind) .+ 5.0
        )
        y_path = joinpath(data_dir, "phenotypes_multi.csv")
        CSV.write(y_path, pheno_df; missingstring="NA")
        
        # Define network
        layers = [
            Layer(layer_name="geno", data_path=[geno_path]),
            Layer(layer_name="latent", data_path=o_path, missing_value="NA"),
            Layer(layer_name="phenotypes", data_path=y_path, missing_value="NA")
        ]
        
        equations = [
            Equation(
                from_layer_name="geno",
                to_layer_name="latent",
                equation="latent = intercept + geno",
                omics_name=["latent1", "latent2"],
                method="BayesC"
            ),
            Equation(
                from_layer_name="latent",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + latent",
                phenotype_name=["y1", "y2"],
                method="BayesC",
                activation_function="linear"
            )
        ]
        
        result = runNNMM(layers, equations; chain_length=10, printout_frequency=100)
        
        @test result !== nothing
        @test haskey(result, "EBV_y1")
        @test haskey(result, "EBV_y2")
    end
    
    # Cleanup
    if isdir(data_dir)
        rm(data_dir, recursive=true)
    end
end

