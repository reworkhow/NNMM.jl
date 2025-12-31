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
    - Latent nodes in the middle layer that are COMPLETELY MISSING for all individuals
    - A LINEAR activation function
    
    The NNMM framework can be used for genomic prediction:
    
        Genotypes ──► [Missing Latent Nodes] ──► Phenotype
                          (sampled)              (linear)
    
    Note: Current implementation requires at least 2 omics variables due to 
    matrix operations in HMC. Use 2 latent nodes that map to a single phenotype.
    =#
    
    # Setup test data directory
    data_dir = joinpath(@__DIR__, "fixtures", "bayesc_pattern")
    mkpath(data_dir)
    
    # Get genotype data
    geno_path = dataset("genotypes0.csv")
    Random.seed!(42)
    geno = NNMM.nnmm_get_genotypes(geno_path)
    nind = length(geno.obsID)
    
    @testset "BayesC via NNMM with latent layer" begin
        # Create middle layer with completely missing latent nodes
        # Note: Using 2 latent nodes to avoid scalar matrix issue in HMC
        omics_df = DataFrame(
            ID = geno.obsID,
            latent1 = fill(missing, nind),
            latent2 = fill(missing, nind)
        )
        o_path = joinpath(data_dir, "latent_missing.csv")
        CSV.write(o_path, omics_df; missingstring="NA")
        
        # Verify the file has all missing values
        omics_check = CSV.read(o_path, DataFrame; missingstring="NA")
        @test all(ismissing, omics_check.latent1)
        @test all(ismissing, omics_check.latent2)
        
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
        # 2. Latent → Phenotype (LINEAR activation)
        equations = [
            Equation(
                from_layer_name="geno",
                to_layer_name="latent",
                equation="latent = intercept + geno",
                omics_name=["latent1", "latent2"],
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
        @test typeof(result) <: Dict
        
        # Check for expected output keys (NNMM uses EBV_NonLinear for final phenotype EBV)
        @test haskey(result, "EBV_NonLinear")
        
        ebv_df = result["EBV_NonLinear"]
        @test nrow(ebv_df) == nind
        @test :ID in propertynames(ebv_df)
        @test :EBV in propertynames(ebv_df)
        
        # EBV values should be finite
        @test all(!isnan, ebv_df.EBV)
        @test all(!isinf, ebv_df.EBV)
    end
    
    @testset "Different Bayesian methods via NNMM" begin
        # Test that different methods work with the latent layer approach
        o_path = joinpath(data_dir, "latent_missing.csv")
        y_path = joinpath(data_dir, "phenotypes.csv")
        
        for method in ["BayesC", "BayesA"]
            @testset "Method: $method" begin
                layers = [
                    Layer(layer_name="geno", data_path=[geno_path]),
                    Layer(layer_name="latent", data_path=o_path, missing_value="NA"),
                    Layer(layer_name="phenotypes", data_path=y_path, missing_value="NA")
                ]
                
                estimate_pi = method in ["BayesB", "BayesC"]
                
                equations = [
                    Equation(
                        from_layer_name="geno",
                        to_layer_name="latent",
                        equation="latent = intercept + geno",
                        omics_name=["latent1", "latent2"],
                        method=method,
                        estimatePi=estimate_pi
                    ),
                    Equation(
                        from_layer_name="latent",
                        to_layer_name="phenotypes",
                        equation="phenotypes = intercept + latent",
                        phenotype_name=["y1"],
                        method=method,
                        activation_function="linear"
                    )
                ]
                
                result = runNNMM(layers, equations; chain_length=5, printout_frequency=100)
                
                @test result !== nothing
                @test haskey(result, "EBV_NonLinear")
                @test nrow(result["EBV_NonLinear"]) == nind
            end
        end
    end
    
    # Cleanup
    if isdir(data_dir)
        rm(data_dir, recursive=true)
    end
end
