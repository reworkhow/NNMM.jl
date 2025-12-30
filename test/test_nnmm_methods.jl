using Test
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Random

@testset "NNMM Bayesian Methods" begin
    # Setup test data directory
    data_dir = joinpath(@__DIR__, "fixtures", "methods")
    mkpath(data_dir)
    
    # Get genotype data
    geno_path = dataset("genotypes0.csv")
    Random.seed!(321)
    geno = NNMM.nnmm_get_genotypes(geno_path)
    nind = length(geno.obsID)
    
    # Create synthetic omics data
    omics_df = DataFrame(ID=geno.obsID)
    for i in 1:3
        omics_df[!, Symbol("o$(i)")] = randn(nind)
    end
    o_path = joinpath(data_dir, "omics.csv")
    CSV.write(o_path, omics_df; missingstring="NA")
    
    # Create phenotype data
    pheno_df = DataFrame(
        ID=geno.obsID,
        y1=randn(nind)
    )
    y_path = joinpath(data_dir, "phenotypes.csv")
    CSV.write(y_path, pheno_df; missingstring="NA")
    
    # Test different Bayesian methods
    # Note: Only testing methods that are compatible with NNMM framework
    bayesian_methods = ["BayesC", "BayesA", "BayesB", "BayesL", "RR-BLUP"]
    
    for method in bayesian_methods
        @testset "Method: $method" begin
            layers = [
                Layer(layer_name="geno", data_path=[geno_path]),
                Layer(layer_name="omics", data_path=o_path, missing_value="NA"),
                Layer(layer_name="phenotypes", data_path=y_path, missing_value="NA")
            ]
            
            # For BayesB and BayesC, enable Pi estimation
            estimate_pi = method in ["BayesB", "BayesC"]
            
            equations = [
                Equation(
                    from_layer_name="geno",
                    to_layer_name="omics",
                    equation="omics = intercept + geno",
                    omics_name=["o1", "o2", "o3"],
                    method=method,
                    estimatePi=estimate_pi
                ),
                Equation(
                    from_layer_name="omics",
                    to_layer_name="phenotypes",
                    equation="phenotypes = intercept + omics",
                    phenotype_name=["y1"],
                    method=method,
                    estimatePi=estimate_pi
                )
            ]
            
            # Verify equations have correct method
            @test equations[1].method == method
            @test equations[2].method == method
            
            # Run NNMM
            result = runNNMM(layers, equations; chain_length=5, printout_frequency=100)
            
            # Verify results
            @test result !== nothing
            @test haskey(result, "EBV_y1")
            
            ebv_df = result["EBV_y1"]
            @test nrow(ebv_df) > 0
            @test :ID in propertynames(ebv_df)
            @test :EBV in propertynames(ebv_df)
            
            # EBV values should be finite
            @test all(!isnan, ebv_df.EBV)
            @test all(!isinf, ebv_df.EBV)
        end
    end
    
    @testset "Pi estimation options" begin
        layers = [
            Layer(layer_name="geno", data_path=[geno_path]),
            Layer(layer_name="omics", data_path=o_path, missing_value="NA"),
            Layer(layer_name="phenotypes", data_path=y_path, missing_value="NA")
        ]
        
        # Test with fixed Pi (estimatePi=false)
        equations_fixed_pi = [
            Equation(
                from_layer_name="geno",
                to_layer_name="omics",
                equation="omics = intercept + geno",
                omics_name=["o1", "o2", "o3"],
                method="BayesC",
                Pi=0.95,
                estimatePi=false
            ),
            Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + omics",
                phenotype_name=["y1"],
                method="BayesC",
                Pi=0.95,
                estimatePi=false
            )
        ]
        
        @test equations_fixed_pi[1].estimatePi == false
        @test equations_fixed_pi[1].Pi == 0.95
        
        result = runNNMM(layers, equations_fixed_pi; chain_length=5, printout_frequency=100)
        @test result !== nothing
        @test haskey(result, "EBV_y1")
    end
    
    # Cleanup
    if isdir(data_dir)
        rm(data_dir, recursive=true)
    end
end

