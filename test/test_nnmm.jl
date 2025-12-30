using Test
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Random
using DelimitedFiles
using LinearAlgebra

@testset "NNMM Module" begin
    # Setup test data directory
    test_data_dir = joinpath(@__DIR__, "fixtures")
    mkpath(test_data_dir)
    
    @testset "Layer construction" begin
        layer = Layer(
            layer_name="test",
            data_path="test.csv",
            separator=',',
            header=true
        )
        @test layer.layer_name == "test"
        @test layer.data_path == "test.csv"
        @test layer.MAF == 0.01
        @test layer.center == true
    end
    
    @testset "Equation construction" begin
        eq = Equation(
            from_layer_name="geno",
            to_layer_name="omics",
            equation="omics = intercept + geno",
            omics_name=["o1"]
        )
        @test eq.from_layer_name == "geno"
        @test eq.to_layer_name == "omics"
        @test eq.method == "BayesC"
        @test eq.Pi == 0.0
        @test eq.estimatePi == true
    end
    
    @testset "NNMM workflow with test data" begin
        # Get genotype file path
        geno_path = dataset("genotypes0.csv")
        
        # Read genotypes to get individual IDs
        Random.seed!(123)
        geno = NNMM.nnmm_get_genotypes(geno_path)
        nind = length(geno.obsID)
        
        # Create synthetic omics data
        omics_df = DataFrame(ID=geno.obsID)
        for i in 1:3
            omics_df[!, Symbol("o$(i)")] = randn(nind)
        end
        omics_path = joinpath(test_data_dir, "test_omics.csv")
        CSV.write(omics_path, omics_df; missingstring="NA")
        
        # Create synthetic phenotype data
        pheno_df = DataFrame(
            ID=geno.obsID,
            y1=randn(nind)
        )
        pheno_path = joinpath(test_data_dir, "test_phenotypes.csv")
        CSV.write(pheno_path, pheno_df; missingstring="NA")
        
        # Define layers
        layers = [
            Layer(layer_name="geno", data_path=geno_path),
            Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
            Layer(layer_name="phenotypes", data_path=pheno_path, missing_value="NA")
        ]
        
        # Define equations
        equations = [
            Equation(
                from_layer_name="geno",
                to_layer_name="omics",
                equation="omics = intercept + geno",
                omics_name=["o1", "o2", "o3"],
                method="BayesC"
            ),
            Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + omics",
                phenotype_name=["y1"],
                method="BayesC"
            )
        ]
        
        # Run NNMM with short chain for testing
        result = runNNMM(layers, equations; chain_length=5, printout_frequency=10)
        
        # Test that results are returned
        @test result !== nothing
        @test haskey(result, "EBV_y1")
        
        # Test that EBV DataFrame has expected structure
        ebv_df = result["EBV_y1"]
        @test :ID in propertynames(ebv_df)
        @test :EBV in propertynames(ebv_df)
        
        # Cleanup test files
        rm(omics_path, force=true)
        rm(pheno_path, force=true)
    end
    
    @testset "nnmm_get_genotypes function" begin
        geno_path = dataset("genotypes0.csv")
        geno = NNMM.nnmm_get_genotypes(geno_path)
        
        @test typeof(geno) == NNMM.Genotypes
        @test length(geno.obsID) > 0
        @test geno.nMarkers > 0
        @test geno.nObs > 0
        @test size(geno.genotypes, 1) == geno.nObs
        @test size(geno.genotypes, 2) == geno.nMarkers
    end
end

