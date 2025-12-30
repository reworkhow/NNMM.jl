using Test
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using DelimitedFiles
using LinearAlgebra
using Random

@testset "Full NNMM Workflow" begin
    # Setup test data directory
    data_dir = joinpath(@__DIR__, "fixtures", "nnmm_full")
    mkpath(data_dir)
    
    @testset "Setup test data" begin
        # Get genotype data
        geno_path = dataset("genotypes0.csv")
        @test isfile(geno_path)
        
        Random.seed!(123)
        geno = NNMM.nnmm_get_genotypes(geno_path)
        nind = length(geno.obsID)
        @test nind > 0
        
        # Create synthetic omics data with covariates
        omics_df = DataFrame(ID=geno.obsID)
        for i in 1:5
            omics_df[!, Symbol("o$(i)")] = randn(nind)
        end
        omics_df.x1 = randn(nind)
        omics_df.x2 = randn(nind)
        omics_df.x3 = rand(["m", "f"], nind)
        o_path = joinpath(data_dir, "omics.csv")
        CSV.write(o_path, omics_df; missingstring="NA")
        @test isfile(o_path)
        
        # Create phenotype data with covariates
        pheno_df = CSV.read(dataset("phenotypes.csv"), DataFrame)
        pheno_df.x4 = randn(nrow(pheno_df))
        pheno_df.x5 = rand(1:3, nrow(pheno_df))
        y_path = joinpath(data_dir, "phenotypes.csv")
        CSV.write(y_path, pheno_df[:, [:ID, :y1, :x4, :x5]]; missingstring="NA")
        @test isfile(y_path)
    end
    
    @testset "NNMM with pedigree and covariates" begin
        geno_path = dataset("genotypes0.csv")
        o_path = joinpath(data_dir, "omics.csv")
        y_path = joinpath(data_dir, "phenotypes.csv")
        
        # Load pedigree
        pedfile = dataset("pedigree.csv")
        pedigree = get_pedigree(pedfile, separator=",", header=true)
        @test typeof(pedigree) == NNMM.PedModule.Pedigree
        
        # Load GRM (for reference)
        GRM = readdlm(dataset("GRM.csv"), ',')
        grm_names = GRM[:, 1]
        GRM_matrix = Float32.(Matrix(GRM[:, 2:end])) + I * 0.0001
        @test size(GRM_matrix, 1) == size(GRM_matrix, 2)
        
        # Variance components
        G2 = 0.5
        G3 = 0.1
        
        # Define layers
        layers = [
            Layer(layer_name="geno", data_path=[geno_path]),
            Layer(layer_name="omics", data_path=o_path, missing_value="NA"),
            Layer(layer_name="phenotypes", data_path=y_path, missing_value="NA")
        ]
        @test length(layers) == 3
        @test layers[1].layer_name == "geno"
        @test layers[2].layer_name == "omics"
        @test layers[3].layer_name == "phenotypes"
        
        # Define equations with full model specification
        equations = [
            Equation(
                from_layer_name="geno",
                to_layer_name="omics",
                equation="omics = intercept + x1 + x2 + x3 + ID + geno",
                omics_name=["o1", "o2", "o3", "o4", "o5"],
                covariate=["x1", "x2"],
                random=[
                    (name="ID", pedigree=pedigree),
                    (name="x3",)
                ],
                method="BayesC"
            ),
            Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + ID + x4 + x5 + omics",
                phenotype_name=["y1"],
                covariate=["x4"],
                random=[
                    (name="x5", G=G3),
                    (name="ID", pedigree=pedigree)
                ],
                method="BayesC",
                activation_function="sigmoid"
            )
        ]
        @test length(equations) == 2
        @test equations[1].method == "BayesC"
        @test equations[2].activation_function == "sigmoid"
        
        # Run NNMM with short chain for testing
        result = runNNMM(layers, equations; chain_length=10, printout_frequency=100)
        
        # Verify results structure
        @test result !== nothing
        @test typeof(result) <: Dict
        
        # Check for expected output keys
        @test haskey(result, "EBV_y1")
        
        # Verify EBV DataFrame structure
        ebv_df = result["EBV_y1"]
        @test :ID in propertynames(ebv_df)
        @test :EBV in propertynames(ebv_df)
        @test nrow(ebv_df) > 0
        
        # Verify EBV values are numeric (not NaN or Inf)
        @test all(!isnan, ebv_df.EBV)
        @test all(!isinf, ebv_df.EBV)
    end
    
    # Cleanup test files
    @testset "Cleanup" begin
        if isdir(data_dir)
            rm(data_dir, recursive=true)
        end
        @test !isdir(data_dir)
    end
end

