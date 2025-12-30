using Test
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Random

@testset "NNMM Missing Data Handling" begin
    # Setup test data directory
    data_dir = joinpath(@__DIR__, "fixtures", "missing")
    mkpath(data_dir)
    
    # Get genotype data
    geno_path = dataset("genotypes0.csv")
    Random.seed!(789)
    geno = NNMM.nnmm_get_genotypes(geno_path)
    nind = length(geno.obsID)
    
    @testset "Missing values in omics layer" begin
        # Create omics data with missing values
        omics_df = DataFrame(ID=geno.obsID)
        for i in 1:3
            col_data = randn(nind)
            # Introduce missing values (10% missing)
            missing_idx = rand(1:nind, max(1, nind ÷ 10))
            col_data_union = Vector{Union{Float64, Missing}}(col_data)
            col_data_union[missing_idx] .= missing
            omics_df[!, Symbol("o$(i)")] = col_data_union
        end
        o_path = joinpath(data_dir, "omics_missing.csv")
        CSV.write(o_path, omics_df; missingstring="NA")
        @test isfile(o_path)
        
        # Create phenotype data (complete)
        pheno_df = DataFrame(
            ID=geno.obsID,
            y1=randn(nind)
        )
        y_path = joinpath(data_dir, "phenotypes_complete.csv")
        CSV.write(y_path, pheno_df; missingstring="NA")
        
        # Define layers
        layers = [
            Layer(layer_name="geno", data_path=[geno_path]),
            Layer(layer_name="omics", data_path=o_path, missing_value="NA"),
            Layer(layer_name="phenotypes", data_path=y_path, missing_value="NA")
        ]
        
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
        
        # Run NNMM - should handle missing omics values
        result = runNNMM(layers, equations; chain_length=5, printout_frequency=100)
        
        @test result !== nothing
        @test haskey(result, "EBV_y1")
        @test nrow(result["EBV_y1"]) > 0
    end
    
    @testset "Missing values in phenotype layer" begin
        # Create complete omics data
        omics_df = DataFrame(ID=geno.obsID)
        for i in 1:3
            omics_df[!, Symbol("o$(i)")] = randn(nind)
        end
        o_path = joinpath(data_dir, "omics_complete.csv")
        CSV.write(o_path, omics_df; missingstring="NA")
        
        # Create phenotype data with missing values
        y1_data = randn(nind)
        y1_data_union = Vector{Union{Float64, Missing}}(y1_data)
        # Make 5% of phenotypes missing
        missing_idx = rand(1:nind, max(1, nind ÷ 20))
        y1_data_union[missing_idx] .= missing
        
        pheno_df = DataFrame(
            ID=geno.obsID,
            y1=y1_data_union
        )
        y_path = joinpath(data_dir, "phenotypes_missing.csv")
        CSV.write(y_path, pheno_df; missingstring="NA")
        @test isfile(y_path)
        
        # Define layers
        layers = [
            Layer(layer_name="geno", data_path=[geno_path]),
            Layer(layer_name="omics", data_path=o_path, missing_value="NA"),
            Layer(layer_name="phenotypes", data_path=y_path, missing_value="NA")
        ]
        
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
        
        # Run NNMM - should handle missing phenotype values
        result = runNNMM(layers, equations; chain_length=5, printout_frequency=100)
        
        @test result !== nothing
        @test haskey(result, "EBV_y1")
        # EBV should be available for all individuals, including those with missing phenotypes
        @test nrow(result["EBV_y1"]) > 0
    end
    
    # Cleanup
    if isdir(data_dir)
        rm(data_dir, recursive=true)
    end
end

