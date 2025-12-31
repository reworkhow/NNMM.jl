using Test
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Random

@testset "NNMM Missing Data Handling" begin
    # Use simulated omics dataset
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    
    # Setup test data directory
    data_dir = joinpath(@__DIR__, "fixtures", "missing")
    mkpath(data_dir)
    
    # Load phenotype data
    pheno_df = CSV.read(pheno_path, DataFrame)
    nind = nrow(pheno_df)
    
    @testset "Missing values in omics layer" begin
        # Create omics data with missing values
        omics_df = pheno_df[:, [:ID, :omic1, :omic2, :omic3]]
        
        # Introduce missing values (10% missing in each column)
        Random.seed!(123)
        for col in [:omic1, :omic2, :omic3]
            omics_df[!, col] = Vector{Union{Float64, Missing}}(omics_df[!, col])
            missing_idx = rand(1:nind, nind ÷ 10)
            omics_df[missing_idx, col] .= missing
        end
        
        o_path = joinpath(data_dir, "omics_missing.csv")
        CSV.write(o_path, omics_df; missingstring="NA")
        
        # Verify missing values were introduced
        omics_check = CSV.read(o_path, DataFrame; missingstring="NA")
        @test sum(ismissing.(omics_check.omic1)) > 0
        
        # Create complete phenotype data
        pheno_out_df = pheno_df[:, [:ID, :trait1]]
        y_path = joinpath(data_dir, "phenotypes_complete.csv")
        CSV.write(y_path, pheno_out_df; missingstring="NA")
        
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
                omics_name=["omic1", "omic2", "omic3"],
                method="BayesC"
            ),
            Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + omics",
                phenotype_name=["trait1"],
                method="BayesC"
            )
        ]
        
        # Run NNMM - should handle missing omics values via HMC sampling
        result = runNNMM(layers, equations; chain_length=5, printout_frequency=100)
        
        @test result !== nothing
        @test haskey(result, "EBV_NonLinear")
        @test nrow(result["EBV_NonLinear"]) > 3000
    end
    
    @testset "Missing values in phenotype layer" begin
        # Create complete omics data
        omics_df = pheno_df[:, [:ID, :omic1, :omic2, :omic3]]
        o_path = joinpath(data_dir, "omics_complete.csv")
        CSV.write(o_path, omics_df; missingstring="NA")
        
        # Create phenotype data with missing values
        pheno_out_df = pheno_df[:, [:ID, :trait1]]
        pheno_out_df[!, :trait1] = Vector{Union{Float64, Missing}}(pheno_out_df[!, :trait1])
        
        # Make 5% of phenotypes missing
        Random.seed!(456)
        missing_idx = rand(1:nind, nind ÷ 20)
        pheno_out_df[missing_idx, :trait1] .= missing
        
        y_path = joinpath(data_dir, "phenotypes_missing.csv")
        CSV.write(y_path, pheno_out_df; missingstring="NA")
        
        # Verify missing values
        @test sum(ismissing.(pheno_out_df.trait1)) > 0
        
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
                omics_name=["omic1", "omic2", "omic3"],
                method="BayesC"
            ),
            Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + omics",
                phenotype_name=["trait1"],
                method="BayesC"
            )
        ]
        
        # Run NNMM - should handle missing phenotype values
        result = runNNMM(layers, equations; chain_length=5, printout_frequency=100)
        
        @test result !== nothing
        @test haskey(result, "EBV_NonLinear")
        # EBV should be available for all individuals, including those with missing phenotypes
        @test nrow(result["EBV_NonLinear"]) > 3000
    end
    
    # Cleanup
    rm(data_dir, recursive=true, force=true)
end
