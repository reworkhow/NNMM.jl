using Test
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Random

@testset "NNMM Activation Functions" begin
    # Setup test data directory
    data_dir = joinpath(@__DIR__, "fixtures", "activation")
    mkpath(data_dir)
    
    # Prepare test data once for all activation function tests
    geno_path = dataset("genotypes0.csv")
    Random.seed!(456)
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
    
    # Test each activation function
    for activation in ["linear", "sigmoid"]
        @testset "Activation: $activation" begin
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
                    method="BayesC",
                    activation_function=activation
                )
            ]
            
            # Verify equation has correct activation function
            @test equations[2].activation_function == activation
            
            # Run NNMM
            result = runNNMM(layers, equations; chain_length=5, printout_frequency=100)
            
            # Verify results
            @test result !== nothing
            @test haskey(result, "EBV_y1")
            
            ebv_df = result["EBV_y1"]
            @test nrow(ebv_df) > 0
            @test all(!isnan, ebv_df.EBV)
        end
    end
    
    # Cleanup
    if isdir(data_dir)
        rm(data_dir, recursive=true)
    end
end

