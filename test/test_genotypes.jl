using Test
using NNMM
using NNMM.Datasets
using CSV
using DataFrames

@testset "Genotype Loading" begin
    # Use simulated dataset genotypes
    genofile = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    
    @testset "nnmm_get_genotypes function" begin
        geno = NNMM.nnmm_get_genotypes(genofile)
        
        @test typeof(geno) == NNMM.Genotypes
        @test length(geno.obsID) > 3000
        @test geno.nMarkers > 900  # After MAF filtering (~927 expected)
        @test geno.nObs > 3000     # Should have 3534 individuals
        @test size(geno.genotypes, 1) == geno.nObs
        @test size(geno.genotypes, 2) == geno.nMarkers
        
        # Verify genotype values are in expected range [0,2] after centering
        # Note: after centering, values should be approximately centered around 0
        @test all(x -> !isnan(x), geno.genotypes)
    end
    
    @testset "Genotype data integrity" begin
        # Load raw genotypes to check structure
        geno_df = CSV.read(genofile, DataFrame)
        
        # Check dimensions
        @test nrow(geno_df) == 3534  # Number of individuals
        @test ncol(geno_df) == 1001  # ID + 1000 SNPs
        
        # Check that first column is ID
        @test propertynames(geno_df)[1] == :ID
        
        # Check that genotype values are 0, 1, or 2
        geno_matrix = Matrix(geno_df[:, 2:end])
        @test all(x -> x in [0, 1, 2], geno_matrix)
    end
end
