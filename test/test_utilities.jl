using Test
using NNMM
using LinearAlgebra

@testset "Utility Functions" begin
    
    @testset "print_matrix_truncated" begin
        # Test small matrix (should print fully)
        small = rand(3, 3)
        # Capture output or just verify it doesn't throw
        @test begin
            io = IOBuffer()
            # Function prints to stdout, so we just verify no error
            NNMM.print_matrix_truncated(small)
            true
        end
        
        # Test large matrix (should truncate)
        large = rand(10, 10)
        @test begin
            NNMM.print_matrix_truncated(large)
            true
        end
        
        # Test with custom max_size
        @test begin
            NNMM.print_matrix_truncated(large; max_size=3)
            true
        end
        
        # Test with different digits
        @test begin
            NNMM.print_matrix_truncated(small; digits=6)
            true
        end
    end
    
    @testset "Gibbs Sampler Functions" begin
        # Test basic Gibbs functionality with simple inputs
        # Gibbs(A, x, b, vare) - Gibbs sampler with residual variance
        
        n = 5
        A = randn(n, n)
        A = A' * A + I  # Make positive definite
        x = randn(n)
        b = A * x + 0.1 * randn(n)
        vare = 1.0
        
        # Test that Gibbs function exists and can be called
        @test hasmethod(NNMM.Gibbs, Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}, Float64})
        
        # Call Gibbs and verify it modifies x
        x_new = copy(x)
        NNMM.Gibbs(A, x_new, b, vare)
        @test length(x_new) == n
        @test all(!isnan, x_new)
    end
    
    @testset "Variance struct operations" begin
        # Test Variance struct
        v = NNMM.Variance(1.0, 4.0, 0.5, true, false, false)
        
        @test v.val == 1.0
        @test v.df == 4.0
        @test v.scale == 0.5
        @test v.estimate_variance == true
        @test v.estimate_scale == false
        @test v.constraint == false
        
        # Test with constraint=true
        v2 = NNMM.Variance(2.0, 5.0, 1.0, true, true, true)
        @test v2.constraint == true
    end
    
    @testset "ModelTerm struct" begin
        # ModelTerm is constructed internally during model building
        # We verify the struct type exists
        @test isdefined(NNMM, :ModelTerm)
        
        # Verify key fields exist in the struct definition
        @test hasfield(NNMM.ModelTerm, :trmStr)
        @test hasfield(NNMM.ModelTerm, :nFactors)
        @test hasfield(NNMM.ModelTerm, :nLevels)
    end
    
    @testset "ResVar struct" begin
        # ResVar stores residual variance with R0 and RiDict
        @test isdefined(NNMM, :ResVar)
        @test hasfield(NNMM.ResVar, :R0)
        @test hasfield(NNMM.ResVar, :RiDict)
        
        # Test construction
        R0 = ones(3, 3)
        RiDict = Dict{BitArray{1}, Array{Float64,2}}()
        rv = NNMM.ResVar(R0, RiDict)
        @test size(rv.R0) == (3, 3)
    end
    
    @testset "RandomEffect struct" begin
        # RandomEffect stores random effect information
        @test isdefined(NNMM, :RandomEffect)
        @test hasfield(NNMM.RandomEffect, :term_array)
        @test hasfield(NNMM.RandomEffect, :Gi)
        @test hasfield(NNMM.RandomEffect, :randomType)
    end
end

@testset "Dataset Loading" begin
    using NNMM.Datasets
    
    @testset "dataset function" begin
        # Test loading built-in datasets
        geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
        @test isfile(geno_path)
        
        pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
        @test isfile(pheno_path)
        
        ped_path = Datasets.dataset("pedigree.txt", dataset_name="simulated_omics_data")
        @test isfile(ped_path)
    end
    
    @testset "Example datasets" begin
        # Test loading example folder datasets (need dataset_name="example")
        example_geno = Datasets.dataset("genotypes.txt", dataset_name="example")
        @test isfile(example_geno)
        
        example_pheno = Datasets.dataset("phenotypes.txt", dataset_name="example")
        @test isfile(example_pheno)
        
        example_ped = Datasets.dataset("pedigree.txt", dataset_name="example")
        @test isfile(example_ped)
        
        # Test root-level datasets (without subdirectory)
        geno_csv = Datasets.dataset("genotypes.csv")
        @test isfile(geno_csv)
    end
end

