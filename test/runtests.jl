using Test
using NNMM

@testset "NNMM.jl" begin
    # Core type tests
    include("test_types.jl")
    
    # Pedigree module tests
    include("test_pedigree.jl")
    
    # Genotype loading tests
    include("test_genotypes.jl")
    
    # NNMM basic workflow tests
    include("test_nnmm_basic.jl")
    
    # NNMM full workflow with pedigree and covariates
    include("test_nnmm_full.jl")
    
    # NNMM activation function tests
    include("test_nnmm_activation.jl")
    
    # NNMM missing data handling tests
    include("test_nnmm_missing.jl")
    
    # NNMM Bayesian method tests
    include("test_nnmm_methods.jl")
end
