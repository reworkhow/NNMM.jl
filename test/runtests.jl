using Test

@testset "NNMM.jl" begin
    # Core type tests
    include("test_types.jl")
    
    # Pedigree module tests
    include("test_pedigree.jl")
    
    # NNMM-specific tests
    include("test_nnmm.jl")
    
    # Bayesian Alphabet tests
    include("test_BayesianAlphabet.jl")
    include("test_BayesianAlphabet_deprecated.jl")
    
    # Genotype loading tests
    include("test_genotypes.jl")
end
