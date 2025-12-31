using Test
using NNMM
using NNMM.Datasets
using DataFrames
using CSV

@testset "Pedigree Module" begin
    # Use simulated dataset pedigree
    pedfile = Datasets.dataset("pedigree.txt", dataset_name="simulated_omics_data")
    
    @testset "get_pedigree basic functionality" begin
        pedigree = get_pedigree(pedfile, separator=",", header=true)
        
        # Test that pedigree object is created
        @test typeof(pedigree) == NNMM.PedModule.Pedigree
        
        # Test that IDs are populated
        @test length(pedigree.IDs) > 0
        
        # Test that idMap is populated
        @test length(pedigree.idMap) > 0
        
        # Simulated pedigree has 6473 animals
        @test length(pedigree.IDs) >= 3000
    end
    
    @testset "get_pedigree from DataFrame" begin
        df = CSV.read(pedfile, DataFrame, header=true)
        pedigree = get_pedigree(df, separator=",", header=true)
        
        @test typeof(pedigree) == NNMM.PedModule.Pedigree
        @test length(pedigree.IDs) > 0
    end
    
    @testset "get_info function" begin
        pedigree = get_pedigree(pedfile, separator=",", header=true)
        # get_info should not error
        @test_nowarn NNMM.PedModule.get_info(pedigree)
    end
end
