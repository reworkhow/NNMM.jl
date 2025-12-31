using Test
using NNMM

@testset "Core Types" begin
    @testset "Variance struct" begin
        # Test basic variance construction
        v = NNMM.Variance(1.0, 4.0, 0.5, true, false, false)
        @test v.val == 1.0
        @test v.df == 4.0
        @test v.scale == 0.5
        @test v.estimate_variance == true
        @test v.estimate_scale == false
        @test v.constraint == false
    end

    @testset "Layer struct" begin
        # Test Layer construction with keyword arguments
        layer = NNMM.Layer(
            layer_name="test_layer",
            data_path="test.csv"
        )
        @test layer.layer_name == "test_layer"
        @test layer.data_path == "test.csv"
        @test layer.separator == ','
        @test layer.header == true
        @test layer.quality_control == true
        @test layer.MAF == 0.01
        @test layer.center == true
    end

    @testset "Equation struct" begin
        # Test Equation construction
        eq = NNMM.Equation(
            from_layer_name="geno",
            to_layer_name="omics",
            equation="omics = intercept + geno",
            omics_name=["o1", "o2"]
        )
        @test eq.from_layer_name == "geno"
        @test eq.to_layer_name == "omics"
        @test eq.equation == "omics = intercept + geno"
        @test eq.method == "BayesC"
        @test eq.activation_function == "linear"
        @test eq.estimatePi == true
    end

    @testset "Phenotypes struct" begin
        obsID = ["ind1", "ind2", "ind3"]
        featureID = ["y1", "y2"]
        data = rand(3, 2)
        pheno = NNMM.Phenotypes(obsID, featureID, 3, 2, data)
        @test pheno.obsID == obsID
        @test pheno.featureID == featureID
        @test pheno.nObs == 3
        @test pheno.nPheno == 2
        @test pheno.nFeatures == 2
    end

    @testset "Omics struct" begin
        obsID = ["ind1", "ind2", "ind3"]
        featureID = ["o1", "o2", "o3"]
        data = rand(3, 3)
        omics = NNMM.Omics(obsID, featureID, 3, 3, data)
        @test omics.obsID == obsID
        @test omics.featureID == featureID
        @test omics.nObs == 3
        @test omics.nFeatures == 3
        @test omics.nMarkers == 3  # Should equal nFeatures
    end
end

@testset "build_model function" begin
    # Test single-trait model building
    model_eq = "y1 = intercept + age + sex"
    R = 1.0
    model = NNMM.build_model(model_eq, R)
    @test model.nModels == 1
    @test :y1 in model.lhsVec

    # Test multi-trait model building
    model_eq_mt = "y1 = intercept + age
                   y2 = intercept + age"
    R_mt = [1.0 0.5; 0.5 1.0]
    model_mt = NNMM.build_model(model_eq_mt, R_mt)
    @test model_mt.nModels == 2
    @test :y1 in model_mt.lhsVec
    @test :y2 in model_mt.lhsVec
end

