using DataFrames
using CSV
using Random
using DelimitedFiles

@testset "Grouped omics marker classes" begin
    Random.seed!(2026)

    mktempdir() do tmpdir
        ids = string.(1:24)

        geno = DataFrame(ID=ids)
        for j in 1:8
            geno[!, "m$j"] = rand(0:2, length(ids))
        end
        geno_path = joinpath(tmpdir, "geno.csv")
        CSV.write(geno_path, geno)

        omics = DataFrame(ID=ids)
        omics[!, "o1a"] = randn(length(ids))
        omics[!, "o1b"] = randn(length(ids))
        omics[!, "o2a"] = randn(length(ids))
        omics[!, "o2b"] = randn(length(ids))
        omics[!, "o3a"] = randn(length(ids))
        omics[!, "o3b"] = randn(length(ids))
        omics_path = joinpath(tmpdir, "omics.csv")
        CSV.write(omics_path, omics; missingstring="NA")

        pheno = DataFrame(
            ID=ids,
            age=collect(21.0:44.0),
            y=0.8 .* omics.o1a .- 0.4 .* omics.o2b .+ 0.2 .* omics.o3a .+
              0.03 .* collect(21.0:44.0) .+ 0.1 .* randn(length(ids)),
        )
        pheno_path = joinpath(tmpdir, "pheno.csv")
        CSV.write(pheno_path, pheno; missingstring="NA")

        groups = Dict(
            "omics1" => ["o1a", "o1b"],
            "omics2" => ["o2a", "o2b"],
            "omics3" => ["o3a", "o3b"],
        )
        class_priors = Dict(
            "omics1" => (method="BayesC", Pi=0.75, estimatePi=false),
            "omics2" => (method="BayesC", Pi=0.25, estimatePi=false),
            "omics3" => (method="BayesC", Pi=0.50, estimatePi=false),
            "geno" => (method="BayesC", Pi=0.0, estimatePi=false),
        )

        layers = [
            Layer(layer_name="geno", data_path=[geno_path]),
            Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
            Layer(layer_name="phenotypes", data_path=pheno_path, missing_value="NA"),
        ]
        equations = [
            Equation(
                from_layer_name="geno",
                to_layer_name="omics",
                equation="omics = intercept + geno",
                traits=vcat(groups["omics1"], groups["omics2"], groups["omics3"]),
                method="BayesC",
                omics_groups=groups,
            ),
            Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + age + omics1 + omics2 + omics3 + geno",
                traits=["y"],
                covariate=["age"],
                class_priors=class_priors,
                omics_groups=groups,
                activation_function="linear",
            ),
        ]

        result = runNNMM(
            layers,
            equations;
            chain_length=4,
            burnin=0,
            output_samples_frequency=1,
            output_prediction_frequency=1,
            printout_model_info=false,
            printout_frequency=10^9,
            output_folder=joinpath(tmpdir, "out"),
            seed=2026,
        )

        @test haskey(result, "EBV_NonLinear")
        @test haskey(result, "EPV_Output_NonLinear")
        @test haskey(result, "fixed effects 1->2")
        @test haskey(result, "fixed effects 2->3")
        @test Set(result["fixed effects 2->3"].Effect) == Set(["intercept", "age"])
        @test all(isfinite, result["fixed effects 2->3"].Estimate)
        @test isfile(joinpath(tmpdir, "out", "fixed_effects_1_to_2.csv"))
        @test isfile(joinpath(tmpdir, "out", "fixed_effects_2_to_3.csv"))
        @test haskey(result, "marker effects omics3")
        @test nrow(result["EBV_NonLinear"]) == length(ids)

        @testset "BayesA/B locus-specific variances with missing omics" begin
            # Mix locus-specific and shared variances in the same model, with
            # multiple features per class so an incorrect matrix index fails.
            mixed_priors = Dict(
                "omics1" => (method="BayesA", estimatePi=false),
                "omics2" => (method="BayesB", Pi=0.25, estimatePi=false),
                "omics3" => (method="BayesC", Pi=0.50, estimatePi=false),
            )
            omics_missing = allowmissing(copy(omics))
            omics_missing[1:3, :o1a] .= missing
            omics_missing[4:6, :o2b] .= missing
            omics_missing[21:24, Not(:ID)] .= missing
            pheno_missing = allowmissing(copy(pheno), :y)
            pheno_missing[21:24, :y] .= missing
            missing_omics_path = joinpath(tmpdir, "omics_missing.csv")
            missing_pheno_path = joinpath(tmpdir, "pheno_missing.csv")
            CSV.write(missing_omics_path, omics_missing; missingstring="NA")
            CSV.write(missing_pheno_path, pheno_missing; missingstring="NA")

            mixed_layers = [
                Layer(layer_name="geno", data_path=[geno_path]),
                Layer(layer_name="omics", data_path=missing_omics_path, missing_value="NA"),
                Layer(layer_name="phenotypes", data_path=missing_pheno_path, missing_value="NA"),
            ]
            mixed_equations = [
                Equation(
                    from_layer_name="geno", to_layer_name="omics",
                    equation="omics = intercept + geno",
                    traits=vcat(groups["omics1"], groups["omics2"], groups["omics3"]),
                ),
                Equation(
                    from_layer_name="omics", to_layer_name="phenotypes",
                    equation="phenotypes = intercept + age + omics1 + omics2 + omics3",
                    traits=["y"], covariate=["age"],
                    class_priors=mixed_priors, omics_groups=groups,
                ),
            ]
            mixed_outdir = joinpath(tmpdir, "out_mixed_methods")
            mixed_result = runNNMM(
                mixed_layers,
                mixed_equations;
                chain_length=4,
                burnin=0,
                output_samples_frequency=1,
                output_prediction_frequency=1,
                printout_model_info=false,
                printout_frequency=10^9,
                output_folder=mixed_outdir,
                seed=2026,
            )

            @test nrow(mixed_result["EBV_NonLinear"]) == length(ids)
            @test nrow(mixed_result["EPV_Output_NonLinear"]) == length(ids)
            @test all(isfinite, mixed_result["EBV_NonLinear"].EBV)
            @test all(isfinite, mixed_result["EPV_Output_NonLinear"].EPV)
            for group in ("omics1", "omics2", "omics3")
                effects = mixed_result["marker effects " * group]
                @test Set(effects.Marker_ID) == Set(groups[group])
                @test all(isfinite, effects.Estimate)
            end

            variances = readdlm(joinpath(mixed_outdir, "MCMC_samples_layer2_effect_variance.txt"), ',')
            @test size(variances) == (4, 6)
            @test all(x -> isfinite(x) && x > 0, variances)
            @test any(variances[:, 1] .!= variances[:, 2]) # BayesA: per feature
            @test any(variances[:, 3] .!= variances[:, 4]) # BayesB: per feature
            @test variances[:, 5] == variances[:, 6]      # BayesC: shared within class
        end
    end
end
