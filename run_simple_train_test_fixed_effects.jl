using NNMM, CSV, DataFrames

# Edit these settings.
const TRAIT = :trait1
const OMICS = [:omic1, :omic2, :omic3]
const FIXED_EFFECTS = [:age]  # Numeric covariates; dummy-code categories as 0/1.
const CHAIN_LENGTH = 5_000
const BURNIN = 1_000

# Usage:
# julia --project=. run_simple_train_test_fixed_effects.jl \
#   genotype.csv omics.csv phenotype.csv train_ids.csv test_ids.csv results_folder
length(ARGS) == 6 || error("Expected six input arguments")
geno_file, omics_file, pheno_file, train_file, test_file, results_folder = ARGS
ispath(results_folder) && error("Results folder already exists: $results_folder")

geno = CSV.read(geno_file, DataFrame)
omics = CSV.read(omics_file, DataFrame)
pheno = CSV.read(pheno_file, DataFrame)
train_ids = string.(CSV.read(train_file, DataFrame).ID)
test_ids = string.(CSV.read(test_file, DataFrame).ID)
ids = vcat(train_ids, test_ids)

# Put all three files in exactly the same train-then-test ID order.
function align_by_id(df, ids)
    df.ID = string.(df.ID)
    row = Dict(id => i for (i, id) in enumerate(df.ID))
    all(id -> haskey(row, id), ids) || error("An ID is missing from an input file")
    return copy(df[[row[id] for id in ids], :])
end
geno = align_by_id(geno, ids)
omics = align_by_id(omics, ids)
pheno = align_by_id(pheno, ids)

# Mask testing phenotypes. Fixed-effect columns remain observed for all IDs.
allowmissing!(pheno, TRAIT)
pheno[(length(train_ids) + 1):end, TRAIT] .= missing

result = mktempdir() do prepared
    geno_path = joinpath(prepared, "genotype.csv")
    omics_path = joinpath(prepared, "omics.csv")
    pheno_path = joinpath(prepared, "phenotype.csv")
    CSV.write(geno_path, geno)                         # Genotype missing value: 9.0
    CSV.write(omics_path, omics; missingstring="NA")
    CSV.write(pheno_path, pheno; missingstring="NA")

    fixed_rhs = join(vcat(["intercept"], string.(FIXED_EFFECTS), ["omics"]), " + ")
    layers = [
        Layer(layer_name="geno", data_path=[geno_path], missing_value=9.0),
        Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
        Layer(layer_name="phenotypes", data_path=pheno_path, missing_value="NA"),
    ]
    equations = [
        Equation(
            from_layer_name="geno", to_layer_name="omics",
            equation="omics = intercept + geno", traits=string.(OMICS),
            method="BayesC", estimatePi=true,
        ),
        Equation(
            from_layer_name="omics", to_layer_name="phenotypes",
            equation="phenotypes = " * fixed_rhs,
            traits=[string(TRAIT)], covariate=string.(FIXED_EFFECTS),
            method="BayesC", estimatePi=true, activation_function="linear",
        ),
    ]

    runNNMM(
        layers, equations;
        chain_length=CHAIN_LENGTH,
        burnin=BURNIN,
        output_samples_frequency=10,
        output_prediction_frequency=10,
        seed=2026,
        output_folder=results_folder,
    )
end

# Fixed-effect estimates.
CSV.write(
    joinpath(results_folder, "fixed_effects_2_to_3.csv"),
    result["fixed effects 2->3"],
)

# EPV estimates for testing individuals, kept in test-ID order.
epv = copy(result["EPV_Output_NonLinear"])
epv.ID = string.(epv.ID)
epv_test = leftjoin(DataFrame(ID=test_ids), epv, on=:ID)
CSV.write(joinpath(results_folder, "EPV_test.csv"), epv_test)

println("EPV output: ", joinpath(results_folder, "EPV_test.csv"))
