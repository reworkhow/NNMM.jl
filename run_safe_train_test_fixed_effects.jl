#!/usr/bin/env julia

using NNMM
using CSV
using DataFrames
using Dates

# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Run with:
#   julia --project=. run_safe_train_test_fixed_effects.jl \
#       genotype.csv omics.csv phenotype.csv train_ids.csv test_ids.csv output_parent
#
# Required input columns:
#   genotype.csv : ID, marker1, marker2, ...
#   omics.csv    : ID, omic1, omic2, ...
#   phenotype.csv: ID, trait1, fixed-effect columns, ...
#   train_ids.csv/test_ids.csv: an ID column with a header

length(ARGS) == 6 || error(
    "Expected 6 arguments: genotype.csv omics.csv phenotype.csv " *
    "train_ids.csv test_ids.csv output_parent",
)

const GENOTYPE_FILE = abspath(ARGS[1])
const OMICS_FILE = abspath(ARGS[2])
const PHENOTYPE_FILE = abspath(ARGS[3])
const TRAIN_ID_FILE = abspath(ARGS[4])
const TEST_ID_FILE = abspath(ARGS[5])
const OUTPUT_PARENT = abspath(ARGS[6])

# Change these names to match your data.
const PHENOTYPE = :trait1
const OMICS_COLUMNS = [:omic1, :omic2, :omic3]

# Fixed effects for genotype -> omics (equation 1).
const CONTINUOUS_FIXED_12 = Symbol[]
const CATEGORICAL_FIXED_12 = Symbol[]

# Fixed effects for omics -> phenotype (equation 2).
const CONTINUOUS_FIXED_23 = [:age]
const CATEGORICAL_FIXED_23 = [:sex]

# MCMC settings.
const CHAIN_LENGTH = parse(Int, get(ENV, "NNMM_CHAIN_LENGTH", "5000"))
const BURNIN = parse(Int, get(ENV, "NNMM_BURNIN", "1000"))
const OUTPUT_SAMPLES_FREQUENCY = parse(Int, get(ENV, "NNMM_OUTPUT_FREQUENCY", "10"))
const OUTPUT_PREDICTION_FREQUENCY = parse(Int, get(ENV, "NNMM_PREDICTION_FREQUENCY", "10"))
const RANDOM_SEED = 2026
const METHOD_12 = "BayesC"
const METHOD_23 = "BayesC"
const ACTIVATION_FUNCTION = "linear"

# Genotype quality-control settings.
const QUALITY_CONTROL = true
const MAF = 0.01
const CENTER_GENOTYPES = true

# =============================================================================
# VALIDATION AND PREPARATION HELPERS
# =============================================================================

function require_file(path::AbstractString)
    isfile(path) || error("Input file does not exist: $path")
end

function require_columns(df::DataFrame, columns, label::AbstractString)
    available = Set(propertynames(df))
    missing_columns = [column for column in columns if column ∉ available]
    isempty(missing_columns) || error(
        "$label is missing required columns: $(join(string.(missing_columns), ", "))",
    )
end

function normalize_ids!(df::DataFrame, label::AbstractString)
    require_columns(df, [:ID], label)
    any(ismissing, df.ID) && error("$label contains missing IDs.")
    df.ID = strip.(string.(df.ID))
    any(isempty, df.ID) && error("$label contains empty IDs.")
    duplicated = unique(df.ID[nonunique(df, :ID)])
    isempty(duplicated) || error(
        "$label contains duplicate IDs: $(join(first(duplicated, min(10, length(duplicated))), ", "))",
    )
    return df
end

function read_id_file(path::AbstractString, label::AbstractString)
    df = CSV.read(path, DataFrame)
    normalize_ids!(df, label)
    ncol(df) == 1 || error("$label must contain exactly one column named ID.")
    return Vector{String}(df.ID)
end

function require_ids(df::DataFrame, ids::Vector{String}, label::AbstractString)
    absent = setdiff(ids, df.ID)
    isempty(absent) || error(
        "$label is missing requested IDs: $(join(first(absent, min(10, length(absent))), ", "))",
    )
end

function rows_in_id_order(df::DataFrame, ids::Vector{String})
    row_for_id = Dict(id => row for (row, id) in enumerate(df.ID))
    return copy(df[[row_for_id[id] for id in ids], :])
end

function validate_numeric_fixed_effects(
    df::DataFrame,
    columns::Vector{Symbol},
    label::AbstractString,
)
    require_columns(df, columns, label)
    for column in columns
        values = df[!, column]
        any(ismissing, values) && error("Fixed effect $column contains missing values.")
        all(value -> value isa Number, values) ||
            error("Continuous fixed effect $column must contain only numeric values.")
    end
end

function encode_genotype_missing_values!(df::DataFrame; sentinel::Float64=9.0)
    ncol(df) >= 2 || error("Genotype file must contain ID and at least one marker column.")
    for column in propertynames(df)[2:end]
        values = df[!, column]
        all(value -> ismissing(value) || value isa Number, values) ||
            error("Genotype marker $column contains a nonnumeric value.")
        df[!, column] = Float64[ismissing(value) ? sentinel : Float64(value) for value in values]
    end
    return df
end

"""
Reference-code categorical fixed effects as k-1 numeric dummy covariates.

Levels are learned from training individuals only. A test-only level is rejected because
its effect cannot be estimated from training data. The first sorted level is the reference.
"""
function add_categorical_dummies!(
    phenotype::DataFrame,
    train_ids::Vector{String},
    categorical_columns::Vector{Symbol},
)
    require_columns(phenotype, categorical_columns, "phenotype file")
    train_set = Set(train_ids)
    train_mask = in.(phenotype.ID, Ref(train_set))
    dummy_columns = Dict{Symbol, Vector{Symbol}}()
    coding_rows = NamedTuple[]

    for column in categorical_columns
        values = phenotype[!, column]
        any(ismissing, values) && error("Categorical fixed effect $column contains missing values.")
        string_values = string.(values)
        train_levels = sort(unique(string_values[train_mask]))
        length(train_levels) >= 2 || error(
            "Categorical fixed effect $column has fewer than two levels in training data.",
        )

        unknown_test_levels = setdiff(unique(string_values[.!train_mask]), train_levels)
        isempty(unknown_test_levels) || error(
            "Categorical fixed effect $column has test-only levels that cannot be estimated: " *
            join(unknown_test_levels, ", "),
        )

        reference_level = first(train_levels)
        generated = Symbol[]
        for (dummy_index, level) in enumerate(train_levels[2:end])
            level_index = dummy_index + 1
            dummy = Symbol("fixed_", column, "_level", level_index)
            dummy ∉ propertynames(phenotype) || error("Generated dummy column already exists: $dummy")
            phenotype[!, dummy] = Float64.(string_values .== level)
            push!(generated, dummy)
            push!(coding_rows, (
                SourceEffect=string(column),
                DummyEffect=string(dummy),
                ReferenceLevel=reference_level,
                RepresentedLevel=level,
            ))
        end
        dummy_columns[column] = generated
    end

    coding = isempty(coding_rows) ? DataFrame(
        SourceEffect=String[], DummyEffect=String[], ReferenceLevel=String[], RepresentedLevel=String[],
    ) : DataFrame(coding_rows)
    return dummy_columns, coding
end

function equation_rhs(fixed_terms::Vector{Symbol}, marker_term::AbstractString)
    terms = vcat(["intercept"], string.(fixed_terms), [marker_term])
    return join(terms, " + ")
end

# =============================================================================
# READ, VALIDATE, ALIGN, AND MASK DATA
# =============================================================================

foreach(require_file, [GENOTYPE_FILE, OMICS_FILE, PHENOTYPE_FILE, TRAIN_ID_FILE, TEST_ID_FILE])
mkpath(OUTPUT_PARENT)
CHAIN_LENGTH > 0 || error("CHAIN_LENGTH must be positive.")
0 <= BURNIN < CHAIN_LENGTH || error("BURNIN must satisfy 0 <= BURNIN < CHAIN_LENGTH.")

genotype_raw = normalize_ids!(CSV.read(GENOTYPE_FILE, DataFrame), "genotype file")
omics_raw = normalize_ids!(CSV.read(OMICS_FILE, DataFrame), "omics file")
phenotype_raw = normalize_ids!(CSV.read(PHENOTYPE_FILE, DataFrame), "phenotype file")
train_ids = read_id_file(TRAIN_ID_FILE, "training-ID file")
test_ids = read_id_file(TEST_ID_FILE, "testing-ID file")

isempty(train_ids) && error("Training-ID file is empty.")
isempty(test_ids) && error("Testing-ID file is empty.")
overlap = intersect(train_ids, test_ids)
isempty(overlap) || error("Training and testing ID files overlap: $(join(overlap, ", "))")

ordered_ids = vcat(train_ids, test_ids)
require_ids(genotype_raw, ordered_ids, "genotype file")
require_ids(omics_raw, ordered_ids, "omics file")
require_ids(phenotype_raw, ordered_ids, "phenotype file")

require_columns(omics_raw, OMICS_COLUMNS, "omics file")
all_fixed_continuous = unique(vcat(CONTINUOUS_FIXED_12, CONTINUOUS_FIXED_23))
all_fixed_categorical = unique(vcat(CATEGORICAL_FIXED_12, CATEGORICAL_FIXED_23))
require_columns(phenotype_raw, vcat([PHENOTYPE], all_fixed_continuous, all_fixed_categorical), "phenotype file")

genotype = rows_in_id_order(genotype_raw, ordered_ids)
omics = rows_in_id_order(omics_raw, ordered_ids)
phenotype_truth = rows_in_id_order(phenotype_raw, ordered_ids)
phenotype_model = copy(phenotype_truth)
encode_genotype_missing_values!(genotype)

genotype.ID == omics.ID || error("Internal error: genotype and omics IDs are not aligned.")
genotype.ID == phenotype_model.ID || error("Internal error: genotype and phenotype IDs are not aligned.")

# NNMM left-joins phenotype columns onto omics. Overlapping non-ID names are unsafe.
overlapping_columns = intersect(Set(names(omics)[2:end]), Set(names(phenotype_model)[2:end]))
isempty(overlapping_columns) || error(
    "Omics and phenotype files have overlapping non-ID columns: $(join(overlapping_columns, ", "))",
)

validate_numeric_fixed_effects(phenotype_model, all_fixed_continuous, "phenotype file")
dummy_by_effect, coding = add_categorical_dummies!(
    phenotype_model,
    train_ids,
    all_fixed_categorical,
)

function expanded_fixed_terms(continuous, categorical, dummy_by_effect)
    result = copy(continuous)
    for effect in categorical
        append!(result, dummy_by_effect[effect])
    end
    return unique(result)
end

fixed_terms_12 = expanded_fixed_terms(
    CONTINUOUS_FIXED_12,
    CATEGORICAL_FIXED_12,
    dummy_by_effect,
)
fixed_terms_23 = expanded_fixed_terms(
    CONTINUOUS_FIXED_23,
    CATEGORICAL_FIXED_23,
    dummy_by_effect,
)

train_rows = 1:length(train_ids)
test_rows = (length(train_ids) + 1):length(ordered_ids)
any(ismissing, phenotype_model[train_rows, PHENOTYPE]) && error(
    "Some training IDs have missing $PHENOTYPE values.",
)
allowmissing!(phenotype_model, PHENOTYPE)
phenotype_model[test_rows, PHENOTYPE] .= missing

# Use a unique run name so runNNMM receives an output path that does not exist yet.
run_stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
prepared_dir = joinpath(OUTPUT_PARENT, "prepared_" * run_stamp)
results_dir = joinpath(OUTPUT_PARENT, "results_" * run_stamp)
ispath(prepared_dir) && error("Prepared-data directory already exists: $prepared_dir")
ispath(results_dir) && error("Results directory already exists: $results_dir")
mkdir(prepared_dir)

genotype_path = joinpath(prepared_dir, "genotype_aligned.csv")
omics_path = joinpath(prepared_dir, "omics_aligned.csv")
phenotype_path = joinpath(prepared_dir, "phenotype_aligned_masked.csv")
truth_path = joinpath(prepared_dir, "test_truth.csv")

CSV.write(genotype_path, genotype)
CSV.write(omics_path, omics; missingstring="NA")
CSV.write(phenotype_path, phenotype_model; missingstring="NA")
CSV.write(truth_path, phenotype_truth[test_rows, [:ID, PHENOTYPE]]; missingstring="NA")
if nrow(coding) > 0
    CSV.write(joinpath(prepared_dir, "categorical_fixed_effect_coding.csv"), coding)
end

# =============================================================================
# DEFINE AND RUN NNMM
# =============================================================================

layers = [
    Layer(
        layer_name="geno",
        data_path=[genotype_path],
        missing_value=9.0,
        quality_control=QUALITY_CONTROL,
        MAF=MAF,
        center=CENTER_GENOTYPES,
    ),
    Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
    Layer(layer_name="phenotypes", data_path=phenotype_path, missing_value="NA"),
]

equations = [
    Equation(
        from_layer_name="geno",
        to_layer_name="omics",
        equation="omics = " * equation_rhs(fixed_terms_12, "geno"),
        traits=string.(OMICS_COLUMNS),
        covariate=string.(fixed_terms_12),
        method=METHOD_12,
        estimatePi=true,
    ),
    Equation(
        from_layer_name="omics",
        to_layer_name="phenotypes",
        equation="phenotypes = " * equation_rhs(fixed_terms_23, "omics"),
        traits=[string(PHENOTYPE)],
        covariate=string.(fixed_terms_23),
        method=METHOD_23,
        estimatePi=true,
        activation_function=ACTIVATION_FUNCTION,
    ),
]

println("Prepared exact ID order: $(length(train_ids)) training IDs followed by $(length(test_ids)) testing IDs.")
println("Equation 1: ", equations[1].equation)
println("Equation 2: ", equations[2].equation)
println("Results directory: ", results_dir)

result = runNNMM(
    layers,
    equations;
    chain_length=CHAIN_LENGTH,
    burnin=BURNIN,
    output_samples_frequency=OUTPUT_SAMPLES_FREQUENCY,
    output_prediction_frequency=OUTPUT_PREDICTION_FREQUENCY,
    seed=RANDOM_SEED,
    output_folder=results_dir,
)

# =============================================================================
# SAVE FIXED EFFECTS AND TEST-SET PREDICTIONS
# =============================================================================

for key in ("fixed effects 1->2", "fixed effects 2->3")
    haskey(result, key) || error(
        "Result does not contain '$key'. Use the local NNMM version with fixed-effect output support.",
    )
end

CSV.write(joinpath(results_dir, "fixed_effects_1_to_2.csv"), result["fixed effects 1->2"])
CSV.write(joinpath(results_dir, "fixed_effects_2_to_3.csv"), result["fixed effects 2->3"])

test_order = DataFrame(ID=test_ids)
test_output = leftjoin(test_order, phenotype_truth[test_rows, [:ID, PHENOTYPE]], on=:ID)

if haskey(result, "EBV_NonLinear")
    ebv = copy(result["EBV_NonLinear"])
    ebv.ID = string.(ebv.ID)
    rename!(ebv, :EBV => :EBV_NonLinear, :PEV => :PEV_EBV_NonLinear)
    test_output = leftjoin(test_output, ebv, on=:ID)
end

if haskey(result, "EPV_Output_NonLinear")
    epv = copy(result["EPV_Output_NonLinear"])
    epv.ID = string.(epv.ID)
    rename!(epv, :EPV => :EPV_Output_NonLinear, :PEV => :PEV_EPV_Output_NonLinear)
    test_output = leftjoin(test_output, epv, on=:ID)
end

CSV.write(joinpath(results_dir, "test_predictions.csv"), test_output; missingstring="NA")

println("Completed successfully.")
println("Prepared inputs: ", prepared_dir)
println("Model results:   ", results_dir)
println("Test predictions: ", joinpath(results_dir, "test_predictions.csv"))
