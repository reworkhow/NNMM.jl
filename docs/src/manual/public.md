# Public API Reference

Documentation for NNMM.jl's public interface. Below are functions available to general users.

!!! note "Version Compatibility"
    This documentation reflects **NNMM.jl v0.3+** using the `Layer`/`Equation`/`runNNMM` API.

## Index

```@index
Pages = ["public.md"]
Modules = [NNMM]
```

## Core Types

### Layer

Defines a layer in the neural network architecture.

```@docs
Layer
```

**Usage:**
```julia
# Genotype layer (input) - note the [] around the path
layer1 = Layer(layer_name="geno", data_path=["genotypes.csv"])

# Omics layer (hidden) with missing value handling
layer2 = Layer(layer_name="omics", data_path="omics.csv", missing_value="NA")

# Phenotype layer (output)
layer3 = Layer(layer_name="pheno", data_path="phenotypes.csv", missing_value="NA")
```

### Equation

Defines the statistical model connecting two layers.

```@docs
Equation
```

**Usage:**
```julia
# Genotypes → Omics (BayesC with 10 omics features)
eq1 = Equation(
    from_layer_name = "geno",
    to_layer_name = "omics", 
    equation = "omics = intercept + geno",
    omics_name = ["omic1", "omic2", "omic3"],
    method = "BayesC",
    estimatePi = true
)

# Omics → Phenotypes (with sigmoid activation)
eq2 = Equation(
    from_layer_name = "omics",
    to_layer_name = "pheno",
    equation = "pheno = intercept + omics",
    phenotype_name = ["y1"],
    method = "BayesC",
    activation_function = "sigmoid"
)
```

### Supporting Types

```@docs
Omics
Phenotypes
```

## Main Functions

### runNNMM

The primary function for running NNMM analysis.

```@docs
runNNMM
```

**Usage:**
```julia
results = runNNMM(layers, equations;
    chain_length = 10000,
    burnin = 2000,
    output_folder = "my_results",
    seed = 42
)
```

### describe

Print model summary information.

```@docs
describe
```

## Data Reading Functions

### read_phenotypes

Read phenotype data from a file.

```@docs
read_phenotypes
```

### nnmm_get_genotypes

Read genotype data from a file or matrix.

```@docs
nnmm_get_genotypes
```

### get_genotypes

Alias for `nnmm_get_genotypes`.

```@docs
get_genotypes
```

### nnmm_get_omics

Read omics data from a file.

```@docs
nnmm_get_omics
```

## Post-Analysis Functions

### GWAS

Genome-wide association study on MCMC results.

```@docs
GWAS
```

**Usage:**
```julia
# Run GWAS on marker effect samples
gwas_result = GWAS("results/MCMC_samples_marker_effects_geno_omic1.txt")

# Sort by model frequency
sorted = sort(gwas_result, :modelfrequency, rev=true)
println(first(sorted, 10))
```

### getEBV

Extract estimated breeding values from results.

```@docs
getEBV
```

## Built-in Datasets

### dataset

Access built-in example datasets.

```@docs
NNMM.Datasets.dataset
```

**Usage:**
```julia
using NNMM.Datasets

# Access default example data
pheno_path = Datasets.dataset("phenotypes.csv")
geno_path = Datasets.dataset("genotypes.csv")

# Access simulated omics dataset
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
```

**Available Datasets:**

| Dataset | Files | Description |
|---------|-------|-------------|
| (default) | `phenotypes.csv`, `genotypes.csv`, `genotypes0.csv`, `pedigree.csv`, `GRM.csv`, `map.csv` | Small example data |
| (default) | `genotypes_group1.csv`, `genotypes_group2.csv`, `genotypes_group3.csv` | Genotype groups for partial networks |
| `example` | `phenotypes.txt`, `genotypes.txt`, `pedigree.txt`, etc. | Tab-separated example data |
| `simulated_omics_data` | `genotypes_1000snps.txt`, `phenotypes_sim.txt`, `pedigree.txt` | Simulated dataset with 1000 SNPs and 10 omics |

## Pedigree Functions

### get_pedigree

Read and process pedigree information.

```@docs
get_pedigree
```

**Usage:**
```julia
# Read pedigree file
pedigree = get_pedigree("pedigree.csv", separator=',', header=true)

# Use in random effect specification
random_spec = [(name="ID", pedigree=pedigree)]
```

## Parameter Reference Tables

### Bayesian Methods

| Method | Description |
|--------|-------------|
| `"BayesA"` | All markers have non-zero effects with marker-specific variances |
| `"BayesB"` | Subset of markers have non-zero effects with marker-specific variances |
| `"BayesC"` | Subset of markers have non-zero effects with common variance |
| `"BayesL"` | Bayesian LASSO |
| `"RR-BLUP"` | Ridge regression BLUP (all markers, common variance) |
| `"GBLUP"` | Genomic BLUP using relationship matrix (Layer 1→2 only) |

### Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| `"linear"` | f(x) = x | (-∞, ∞) | Traditional regression |
| `"sigmoid"` | f(x) = 1/(1+e^(-x)) | (0, 1) | Bounded outputs |
| `"tanh"` | f(x) = tanh(x) | (-1, 1) | Centered bounded outputs |
| `"relu"` | f(x) = max(0, x) | [0, ∞) | Sparse activation |
| `"leakyrelu"` | f(x) = max(0.01x, x) | (-∞, ∞) | Sparse with gradient flow |

### runNNMM Keyword Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `chain_length` | Integer | 100 | Total MCMC iterations |
| `burnin` | Integer | 0 | Burn-in iterations to discard |
| `output_samples_frequency` | Integer | auto | Save every Nth sample |
| `outputEBV` | Bool | true | Output estimated breeding values |
| `output_heritability` | Bool | true | Calculate heritability |
| `output_folder` | String | "nnmm_results" | Output directory |
| `seed` | Int/Bool | false | Random seed (false = random) |
| `printout_frequency` | Integer | chain_length+1 | Print progress frequency |
| `double_precision` | Bool | false | Use Float64 instead of Float32 |
| `big_memory` | Bool | false | Enable memory-intensive optimizations |
