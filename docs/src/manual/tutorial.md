# Step-by-Step Tutorial

This tutorial walks through a complete NNMM analysis using simulated data.

## Overview

We'll perform genomic prediction using a two-layer Neural Network Mixed Model:

```
Genotypes (SNPs) → Omics (gene expression) → Phenotype (trait)
```

This architecture allows us to:
- Model the biological pathway from DNA to gene expression to phenotype
- Handle missing omics data via HMC sampling
- Obtain estimated breeding values (EBV) for selection

## Step 1: Load Packages and Data

```julia
using NNMM
using NNMM.Datasets
using DataFrames
using CSV

# Load built-in simulated dataset
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
ped_path = Datasets.dataset("pedigree.txt", dataset_name="simulated_omics_data")

# Check the data
println("Genotype file: ", geno_path)
println("Phenotype file: ", pheno_path)
```

## Step 2: Prepare Data Files

The phenotype file contains both omics and trait data. We need to split them:

```julia
# Read the combined phenotype file
pheno_df = CSV.read(pheno_path, DataFrame)
println("Columns: ", names(pheno_df))

# Create omics file (use 5 omics features)
omics_df = pheno_df[:, [:ID, :omic1, :omic2, :omic3, :omic4, :omic5]]
omics_file = "omics_data.csv"
CSV.write(omics_file, omics_df; missingstring="NA")

# Create phenotype file (single trait)
trait_df = pheno_df[:, [:ID, :trait1]]
trait_file = "phenotypes.csv"
CSV.write(trait_file, trait_df; missingstring="NA")

println("Created: $omics_file and $trait_file")
```

## Step 3: Define the Network Architecture

NNMM uses `Layer` and `Equation` structs to define the model:

```julia
# Define the three layers
layers = [
    # Layer 1: Genotypes (input layer)
    Layer(
        layer_name = "genotypes",
        data_path = [geno_path]
    ),
    
    # Layer 2: Omics (hidden/intermediate layer)
    Layer(
        layer_name = "omics",
        data_path = omics_file,
        missing_value = "NA"    # Handle missing omics values
    ),
    
    # Layer 3: Phenotypes (output layer)
    Layer(
        layer_name = "phenotypes",
        data_path = trait_file,
        missing_value = "NA"
    )
]
```

## Step 4: Define the Equations

Equations connect the layers and specify the statistical method:

```julia
equations = [
    # Layer 1 → Layer 2: Genotypes predict omics
    Equation(
        from_layer_name = "genotypes",
        to_layer_name = "omics",
        equation = "omics = intercept + genotypes",
        omics_name = ["omic1", "omic2", "omic3", "omic4", "omic5"],
        method = "BayesC",        # Bayesian method
        estimatePi = true         # Estimate marker inclusion probability
    ),
    
    # Layer 2 → Layer 3: Omics predict phenotype
    Equation(
        from_layer_name = "omics",
        to_layer_name = "phenotypes",
        equation = "phenotypes = intercept + omics",
        phenotype_name = ["trait1"],
        method = "BayesC",
        estimatePi = true
    )
]
```

## Step 5: Run the Analysis

```julia
# Run NNMM with MCMC sampling
results = runNNMM(
    layers, 
    equations;
    chain_length = 5000,         # Number of MCMC iterations
    burnin = 1000,               # Burn-in iterations
    printout_frequency = 1000,   # Print progress every N iterations
    output_folder = "nnmm_output"  # Where to save results
)
```

!!! tip "For quick testing"
    Use `chain_length=100` for testing. For production, use at least `chain_length=50000`.

## Step 6: Examine Results

### Estimated Breeding Values (EBV)

```julia
# Get EBV for all individuals
ebv = results["EBV_NonLinear"]
println("EBV for first 10 individuals:")
display(first(ebv, 10))

# Summary statistics
println("\nEBV summary:")
println("Mean: ", round(mean(ebv.EBV), digits=3))
println("Std:  ", round(std(ebv.EBV), digits=3))
println("Min:  ", round(minimum(ebv.EBV), digits=3))
println("Max:  ", round(maximum(ebv.EBV), digits=3))
```

### MCMC Output Files

The output folder contains MCMC samples for:

| File | Description |
|------|-------------|
| `MCMC_samples_marker_effects_*.txt` | Marker effect samples |
| `MCMC_samples_marker_effects_variances_*.txt` | Marker variance samples |
| `MCMC_samples_residual_variance.txt` | Residual variance samples |
| `MCMC_samples_pi_*.txt` | Marker inclusion probability samples |
| `MCMC_samples_EBV_*.txt` | EBV samples per trait/omics |
| `MCMC_samples_EPV_NonLinear.txt` | Final phenotypic value samples |

## Step 7: Post-Analysis with GWAS

Identify important markers using model frequency:

```julia
# GWAS based on model frequency (probability marker is included)
marker_file = "nnmm_output/MCMC_samples_marker_effects_genotypes_omic1.txt"
gwas_result = GWAS(marker_file)

# Top markers for omic1
sorted_gwas = sort(gwas_result, :modelfrequency, rev=true)
println("Top 10 markers for omic1:")
display(first(sorted_gwas, 10))
```

## Complete Script

Here's the complete analysis in one script:

```julia
using NNMM
using NNMM.Datasets
using DataFrames
using CSV

# === Data Preparation ===
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")

pheno_df = CSV.read(pheno_path, DataFrame)

# Create omics and trait files
omics_df = pheno_df[:, [:ID, :omic1, :omic2, :omic3]]
CSV.write("omics.csv", omics_df; missingstring="NA")

trait_df = pheno_df[:, [:ID, :trait1]]
CSV.write("traits.csv", trait_df; missingstring="NA")

# === Define Model ===
layers = [
    Layer(layer_name="geno", data_path=[geno_path]),
    Layer(layer_name="omics", data_path="omics.csv", missing_value="NA"),
    Layer(layer_name="pheno", data_path="traits.csv", missing_value="NA")
]

equations = [
    Equation(
        from_layer_name="geno", to_layer_name="omics",
        equation="omics = intercept + geno",
        omics_name=["omic1", "omic2", "omic3"],
        method="BayesC"
    ),
    Equation(
        from_layer_name="omics", to_layer_name="pheno",
        equation="pheno = intercept + omics",
        phenotype_name=["trait1"],
        method="BayesC"
    )
]

# === Run MCMC ===
results = runNNMM(layers, equations; 
                  chain_length=5000, 
                  burnin=1000,
                  output_folder="results")

# === Results ===
ebv = results["EBV_NonLinear"]
println("Analysis complete! EBV saved in results folder.")
println("Mean EBV: ", mean(ebv.EBV))

# Cleanup temp files
rm("omics.csv", force=true)
rm("traits.csv", force=true)
```

## Next Steps

- **Different methods**: Try `method="BayesA"`, `"BayesB"`, `"BayesL"`, or `"RR-BLUP"`
- **Add pedigree**: Include `pedigree_file` in `Layer` for relationship matrices
- **Multi-trait**: Add more phenotypes in `phenotype_name`
- **Activation functions**: Use `activation_function="tanh"` for non-linear relationships
- **Missing omics**: NNMM automatically handles missing omics via HMC sampling

See the [NNMM Examples](../nnmm/Part2_NNMM.md) for more advanced use cases.

