# Part 6: NNMM as Traditional Genomic Prediction

!!! note "Version Compatibility"
    This documentation reflects **NNMM.jl v0.3+** using the `Layer`/`Equation`/`runNNMM` API.

## Overview

NNMM is a general framework that encompasses traditional genomic prediction models as special cases. By configuring the network architecture appropriately, you can use NNMM to perform standard single-trait BayesC (and other Bayesian alphabet methods).

This demonstrates that **NNMM generalizes traditional genomic prediction** rather than replacing it.

## The Key Insight

Traditional genomic prediction models like BayesC can be expressed as a special case of NNMM by:

1. **Using a completely missing middle layer** - Latent nodes will be sampled based on genotype information
2. **Using a linear activation function** - This ensures the relationship between the latent layer and phenotype is linear

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│  Genotypes  │ ──► │  Latent Nodes   │ ──► │ Phenotype   │
│   (SNPs)    │     │ (100% missing)  │     │    (y)      │
└─────────────┘     └─────────────────┘     └─────────────┘
                    Sampled via BayesC       Linear activation
```

This is mathematically equivalent to the traditional model:

$$y = X\beta + Z\alpha + e$$

where $\alpha$ are marker effects sampled using BayesC.

!!! important "Data Requirements"
    Genotype and phenotype files must have matching individual IDs. This example uses the 
    `simulated_omics_data` dataset which has 3534 aligned individuals.

## Example: Single-Trait BayesC via NNMM

```julia
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random

Random.seed!(42)

# Step 1: Load data (using simulated data with aligned individuals)
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
pheno_df = CSV.read(pheno_path, DataFrame)

# Step 2: Create middle layer with completely missing latent nodes
# Use at least 2 latent nodes for matrix operations in HMC
n_individuals = nrow(pheno_df)
latent_df = DataFrame(
    ID = pheno_df.ID,
    latent1 = fill(missing, n_individuals),
    latent2 = fill(missing, n_individuals)
)
latent_file = "latent_bayesc.csv"
CSV.write(latent_file, latent_df; missingstring="NA")

# Step 3: Create phenotype file
trait_df = pheno_df[:, [:ID, :trait1]]
trait_file = "trait_bayesc.csv"
CSV.write(trait_file, trait_df; missingstring="NA")

# Step 4: Define 3-layer network
layers = [
    Layer(layer_name = "geno", data_path = [geno_path]),
    Layer(layer_name = "latent", data_path = latent_file, missing_value = "NA"),
    Layer(layer_name = "phenotypes", data_path = trait_file, missing_value = "NA")
]

# Step 5: Define equations with LINEAR activation
equations = [
    # Genotypes → Latent (marker regression with BayesC)
    Equation(
        from_layer_name = "geno",
        to_layer_name = "latent",
        equation = "latent = intercept + geno",
        omics_name = ["latent1", "latent2"],
        method = "BayesC",
        estimatePi = true
    ),
    # Latent → Phenotype (LINEAR activation = standard regression)
    Equation(
        from_layer_name = "latent",
        to_layer_name = "phenotypes",
        equation = "phenotypes = intercept + latent",
        phenotype_name = ["trait1"],
        method = "BayesC",
        activation_function = "linear"  # Key: linear activation
    )
]

# Step 6: Run analysis
result = runNNMM(layers, equations;
    chain_length = 5000,
    burnin = 1000,
    output_folder = "nnmm_bayesc_results"
)

# Step 7: Get results
ebv = result["EBV_NonLinear"]
ebv.ID = string.(ebv.ID)
pheno_df.ID = string.(pheno_df.ID)

println("EBV mean: ", round(mean(Float64.(ebv.EBV)), digits=4))
println("EBV std:  ", round(std(Float64.(ebv.EBV)), digits=4))

# Check accuracy with true breeding values
merged = innerjoin(ebv, pheno_df[:, [:ID, :genetic_total]], on=:ID)
accuracy = cor(Float64.(merged.EBV), merged.genetic_total)
println("Accuracy: ", round(accuracy, digits=4))

# Cleanup
rm(latent_file, force=true)
rm(trait_file, force=true)
```

## Current Limitations

!!! warning "Multi-Trait Analysis"
    Multiple phenotypes in the output layer (Layer 3) are not yet fully supported in the current NNMM implementation. Multi-trait genomic prediction will be supported in a future release.

## Comparison: Direct BayesC vs NNMM as BayesC

| Aspect | Traditional BayesC | NNMM as BayesC |
|--------|-------------------|----------------|
| Model equation | `y = intercept + geno` | 3-layer network with missing middle |
| Marker effects | Directly estimated | Estimated via latent layer |
| Flexibility | Fixed model | Can add intermediate omics later |
| Results | EBV, marker effects | EBV, marker effects |
| Complexity | Simpler | Slightly more setup |

## Why Use NNMM for Traditional Analysis?

1. **Unified framework**: Same code structure for simple and complex models
2. **Easy extension**: Add observed omics data to the middle layer when available
3. **Gradual complexity**: Start with traditional model, add neural network components as needed
4. **Future-proof**: As more omics data becomes available, simply fill in the missing values

## Other Bayesian Methods

You can use any Bayesian alphabet method by changing the `method` parameter:

```julia
# BayesA - marker-specific variances, all markers included
Equation(..., method="BayesA")

# BayesB - marker-specific variances, subset of markers included
Equation(..., method="BayesB", estimatePi=true)

# BayesC - common variance, subset of markers included (default)
Equation(..., method="BayesC", estimatePi=true)

# BayesL - Bayesian LASSO
Equation(..., method="BayesL")

# RR-BLUP - Ridge Regression BLUP (all markers, common variance)
Equation(..., method="RR-BLUP")

# GBLUP - Genomic BLUP (only for Layer 1→2)
# Note: GBLUP is only supported for the genotype→omics equation
Equation(..., method="GBLUP")  # Layer 1→2 only
```

## Summary

NNMM provides a **superset** of traditional genomic prediction methods:

| Model Type | NNMM Configuration |
|------------|-------------------|
| Traditional BayesC | Missing middle layer + linear activation |
| NNMM with omics | Observed intermediate traits in middle layer |
| Full neural network | Nonlinear activation function |

This design allows researchers to start with familiar methods and gradually incorporate more complex models as data and understanding grow.
