# Part 6. NNMM as Traditional Genomic Prediction

## Overview

NNMM is a general framework that encompasses traditional genomic prediction models as special cases. By configuring the network architecture appropriately, you can use NNMM to perform standard single-trait or multi-trait BayesC (and other Bayesian alphabet methods).

This demonstrates that **NNMM generalizes traditional genomic prediction** rather than replacing it.

## The Key Insight

Traditional genomic prediction models like BayesC can be expressed as a special case of NNMM by:

1. **Using a single completely missing node in the middle layer** - This latent node will be sampled based on genotype information
2. **Using a linear activation function** - This ensures the relationship between the latent node and phenotype is linear

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│  Genotypes  │ ──► │  Latent Node    │ ──► │ Phenotype   │
│   (SNPs)    │     │ (100% missing)  │     │    (y)      │
└─────────────┘     └─────────────────┘     └─────────────┘
                    Sampled via BayesC       Linear activation
```

This is mathematically equivalent to the traditional model:

$$y = X\beta + Z\alpha + e$$

where $\alpha$ are marker effects sampled using BayesC.

## Example: Single-Trait BayesC

```julia
using NNMM, DataFrames, CSV

# Step 1: Create a middle layer with ONE completely missing node
omics_df = DataFrame(
    ID = individual_ids,
    latent = fill(missing, n_individuals)  # All NA
)
CSV.write("latent.csv", omics_df; missingstring="NA")

# Step 2: Define 3-layer network
layers = [
    Layer(layer_name="geno", data_path=["genotypes.csv"]),
    Layer(layer_name="latent", data_path="latent.csv", missing_value="NA"),
    Layer(layer_name="phenotypes", data_path="phenotypes.csv", missing_value="NA")
]

# Step 3: Define equations with LINEAR activation
equations = [
    # Genotypes → Latent (marker regression)
    Equation(
        from_layer_name="geno",
        to_layer_name="latent",
        equation="latent = intercept + geno",
        omics_name=["latent"],
        method="BayesC",
        estimatePi=true
    ),
    # Latent → Phenotype (LINEAR activation = standard regression)
    Equation(
        from_layer_name="latent",
        to_layer_name="phenotypes",
        equation="phenotypes = intercept + latent",
        phenotype_name=["y1"],
        method="BayesC",
        activation_function="linear"  # Key: linear activation
    )
]

# Step 4: Run analysis
result = runNNMM(layers, equations; chain_length=50000, burnin=10000)
```

## Example: Multi-Trait BayesC

For multi-trait analysis, simply use multiple missing nodes:

```julia
# Middle layer with TWO completely missing nodes
omics_df = DataFrame(
    ID = individual_ids,
    latent1 = fill(missing, n_individuals),
    latent2 = fill(missing, n_individuals)
)
CSV.write("latent_multi.csv", omics_df; missingstring="NA")

# Phenotype data with two traits
pheno_df = DataFrame(
    ID = individual_ids,
    y1 = trait1_values,
    y2 = trait2_values
)

layers = [
    Layer(layer_name="geno", data_path=["genotypes.csv"]),
    Layer(layer_name="latent", data_path="latent_multi.csv", missing_value="NA"),
    Layer(layer_name="phenotypes", data_path="phenotypes_multi.csv", missing_value="NA")
]

equations = [
    Equation(
        from_layer_name="geno",
        to_layer_name="latent",
        equation="latent = intercept + geno",
        omics_name=["latent1", "latent2"],
        method="BayesC"
    ),
    Equation(
        from_layer_name="latent",
        to_layer_name="phenotypes",
        equation="phenotypes = intercept + latent",
        phenotype_name=["y1", "y2"],
        method="BayesC",
        activation_function="linear"
    )
]

result = runNNMM(layers, equations; chain_length=50000)
```

## Comparison with Direct BayesC

| Aspect | Traditional BayesC | NNMM as BayesC |
|--------|-------------------|----------------|
| Model equation | `y = intercept + geno` | 3-layer network with missing middle |
| Marker effects | Directly estimated | Estimated via latent layer |
| Flexibility | Fixed model | Can add intermediate omics later |
| Results | EBV, marker effects | EBV, marker effects |

## Why Use NNMM for Traditional Analysis?

1. **Unified framework**: Same code structure for simple and complex models
2. **Easy extension**: Add observed omics data to the middle layer when available
3. **Gradual complexity**: Start with traditional model, add neural network components as needed
4. **Future-proof**: As more omics data becomes available, simply fill in the missing values

## Other Bayesian Methods

You can use any Bayesian alphabet method by changing the `method` parameter:

```julia
# BayesA
Equation(..., method="BayesA")

# BayesB  
Equation(..., method="BayesB", estimatePi=true)

# BayesL (Bayesian LASSO)
Equation(..., method="BayesL")

# RR-BLUP (Ridge Regression)
Equation(..., method="RR-BLUP")
```

## Summary

NNMM provides a **superset** of traditional genomic prediction methods:

- **Traditional BayesC** = NNMM with completely missing middle layer + linear activation
- **NNMM with omics** = Same framework with observed intermediate traits
- **Full neural network** = Same framework with nonlinear activation functions

This design allows researchers to start with familiar methods and gradually incorporate more complex models as data and understanding grow.

