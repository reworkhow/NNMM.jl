# NNMM.jl

[![CI](https://github.com/reworkhow/NNMM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/reworkhow/NNMM.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/reworkhow/NNMM.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/reworkhow/NNMM.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://reworkhow.github.io/NNMM.jl/dev)

**NNMM.jl** is an open-source Julia package for **Neural Network Mixed Models** that extend traditional linear mixed models to multilayer neural networks for genomic prediction and genome-wide association studies.

* **Documentation**: [https://reworkhow.github.io/NNMM.jl/dev](https://reworkhow.github.io/NNMM.jl/dev)
* **Authors**: [Hao Cheng](https://qtl.rocks), [Tianjing Zhao](https://animalscience.unl.edu/person/tianjing-zhao/)

## Key Features

- **Neural Network Architecture**: Extend linear mixed models to multilayer neural networks
- **Intermediate Omics Integration**: Incorporate gene expression, metabolomics, and other omics data
- **Flexible Missing Data**: Handle any pattern of missing data in intermediate layers
- **Bayesian Framework**: Full Bayesian inference using MCMC and Hamiltonian Monte Carlo
- **Multiple Bayesian Methods**: BayesA, BayesB, BayesC, BayesL, GBLUP
- **Multi-threaded**: Parallel computing support for large-scale analyses

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/reworkhow/NNMM.jl")
```

**Requirements**: Julia 1.9 or later

## Quick Start

```julia
using NNMM, DataFrames, CSV, NNMM.Datasets

# Get example data paths
geno_path = dataset("genotypes0.csv")
pedigree = get_pedigree(dataset("pedigree.csv"), separator=",", header=true)

# Define network layers
layers = [
    Layer(layer_name="geno", data_path=[geno_path]),
    Layer(layer_name="omics", data_path="omics.csv", missing_value="NA"),
    Layer(layer_name="phenotypes", data_path="phenotypes.csv", missing_value="NA")
]

# Define equations between layers
equations = [
    # Genotypes → Omics (intermediate traits)
    Equation(
        from_layer_name="geno",
        to_layer_name="omics",
        equation="omics = intercept + x1 + x2 + ID + geno",
        omics_name=["o1", "o2", "o3"],
        covariate=["x1", "x2"],
        random=[(name="ID", pedigree=pedigree)],
        method="BayesC"
    ),
    # Omics → Phenotypes
    Equation(
        from_layer_name="omics",
        to_layer_name="phenotypes",
        equation="phenotypes = intercept + ID + x4 + omics",
        phenotype_name=["y1"],
        covariate=["x4"],
        random=[(name="ID", pedigree=pedigree)],
        method="BayesC",
        activation_function="sigmoid"
    )
]

# Run NNMM
results = runNNMM(layers, equations; chain_length=5000)
```

## Network Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Genotypes  │ ──► │   Omics     │ ──► │ Phenotypes  │
│  (Layer 1)  │     │  (Layer 2)  │     │  (Layer 3)  │
└─────────────┘     └─────────────┘     └─────────────┘
     SNPs           Gene Expression        Traits
                    Metabolomics
                    (can be missing)
```

NNMM supports:
- **Fully-connected** and **partial-connected** neural networks
- **User-defined activation functions** between layers
- **Any pattern of missing data** in intermediate layers (sampled via HMC)

## Help

```julia
using NNMM
?NNMM      # Show package info
?runNNMM   # Help on specific function
?Layer     # Help on Layer type
?Equation  # Help on Equation type
```

## Citation

If you use NNMM.jl in your research, please cite:

> Zhao, T., Zeng, J., & Cheng, H. (2022). Extend mixed models to multilayer neural networks for genomic prediction including intermediate omics data. *GENETICS*, iyac034. https://doi.org/10.1093/genetics/iyac034

> Zhao, T., Fernando, R., & Cheng, H. (2021). Interpretable artificial neural networks incorporating Bayesian alphabet models for genome-wide prediction and association studies. *G3 Genes|Genomes|Genetics*, jkab228. https://doi.org/10.1093/g3journal/jkab228
