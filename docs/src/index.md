
![NNMM](assets/NNMM.png)

# NNMM.jl - Neural Network Mixed Models

NNMM.jl is a Julia package for **Mixed Effect Neural Networks** that extend traditional linear mixed models to multilayer neural networks for genomic prediction and genome-wide association studies.

## Key Features

* **Neural Network Architecture**: Extend linear mixed models to multilayer neural networks
* **Intermediate Omics Integration**: Incorporate known intermediate omics features (e.g., gene expression) in the middle layer
* **Flexible Missing Data Handling**: Allow any patterns of missing data in intermediate layers
* **Bayesian Framework**: Full Bayesian inference using MCMC and Hamiltonian Monte Carlo
* **Multi-threaded Parallelism**: Efficient parallel computing for large-scale analyses

## Network Architecture

NNMM models relationships between:
- **Input Layer**: Genotypes (SNP markers)
- **Middle Layer**: Intermediate traits (e.g., gene expression, metabolomics)
- **Output Layer**: Phenotypes

The package supports:
- Fully-connected neural networks
- Partial-connected neural networks
- User-defined nonlinear activation functions

## Installation

```julia
using Pkg
Pkg.add("NNMM")
```

Or for the latest development version:
```julia
Pkg.add(url="https://github.com/reworkhow/NNMM.jl")
```

## Quick Start

```julia
using NNMM, CSV, DataFrames

# Load data
phenotypes = CSV.read("phenotypes.csv", DataFrame)
omics = CSV.read("omics.csv", DataFrame)
genotypes = CSV.read("genotypes.csv", DataFrame)

# Define network layers
layer1 = Layer(data=genotypes)
layer2 = Layer(data=omics, n_latent=size(omics, 2)-1)
layer3 = Layer(data=phenotypes)

# Build model equation
eq = Equation(layer1, layer2, layer3, "y")

# Run MCMC
results = runNNMM(eq, chain_length=5000, output_folder="results")
```

## Supporting and Citing

If you use NNMM.jl for your research, please cite:

> Zhao, T., Zeng, J., & Cheng, H. (2022). Extend mixed models to multilayer neural networks for genomic prediction including intermediate omics data. *GENETICS*. https://doi.org/10.1093/genetics/iyac034

> Zhao, T., Fernando, R., & Cheng, H. (2021). Interpretable artificial neural networks incorporating Bayesian alphabet models for genome-wide prediction and association studies. *G3 Genes|Genomes|Genetics*. https://doi.org/10.1093/g3journal/jkab228

Please star the repository [here](https://github.com/reworkhow/NNMM.jl) to help demonstrate community involvement.

## Getting Help

- [Open an issue](https://github.com/reworkhow/NNMM.jl/issues) on GitHub
- Contact: <qtlcheng@ucdavis.edu>

## Tutorials

### NNMM Tutorials
```@contents
Pages = [
  "nnmm/Part1_introduction.md",
  "nnmm/Part2_NNMM.md",
  "nnmm/Part3_NNMMwithIntermediateOmics.md",
  "nnmm/Part4_PartialConnectedNeuralNetwork.md",
  "nnmm/Part5_UserDefinedNonlinearFunction.md",
]
Depth = 2
```

### Manual
```@contents
Pages = [
  "manual/getstarted.md",
  "manual/public.md",
]
Depth = 2
```
