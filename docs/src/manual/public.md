# Public Functions

Documentation for NNMM.jl's public interface. Below are functions available to general users.

## Index

```@index
Pages = ["public.md"]
Modules = [NNMM]
```

## Core Types

Types for defining the NNMM network architecture.

```@docs
Layer
Equation
Omics
Phenotypes
```

## Main Functions

Primary functions for running NNMM analysis.

```@docs
runNNMM
describe
```

## Data Reading Functions

Functions for reading input data files.

```@docs
nnmm_get_genotypes
get_genotypes
nnmm_get_omics
read_phenotypes
```

## Post-Analysis

Functions for analyzing MCMC results.

```@docs
GWAS
getEBV
```

## Built-in Datasets

```@docs
NNMM.Datasets.dataset
```

## Pedigree Functions

```@docs
get_pedigree
```
