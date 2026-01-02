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

## Post-Analysis

Functions for analyzing MCMC results.

```@docs
GWAS
getEBV
```

## Data Reading

Functions for loading genotype, omics, and phenotype data.

```@docs
read_genotypes
read_omics
read_phenotypes
```

## Utility Functions

Helper functions for advanced users.

```@docs
get_pedigree
get_genotypes
set_covariate
set_random
```
