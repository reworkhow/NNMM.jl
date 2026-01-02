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

## Pedigree Functions

```@docs
get_pedigree
```
