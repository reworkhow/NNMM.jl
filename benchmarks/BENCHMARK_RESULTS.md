# NNMM Benchmark Results for PyNNMM Parity

This document contains benchmark results from NNMM.jl that should be used as target values for validating the PyNNMM implementation.

## Dataset: simulated_omics_data

| Property | Value |
|----------|-------|
| Individuals | 3,534 |
| SNPs | 1,000 (927 after MAF filtering) |
| Omics | 10 |
| Target heritability | 0.5 (20% direct, 80% indirect) |

### Ground Truth

The dataset includes true breeding values:
- `genetic_total` = Total genetic value (direct + indirect)
- `genetic_direct` = Direct SNP effects (20% of genetic variance)
- `genetic_indirect` = Omics-mediated effects (80% of genetic variance)

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Chain length | 100 |
| Burnin | 20 |
| Method | BayesC (both layers) |
| Activation | linear |

## Target Accuracy Values for PyNNMM

These are the accuracy metrics that PyNNMM should achieve (within ±0.05):

| Metric | Value |
|--------|-------|
| **cor(EBV, genetic_total)** | **0.7899** |
| cor(EBV, genetic_direct) | 0.0372 |
| **cor(EBV, genetic_indirect)** | **0.8645** |

### EBV Statistics

| Statistic | Value |
|-----------|-------|
| Mean | ~0.0 |
| Std | ~32.08 |

## Interpretation

1. **High accuracy for indirect effects (0.86)**: NNMM captures omics-mediated genetic effects well
2. **High accuracy for total effects (0.79)**: Good overall genomic prediction
3. **Low accuracy for direct effects (0.04)**: Expected since only 20% of genetic variance is direct

## Running the Benchmark

```bash
cd NNMM.jl
julia --project=. benchmarks/benchmark_accuracy.jl
```

## Notes

- PyNNMM results should match within ±0.05 for the same seed and configuration
- The simulated data is deterministic (from `simulate_from_genotypes.py` with seed=42)
- MCMC sampling introduces stochastic variation, so exact matches are not expected

