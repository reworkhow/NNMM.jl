# MCMC Convergence Check Results

## Purpose

This document verifies that the MCMC chain has converged and that the benchmark results at `chain_length=1000` are reliable.

## Configuration

- **Seed**: 42
- **Burnin**: 20% of chain length
- **Method**: BayesC (both layers)
- **Activation**: linear

## Dataset

- **Name**: simulated_omics_data
- **Individuals**: 3534
- **SNPs**: 1000 (927 after MAF filtering)
- **Omics**: 10
- **Target heritability**: 0.5 (20% direct, 80% indirect)

---

## Convergence Results

| Chain Length | Burnin | cor(EBV, total) | cor(EBV, direct) | cor(EBV, indirect) |
|--------------|--------|-----------------|------------------|---------------------|
| 1000 | 200 | **0.8549** | 0.0399 | 0.9358 |
| 1500 | 300 | **0.8552** | 0.0403 | 0.9360 |
| 2000 | 400 | **0.8549** | 0.0394 | 0.9361 |
| 3000 | 600 | **0.8557** | 0.0408 | 0.9363 |

---

## Analysis

### Accuracy Change

| Comparison | Difference |
|------------|------------|
| 1000 → 1500 | +0.0003 |
| 1500 → 2000 | -0.0003 |
| 2000 → 3000 | +0.0008 |
| **1000 → 3000** | **+0.0009** |

### Conclusion

✅ **Chain has converged**

- Total accuracy change from 1000 to 3000 iterations: **0.0009** (< 0.1%)
- All metrics are stable across different chain lengths
- Results are reproducible with the same seed

---

## Recommendations

1. **chain_length=1000** with **burnin=200** is sufficient for this dataset
2. No significant accuracy improvement from longer chains
3. The benchmark results at chain_length=1000 are reliable for PyNNMM comparison

---

## Running the Convergence Check

```bash
cd /Users/haocheng/Github/AFOCUS/NNMM.jl
julia --project=. benchmarks/check_convergence.jl
```

---

*Generated: 2025-12-30*

