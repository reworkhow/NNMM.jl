# NNMM.jl Benchmark Results

## Dataset: simulated_omics_data

- **Individuals**: 3534
- **SNPs**: 1000 (927 after MAF filtering)
- **Omics**: 10
- **Target heritability**: 0.5 (20% direct, 80% indirect)

## Configuration

- **Seed**: 42
- **Chain length**: 1000
- **Burnin**: 200
- **Method**: BayesC (both layers)
- **Activation**: linear

---

## Full Omics Benchmark (`benchmark_accuracy.jl`)

### Accuracy Metrics

| Metric | Value |
|--------|-------|
| cor(EBV, genetic_total) | **0.8549** |
| cor(EBV, genetic_direct) | 0.0399 |
| cor(EBV, genetic_indirect) | 0.9358 |

### EBV Statistics

| Statistic | Value |
|-----------|-------|
| Mean | -0.0 |
| Std | 358.712 |

---

## Missing Omics Benchmark (`benchmark_missing_omics.jl`)

### Accuracy by Missing Percentage

| Missing % | Missing Cells | cor(EBV, total) | cor(EBV, direct) | cor(EBV, indirect) |
|-----------|---------------|-----------------|------------------|---------------------|
| 0% | 0 | **0.8549** | 0.0399 | 0.9358 |
| 30% | 10,600 | **0.7873** | 0.0685 | 0.8460 |
| 50% | 17,670 | **0.7365** | 0.1497 | 0.7486 |

### Accuracy Degradation from Baseline

| Missing % | Reduction |
|-----------|-----------|
| 30% | 7.9% |
| 50% | 13.8% |

---

## PyNNMM Target Values

PyNNMM should achieve similar accuracy metrics:
- Full omics: cor(EBV, genetic_total) ≈ 0.85
- 30% missing: cor(EBV, genetic_total) ≈ 0.79
- 50% missing: cor(EBV, genetic_total) ≈ 0.74

**Tolerance**: Results within ±0.05 are considered acceptable.

---

## Running Benchmarks

```bash
# Full omics benchmark
julia --project=. benchmarks/benchmark_accuracy.jl

# Missing omics benchmark
julia --project=. benchmarks/benchmark_missing_omics.jl
```

---

## Performance Benchmark

### Speed (1000 iterations, 3534 individuals, 927 SNPs, 10 omics)

| Metric | Value |
|--------|-------|
| Total time | ~38s |
| Iterations/sec | **26 iter/s** |

### Cross-Package Comparison

| Package | Accuracy (full) | Accuracy (30% missing) | Speed (iter/s) |
|---------|-----------------|------------------------|----------------|
| **NNMM.jl** | **0.8552** | **0.7810** | **26** |
| PyNNMM | 0.8127 | 0.7711 | 9.6 |
| Parity | 95% | 99% | 37% |

---

## Convergence Check (5 seeds × 1000 iterations)

| Scenario | Mean Accuracy | Std Dev | Status |
|----------|---------------|---------|--------|
| Full omics | **0.8552** | 0.0082 | ✅ CONVERGED |
| 30% missing | **0.7810** | 0.0138 | ✅ CONVERGED |

---

*Generated: 2025-12-31*
