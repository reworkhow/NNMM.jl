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

## Multi-Threading

NNMM.jl supports multi-threaded parallelism for the "mega-trait" approach when `constraint=true`.

### Enabling Multi-Threading

```bash
# Run with 10 threads (one per omics trait)
julia --project=. -t 10 benchmarks/benchmark_accuracy.jl
```

### Conditions for Parallel Speedup

Multi-threading provides speedup only when:
1. `constraint=true` for both residual and marker effect covariances
2. Each trait can be sampled independently (diagonal covariance)

The default benchmark uses `constraint=false` (full covariance), so multi-threading has no effect on speed.

---

## Gap Analysis (NNMM.jl vs PyNNMM)

### Accuracy Gap: 5% (0.85 vs 0.81)

| Factor | Impact | Description |
|--------|--------|-------------|
| RNG differences | Medium | Julia MT19937 vs C++ default_random_engine |
| Numerical precision | Low | Float32 (Julia) vs Float64 (C++) |
| Pre-computed matrices | Medium | Julia updates GibbsMats after X2 changes |
| Posterior mean | Fixed | Both now compute correctly |

### Speed Gap: 2.7x (26 vs 9.6 iter/s)

| Factor | NNMM.jl | PyNNMM | Impact |
|--------|---------|--------|--------|
| Compiler | Julia JIT (LLVM) | Clang -O3 | High |
| BLAS | OpenBLAS (native) | Disabled (Rosetta) | High |
| Memory layout | Column-major | Row-major | Medium |
| Pre-computed matrices | GibbsMats cached | Computed on-fly | Medium |
| Architecture | Native arm64 | x86_64 (Rosetta 2) | High |

### Recommendations for PyNNMM

1. **Use native arm64 Python** - Eliminates Rosetta 2 overhead
2. **Enable OpenBLAS** - After switching to arm64
3. **Add OpenMP parallelism** - For mega-trait mode
4. **Cache column extractions** - Pre-compute like GibbsMats

---

*Generated: 2025-12-31*
