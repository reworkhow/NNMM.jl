#!/usr/bin/env python3
"""Simulate phenotypes with fixed and random effects using a SNP subset."""

import argparse
import csv
import random
from pathlib import Path
from typing import List, Optional, Tuple


def parse_genotype(val: str) -> Optional[int]:
    try:
        fval = float(val)
    except ValueError:
        return None
    if fval in (0.0, 1.0, 2.0) and abs(fval - round(fval)) < 1e-8:
        return int(round(fval))
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Subset evenly spaced SNPs with integer-only genotypes and simulate "
            "phenotypes with fixed (contemporary group), random (litter), and omics effects."
        )
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--genotypes",
        type=Path,
        default=script_dir.parent / "real_data" / "genotypes.txt",
        help="Path to real_data/genotypes.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-snps", type=int, default=1000, help="SNPs to sample (evenly spaced)"
    )
    parser.add_argument(
        "--num-groups", type=int, default=50, help="Number of contemporary groups"
    )
    parser.add_argument(
        "--num-litters", type=int, default=300, help="Number of litters"
    )
    parser.add_argument(
        "--residual-sd", type=float, default=1.0, help="Residual SD"
    )
    parser.add_argument(
        "--group-sd", type=float, default=0.1, help="Group effect SD"
    )
    parser.add_argument(
        "--litter-sd", type=float, default=0.1, help="Litter effect SD"
    )
    parser.add_argument(
        "--num-omics", type=int, default=10, help="Number of omics traits to simulate"
    )
    parser.add_argument(
        "--omics-heritability",
        type=float,
        default=0.3,
        help="Target heritability for each omics trait",
    )
    parser.add_argument(
        "--omics-effect-sd",
        type=float,
        default=1.0,
        help="Effect SD for omics traits on the final phenotype",
    )
    parser.add_argument(
        "--heritability",
        type=float,
        default=0.5,
        help=(
            "Target additive heritability (0-1). If set, genetic values are "
            "simulated from SNPs and residual SD is adjusted so Vg/(Vg+Ve)=h2 "
            "(excluding group/litter effects)."
        ),
    )
    parser.add_argument(
        "--snp-effect-sd",
        type=float,
        default=1.0,
        help="SNP effect SD before scaling to target heritability",
    )
    return parser.parse_args()


def find_integer_only_snps(genotypes_path: Path) -> Tuple[List[str], List[int]]:
    with genotypes_path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        snp_names = header[1:]
        valid = [True] * len(snp_names)
        for row in reader:
            if not row:
                continue
            for idx, val in enumerate(row[1:]):
                if valid[idx] and parse_genotype(val) is None:
                    valid[idx] = False
        valid_indices = [i for i, ok in enumerate(valid) if ok]
    return snp_names, valid_indices


def variance(values: List[float]) -> float:
    if not values:
        return 0.0
    mean_val = sum(values) / len(values)
    return sum((x - mean_val) ** 2 for x in values) / len(values)


def covariance(values_x: List[float], values_y: List[float]) -> float:
    if not values_x or not values_y or len(values_x) != len(values_y):
        return 0.0
    mean_x = sum(values_x) / len(values_x)
    mean_y = sum(values_y) / len(values_y)
    total = 0.0
    for x, y in zip(values_x, values_y):
        total += (x - mean_x) * (y - mean_y)
    return total / len(values_x)


def select_evenly_spaced_indices(valid_indices: List[int], num_snps: int) -> List[int]:
    if num_snps <= 0:
        return []
    if num_snps == 1:
        return [valid_indices[len(valid_indices) // 2]]
    last = len(valid_indices) - 1
    positions = [int(round(i * last / (num_snps - 1))) for i in range(num_snps)]
    selected = [valid_indices[pos] for pos in positions]
    return sorted(set(selected))


def compute_genetic_values(
    genotypes: List[List[int]],
    snp_effect_sd: float,
    target_var: float,
) -> Tuple[List[float], List[float]]:
    if not genotypes or not genotypes[0] or target_var <= 0.0:
        return [0.0 for _ in genotypes], []
    num_animals = len(genotypes)
    num_snps = len(genotypes[0])
    sums = [0.0] * num_snps
    for row in genotypes:
        for i, val in enumerate(row):
            sums[i] += val
    means = [s / num_animals for s in sums]
    snp_effects = [random.gauss(0.0, snp_effect_sd) for _ in range(num_snps)]
    raw_values = []
    for row in genotypes:
        total = 0.0
        for i, val in enumerate(row):
            total += (val - means[i]) * snp_effects[i]
        raw_values.append(total)
    raw_var = variance(raw_values)
    if raw_var == 0.0:
        raise ValueError("Genetic variance is zero; cannot scale to target heritability.")
    scale = (target_var / raw_var) ** 0.5
    scaled_values = [v * scale for v in raw_values]
    scaled_effects = [e * scale for e in snp_effects]
    return scaled_values, scaled_effects


def compute_genetic_values_unscaled(
    genotypes: List[List[int]],
    snp_effect_sd: float,
) -> Tuple[List[float], List[float]]:
    if not genotypes or not genotypes[0]:
        return [0.0 for _ in genotypes], []
    num_animals = len(genotypes)
    num_snps = len(genotypes[0])
    sums = [0.0] * num_snps
    for row in genotypes:
        for i, val in enumerate(row):
            sums[i] += val
    means = [s / num_animals for s in sums]
    snp_effects = [random.gauss(0.0, snp_effect_sd) for _ in range(num_snps)]
    raw_values = []
    for row in genotypes:
        total = 0.0
        for i, val in enumerate(row):
            total += (val - means[i]) * snp_effects[i]
        raw_values.append(total)
    return raw_values, snp_effects


def simulate_omics_from_blocks(
    genotypes: List[List[int]],
    num_omics: int,
    omics_h2: float,
    snp_effect_sd: float,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    if num_omics <= 0 or not genotypes:
        return [], [], []
    num_snps = len(genotypes[0])
    if num_snps % num_omics != 0:
        raise ValueError(
            f"num_snps ({num_snps}) must be divisible by num_omics ({num_omics})."
        )
    block_size = num_snps // num_omics
    residual_sd = (1.0 - omics_h2) ** 0.5
    omics_values = [[0.0 for _ in range(num_omics)] for _ in range(len(genotypes))]
    omics_genetic_values = [
        [0.0 for _ in range(num_omics)] for _ in range(len(genotypes))
    ]
    omics_snp_effects: List[List[float]] = []
    for omic_idx in range(num_omics):
        start = omic_idx * block_size
        end = start + block_size
        block_genotypes = [row[start:end] for row in genotypes]
        genetic_values, snp_effects = compute_genetic_values(
            block_genotypes, snp_effect_sd, omics_h2
        )
        omics_snp_effects.append(snp_effects)
        for i, gval in enumerate(genetic_values):
            omics_genetic_values[i][omic_idx] = gval
            omics_values[i][omic_idx] = gval + random.gauss(0.0, residual_sd)
    return omics_values, omics_genetic_values, omics_snp_effects


def subset_genotypes(
    genotypes_path: Path,
    output_path: Path,
    snp_names: List[str],
    selected_indices: List[int],
) -> Tuple[List[str], List[List[int]]]:
    selected_names = [snp_names[i] for i in selected_indices]
    ids = []
    genotypes = []
    with genotypes_path.open(newline="") as fh, output_path.open(
        "w", newline=""
    ) as out:
        reader = csv.reader(fh)
        writer = csv.writer(out)
        header = next(reader)
        out_header = [header[0]] + selected_names
        writer.writerow(out_header)
        for row in reader:
            if not row:
                continue
            ids.append(row[0])
            selected_vals = []
            for i in selected_indices:
                parsed = parse_genotype(row[i + 1])
                if parsed is None:
                    raise ValueError(
                        f"Non-integer genotype encountered at row {row[0]}, SNP {snp_names[i]}"
                    )
                selected_vals.append(parsed)
            genotypes.append(selected_vals)
            writer.writerow([row[0]] + [str(v) for v in selected_vals])
    return ids, genotypes


def simulate_phenotypes(
    ids: List[str],
    genetic_values: List[float],
    omics_values: List[List[float]],
    omics_genetic_values: List[List[float]],
    omics_effects: List[float],
    output_path: Path,
    effects_path: Path,
    summary_path: Path,
    num_groups: int,
    num_litters: int,
    group_sd: float,
    litter_sd: float,
    residual_sd: float,
    snp_effects: Optional[List[float]],
    snp_names: Optional[List[str]],
) -> None:
    group_effects = [random.gauss(0.0, group_sd) for _ in range(num_groups)]
    litter_effects = [random.gauss(0.0, litter_sd) for _ in range(num_litters)]
    num_omics = len(omics_effects)
    phenotype_values: List[float] = []
    group_values: List[float] = []
    litter_values: List[float] = []
    residual_values: List[float] = []
    omics_contrib_values: List[float] = []
    genetic_total_values: List[float] = []
    genetic_indirect_values: List[float] = []
    per_omic_indirect_values: List[List[float]] = [
        [] for _ in range(num_omics)
    ]

    with output_path.open("w", newline="") as pheno_out:
        writer = csv.writer(pheno_out)
        header = ["ID", "trait1", "group", "litter", "genetic"] + [
            f"omic{i + 1}" for i in range(num_omics)
        ] + ["genetic_direct", "genetic_indirect", "genetic_total"]
        writer.writerow(header)
        for idx, id_ in enumerate(ids):
            group = random.randrange(num_groups)
            litter = random.randrange(num_litters)
            genetic = genetic_values[idx] if idx < len(genetic_values) else 0.0
            omics_row = omics_values[idx] if idx < len(omics_values) else []
            omics_genetic_row = (
                omics_genetic_values[idx] if idx < len(omics_genetic_values) else []
            )
            omics_contrib = sum(
                val * effect for val, effect in zip(omics_row, omics_effects)
            )
            omics_genetic_contribs = [
                val * effect for val, effect in zip(omics_genetic_row, omics_effects)
            ]
            genetic_indirect = sum(omics_genetic_contribs)
            genetic_total = genetic + genetic_indirect
            residual = random.gauss(0.0, residual_sd)
            y = (
                group_effects[group]
                + litter_effects[litter]
                + genetic
                + omics_contrib
                + residual
            )
            writer.writerow(
                [id_, f"{y:.6f}", group, litter, f"{genetic:.6f}"]
                + [f"{val:.6f}" for val in omics_row]
                + [f"{genetic:.6f}", f"{genetic_indirect:.6f}", f"{genetic_total:.6f}"]
            )
            phenotype_values.append(y)
            group_values.append(group_effects[group])
            litter_values.append(litter_effects[litter])
            residual_values.append(residual)
            omics_contrib_values.append(omics_contrib)
            genetic_total_values.append(genetic_total)
            genetic_indirect_values.append(genetic_indirect)
            for i, val in enumerate(omics_genetic_contribs):
                per_omic_indirect_values[i].append(val)

    with effects_path.open("w", newline="") as eff_out:
        writer = csv.writer(eff_out)
        writer.writerow(["type", "id", "effect"])
        for i, val in enumerate(group_effects):
            writer.writerow(["group", i, f"{val:.6f}"])
        for i, val in enumerate(litter_effects):
            writer.writerow(["litter", i, f"{val:.6f}"])
        for i, val in enumerate(omics_effects):
            writer.writerow(["omic_effect", f"omic{i + 1}", f"{val:.6f}"])
        if snp_effects and snp_names:
            for name, val in zip(snp_names, snp_effects):
                writer.writerow(["snp", name, f"{val:.6f}"])

    with summary_path.open("w", newline="") as summary_out:
        writer = csv.writer(summary_out)
        writer.writerow(["component", "variance"])
        writer.writerow(["phenotype", f"{variance(phenotype_values):.6f}"])
        writer.writerow(["group", f"{variance(group_values):.6f}"])
        writer.writerow(["litter", f"{variance(litter_values):.6f}"])
        writer.writerow(["omics", f"{variance(omics_contrib_values):.6f}"])
        writer.writerow(["genetic_total", f"{variance(genetic_total_values):.6f}"])
        writer.writerow(["genetic_direct", f"{variance(genetic_values):.6f}"])
        writer.writerow(["genetic_indirect", f"{variance(genetic_indirect_values):.6f}"])
        for i, values in enumerate(per_omic_indirect_values):
            writer.writerow([f"genetic_indirect_omic{i + 1}", f"{variance(values):.6f}"])
        writer.writerow(["residual", f"{variance(residual_values):.6f}"])


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    genotypes_path = args.genotypes
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not genotypes_path.exists():
        raise FileNotFoundError(f"Genotypes file not found: {genotypes_path}")

    snp_names, valid_indices = find_integer_only_snps(genotypes_path)
    if len(valid_indices) < args.num_snps:
        raise ValueError(
            f"Only {len(valid_indices)} integer-only SNPs available; "
            f"requested {args.num_snps}."
        )

    valid_indices.sort()
    selected_indices = select_evenly_spaced_indices(valid_indices, args.num_snps)
    if len(selected_indices) != args.num_snps:
        raise ValueError(
            f"Could only select {len(selected_indices)} unique SNPs evenly spaced; "
            f"requested {args.num_snps}."
        )

    geno_out = output_dir / "genotypes_1000snps.txt"
    ids, genotypes = subset_genotypes(genotypes_path, geno_out, snp_names, selected_indices)
    selected_names = [snp_names[i] for i in selected_indices]

    pheno_out = output_dir / "phenotypes_sim.txt"
    effects_out = output_dir / "effects_sim.txt"
    genetic_values = [0.0 for _ in ids]
    snp_effects = None
    residual_sd = args.residual_sd
    if not 0.0 <= args.omics_heritability <= 1.0:
        raise ValueError("omics-heritability must be between 0 and 1.")
    omics_values, omics_genetic_values, omics_snp_effects = simulate_omics_from_blocks(
        genotypes,
        args.num_omics,
        args.omics_heritability,
        args.snp_effect_sd,
    )
    omics_effects = [
        random.gauss(0.0, args.omics_effect_sd) for _ in range(args.num_omics)
    ]
    if args.heritability is not None:
        if not 0.0 <= args.heritability <= 1.0:
            raise ValueError("heritability must be between 0 and 1.")
        direct_raw, snp_effects_raw = compute_genetic_values_unscaled(
            genotypes, args.snp_effect_sd
        )
        indirect_raw = [
            sum(val * effect for val, effect in zip(row, omics_effects))
            for row in omics_genetic_values
        ]
        if variance(direct_raw) == 0.0:
            raise ValueError("Direct genetic variance is zero; cannot scale.")
        if variance(indirect_raw) == 0.0:
            raise ValueError("Indirect genetic variance is zero; cannot scale.")
        num_snps = len(genotypes[0])
        beta_indirect = [0.0 for _ in range(num_snps)]
        block_size = num_snps // args.num_omics
        for omic_idx, snp_effects_block in enumerate(omics_snp_effects):
            start = omic_idx * block_size
            for j, effect in enumerate(snp_effects_block):
                beta_indirect[start + j] = effect * omics_effects[omic_idx]
        cov_di = covariance(direct_raw, indirect_raw)
        var_i = variance(indirect_raw)
        k = cov_di / var_i
        direct_orth = [d - k * i for d, i in zip(direct_raw, indirect_raw)]
        snp_effects_orth = [
            d - k * i for d, i in zip(snp_effects_raw, beta_indirect)
        ]
        var_d_orth = variance(direct_orth)
        if var_d_orth == 0.0:
            raise ValueError("Orthogonalized direct variance is zero; cannot scale.")
        target = args.heritability
        direct_scale = (0.2 * target / var_d_orth) ** 0.5 if target > 0 else 0.0
        indirect_scale = (0.8 * target / var_i) ** 0.5 if target > 0 else 0.0
        genetic_values = [val * direct_scale for val in direct_orth]
        snp_effects = [val * direct_scale for val in snp_effects_orth]
        omics_effects = [val * indirect_scale for val in omics_effects]
        residual_sd = (1.0 - args.heritability) ** 0.5
    summary_out = output_dir / "summary_stats.csv"
    simulate_phenotypes(
        ids,
        genetic_values,
        omics_values,
        omics_genetic_values,
        omics_effects,
        pheno_out,
        effects_out,
        summary_out,
        args.num_groups,
        args.num_litters,
        args.group_sd,
        args.litter_sd,
        residual_sd,
        snp_effects,
        selected_names,
    )

    print(f"Wrote {geno_out}")
    print(f"Wrote {pheno_out}")
    print(f"Wrote {effects_out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
