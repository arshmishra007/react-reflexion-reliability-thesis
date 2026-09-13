"""Reconstructed inferential analysis for the final ReAct vs Reflexion thesis run.

IMPORTANT PROVENANCE NOTE
-------------------------
This file is a reconstruction of the documented inferential-analysis procedure using
only the frozen final experiment artefacts. It is NOT claimed to be the original local
analysis script. It was reconstructed so that the inferential results reported in the
dissertation can be regenerated and audited from the preserved outputs.

Expected final-run files inside --run-dir:
    experiment_logs.csv
    question_level_metrics.csv
    hotpot_subset_150.json

The implementation reproduces the dissertation analysis specification:
- 150 paired question-level statistical units
- Wilcoxon signed-rank tests for EM, F1, consistency, trace similarity and latency
- Exact McNemar test for pass@5
- 10,000 paired percentile-bootstrap resamples using numpy default_rng(seed=42)
- Holm correction across six primary comparisons
- matched rank-biserial effect sizes for Wilcoxon outcomes
- discordant-pair odds ratio for pass@5
- question-level directional counts
- bridge/comparison subgroup summaries
- unstable-question counts
- Reflexion initial-to-revised answer analysis

The bootstrap generator is intentionally shared across metrics in the fixed metric order
below. This detail is necessary to reproduce the confidence intervals preserved in the
final dissertation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
DIRECTION_TOLERANCE = 1e-12

# Fixed order used by the original documented bootstrap stage. Do not reorder unless
# you intentionally accept different (but statistically equivalent) Monte-Carlo CIs.
PRIMARY_METRICS: List[Tuple[str, str, str]] = [
    ("Exact Match", "avg_em", "wilcoxon"),
    ("F1 score", "avg_f1", "wilcoxon"),
    ("pass@5", "pass_at_k", "mcnemar"),
    ("Answer consistency", "answer_consistency", "wilcoxon"),
    ("Trace similarity", "trace_similarity", "wilcoxon"),
    ("Latency (seconds)", "avg_latency_seconds", "wilcoxon"),
]


EXPECTED_FINAL_VALUES: Dict[str, Dict[str, float]] = {
    "Exact Match": {
        "difference": 0.009333333333333334,
        "ci_low": -0.012000,
        "ci_high": 0.030667,
        "p": 0.5105691712089317,
    },
    "F1 score": {
        "difference": 0.008967794847051814,
        "ci_low": -0.006517,
        "ci_high": 0.026737,
        "p": 0.5722573554408403,
    },
    "pass@5": {
        "difference": 0.013333333333333334,
        "ci_low": -0.020000,
        "ci_high": 0.053333,
        "p": 0.7265625,
    },
    "Answer consistency": {
        "difference": 0.008,
        "ci_low": -0.010667,
        "ci_high": 0.026667,
        "p": 0.3694444414483904,
    },
    "Trace similarity": {
        "difference": -0.05457270560108853,
        "ci_low": -0.067386,
        "ci_high": -0.041979,
        "p": 6.904830649480084e-13,
    },
    "Latency (seconds)": {
        "difference": 2.586122666666667,
        "ci_low": 2.475460,
        "ci_high": 2.702645,
        "p": 2.2995481506182782e-26,
    },
}


def normalize_answer(text: object) -> str:
    """HotpotQA-style normalisation used by the project metrics implementation."""
    s = "" if pd.isna(text) else str(text)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(prediction: object, gold: object) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_last_final_answer(text: object) -> str:
    """Extract the last Final Answer marker, with Answer fallback."""
    s = "" if pd.isna(text) else str(text)
    matches = list(re.finditer(r"Final Answer\s*:\s*(.*)", s, flags=re.I))
    if matches:
        return matches[-1].group(1).strip().split("\n")[0].strip()
    matches = list(re.finditer(r"Answer\s*:\s*(.*)", s, flags=re.I))
    if matches:
        return matches[-1].group(1).strip().split("\n")[0].strip()
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def load_final_data(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q_path = run_dir / "question_level_metrics.csv"
    logs_path = run_dir / "experiment_logs.csv"
    subset_path = run_dir / "hotpot_subset_150.json"
    for path in (q_path, logs_path, subset_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    q = pd.read_csv(q_path)
    logs = pd.read_csv(logs_path)
    with subset_path.open("r", encoding="utf-8") as f:
        subset_json = json.load(f)
    subset = pd.DataFrame(subset_json)

    required_q_cols = {
        "agent_type", "question_id", "avg_em", "avg_f1", "pass_at_k",
        "answer_consistency", "trace_similarity", "avg_latency_seconds",
    }
    missing = required_q_cols.difference(q.columns)
    if missing:
        raise ValueError(f"question_level_metrics.csv missing columns: {sorted(missing)}")

    if len(q) != 300 or q["question_id"].nunique() != 150:
        raise ValueError(
            "Expected 300 question-pipeline rows and 150 unique questions; "
            f"observed {len(q)} rows and {q['question_id'].nunique()} questions."
        )
    if len(logs) != 1500:
        raise ValueError(f"Expected 1,500 execution rows; observed {len(logs)}")

    q["agent_type"] = q["agent_type"].str.lower()
    return q, logs, subset


def paired_vectors(q: pd.DataFrame, column: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wide = q.pivot(index="question_id", columns="agent_type", values=column).sort_index()
    if not {"react", "reflexion"}.issubset(wide.columns):
        raise ValueError(f"Both react and reflexion rows are required for {column}")
    react = wide["react"].to_numpy(dtype=float)
    reflexion = wide["reflexion"].to_numpy(dtype=float)
    return react, reflexion, reflexion - react


def paired_bootstrap_ci(diff: np.ndarray, rng: np.random.Generator) -> Tuple[float, float]:
    n = len(diff)
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for i in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, n)
        means[i] = diff[idx].mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def matched_rank_biserial(diff: np.ndarray, tolerance: float = DIRECTION_TOLERANCE) -> float:
    """Matched rank-biserial correlation.

    Tiny floating-point differences are treated as ties for the effect-size calculation,
    matching the dissertation's directional-count convention. The Wilcoxon p-value itself
    is calculated on the raw paired differences to reproduce the preserved primary p-value.
    """
    d = np.asarray(diff, dtype=float).copy()
    d[np.abs(d) <= tolerance] = 0.0
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    ranks = rankdata(np.abs(nz), method="average")
    w_pos = float(ranks[nz > 0].sum())
    w_neg = float(ranks[nz < 0].sum())
    denom = w_pos + w_neg
    return 0.0 if denom == 0 else (w_pos - w_neg) / denom


def exact_mcnemar_p(react: np.ndarray, reflexion: np.ndarray) -> Tuple[float, int, int, float]:
    reflexion_only = int(np.sum((reflexion == 1) & (react == 0)))
    react_only = int(np.sum((react == 1) & (reflexion == 0)))
    discordant = reflexion_only + react_only
    if discordant == 0:
        p = 1.0
    else:
        p = float(binomtest(min(reflexion_only, react_only), discordant, 0.5, alternative="two-sided").pvalue)
    odds_ratio = math.inf if react_only == 0 and reflexion_only > 0 else (
        1.0 if react_only == 0 and reflexion_only == 0 else reflexion_only / react_only
    )
    return p, reflexion_only, react_only, float(odds_ratio)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def directional_counts(diff: np.ndarray, metric_name: str) -> Tuple[int, int, int]:
    d = np.asarray(diff, dtype=float)
    pos = int(np.sum(d > DIRECTION_TOLERANCE))
    neg = int(np.sum(d < -DIRECTION_TOLERANCE))
    tie = int(len(d) - pos - neg)
    # Positive is Reflexion - ReAct. For latency, lower is favourable, so direction reverses.
    if metric_name == "Latency (seconds)":
        return neg, pos, tie  # Reflexion favoured, ReAct favoured, tie
    return pos, neg, tie


def primary_analysis(q: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    raw_p = []

    for metric_name, column, test_kind in PRIMARY_METRICS:
        react, reflexion, diff = paired_vectors(q, column)
        mean_diff = float(diff.mean())
        ci_low, ci_high = paired_bootstrap_ci(diff, rng)
        ref_fav, react_fav, ties = directional_counts(diff, metric_name)

        if test_kind == "mcnemar":
            p_value, ref_only, react_only, effect_size = exact_mcnemar_p(react, reflexion)
            test_name = "Exact McNemar"
            effect_label = "discordant-pair OR"
            test_statistic = np.nan
        else:
            result = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="auto")
            p_value = float(result.pvalue)
            test_statistic = float(result.statistic)
            effect_size = float(matched_rank_biserial(diff))
            effect_label = "matched rank-biserial r_rb"
            test_name = "Wilcoxon signed-rank"
            ref_only = react_only = np.nan

        raw_p.append(p_value)
        rows.append({
            "metric": metric_name,
            "react_mean": float(np.mean(react)),
            "reflexion_mean": float(np.mean(reflexion)),
            "difference_reflexion_minus_react": mean_diff,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "primary_test": test_name,
            "test_statistic": test_statistic,
            "primary_p": p_value,
            "effect_size_label": effect_label,
            "effect_size": effect_size,
            "reflexion_favoured": ref_fav,
            "react_favoured": react_fav,
            "ties": ties,
            "reflexion_only_pass": ref_only,
            "react_only_pass": react_only,
        })

    holm = holm_adjust(raw_p)
    for row, adjusted_p in zip(rows, holm):
        row["holm_p"] = float(adjusted_p)
    return pd.DataFrame(rows)


def subgroup_analysis(q: pd.DataFrame, subset: pd.DataFrame) -> pd.DataFrame:
    type_map = subset.set_index("question_id")["type"].to_dict()
    work = q.copy()
    work["question_type"] = work["question_id"].map(type_map)
    if work["question_type"].isna().any():
        raise ValueError("Could not map question type for every question-level record")
    out = (
        work.groupby(["question_type", "agent_type"], as_index=False)
        .agg(
            n_questions=("question_id", "nunique"),
            em=("avg_em", "mean"),
            f1=("avg_f1", "mean"),
            pass_at_5=("pass_at_k", "mean"),
            consistency=("answer_consistency", "mean"),
            trace_similarity=("trace_similarity", "mean"),
            latency=("avg_latency_seconds", "mean"),
        )
    )
    return out


def unstable_question_summary(q: pd.DataFrame) -> pd.DataFrame:
    unstable = q[q["answer_consistency"] < 1.0 - DIRECTION_TOLERANCE]
    return (
        unstable.groupby("agent_type", as_index=False)
        .agg(
            unstable_questions=("question_id", "nunique"),
            minimum_consistency=("answer_consistency", "min"),
        )
    )


def reflexion_revision_analysis(logs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reflex = logs[logs["agent_type"].str.lower() == "reflexion"].copy()
    details = []

    for _, row in reflex.iterrows():
        trace = "" if pd.isna(row.get("reasoning_trace")) else str(row["reasoning_trace"])
        if "REFLECTION_ATTEMPT:" in trace:
            first_text, second_text = trace.split("REFLECTION_ATTEMPT:", 1)
            first_text = first_text.replace("FIRST_ATTEMPT:", "", 1)
        else:
            # Conservative fallback for older log formats.
            first_text = trace
            second_text = "" if pd.isna(row.get("raw_output")) else str(row["raw_output"])

        initial_answer = extract_last_final_answer(first_text)
        revised_answer = extract_last_final_answer(second_text)
        gold = row["gold_answer"]

        initial_em = int(normalize_answer(initial_answer) == normalize_answer(gold))
        revised_em = int(normalize_answer(revised_answer) == normalize_answer(gold))
        initial_f1 = token_f1(initial_answer, gold)
        revised_f1 = token_f1(revised_answer, gold)
        changed = normalize_answer(initial_answer) != normalize_answer(revised_answer)

        details.append({
            "question_id": row["question_id"],
            "run_number": row["run_number"],
            "initial_answer": initial_answer,
            "revised_answer": revised_answer,
            "answer_changed": changed,
            "initial_em": initial_em,
            "revised_em": revised_em,
            "initial_f1": initial_f1,
            "revised_f1": revised_f1,
        })

    d = pd.DataFrame(details)
    summary = pd.DataFrame([
        {"outcome": "Final answer changed", "count": int(d["answer_changed"].sum())},
        {"outcome": "Final answer unchanged", "count": int((~d["answer_changed"]).sum())},
        {"outcome": "Wrong to Exact Match", "count": int(((d["initial_em"] == 0) & (d["revised_em"] == 1)).sum())},
        {"outcome": "Exact Match to wrong", "count": int(((d["initial_em"] == 1) & (d["revised_em"] == 0)).sum())},
        {"outcome": "F1 improved", "count": int((d["revised_f1"] > d["initial_f1"] + DIRECTION_TOLERANCE).sum())},
        {"outcome": "F1 decreased", "count": int((d["revised_f1"] < d["initial_f1"] - DIRECTION_TOLERANCE).sum())},
        {"outcome": "F1 unchanged", "count": int((np.abs(d["revised_f1"] - d["initial_f1"]) <= DIRECTION_TOLERANCE).sum())},
    ])
    summary["percentage"] = summary["count"] / len(d) * 100.0
    return summary, d


def validate_against_dissertation(primary: pd.DataFrame) -> pd.DataFrame:
    checks = []
    for _, row in primary.iterrows():
        metric = row["metric"]
        expected = EXPECTED_FINAL_VALUES[metric]
        # Thesis values were rounded to six decimals in Appendix B.
        observed = {
            "difference": row["difference_reflexion_minus_react"],
            "ci_low": row["bootstrap_ci_low"],
            "ci_high": row["bootstrap_ci_high"],
            "p": row["primary_p"],
        }
        for field, observed_value in observed.items():
            expected_value = expected[field]
            tol = 5e-7 if field != "p" else max(5e-7, abs(expected_value) * 1e-6)
            checks.append({
                "metric": metric,
                "field": field,
                "expected": expected_value,
                "observed": observed_value,
                "absolute_difference": abs(observed_value - expected_value),
                "matches_reported_value": abs(observed_value - expected_value) <= tol,
            })
    return pd.DataFrame(checks)


def write_markdown_summary(
    output_path: Path,
    primary: pd.DataFrame,
    subgroups: pd.DataFrame,
    unstable: pd.DataFrame,
    revisions: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    all_ok = bool(validation["matches_reported_value"].all())
    lines = [
        "# Reconstructed Statistical Analysis Validation",
        "",
        "> Provenance: this is a reconstructed analysis stage based on the frozen final outputs; it is not represented as the original missing local script.",
        "",
        f"- Primary statistical unit: 150 paired questions",
        f"- Bootstrap: {BOOTSTRAP_RESAMPLES:,} paired resamples, seed {BOOTSTRAP_SEED}",
        f"- Validation against dissertation Appendix B values: **{'PASS' if all_ok else 'CHECK REQUIRED'}**",
        "",
        "## Primary results",
        "",
        primary.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Question-type summaries",
        "",
        subgroups.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Unstable-question summary",
        "",
        unstable.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Reflexion revision summary",
        "",
        revisions.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Validation checks",
        "",
        validation.to_markdown(index=False, floatfmt=".8g"),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the final paired inferential analysis.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Final run directory containing experiment_logs.csv, question_level_metrics.csv and hotpot_subset_150.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for regenerated statistical artefacts (default: <run-dir>/reconstructed_statistics)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "reconstructed_statistics")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    q, logs, subset = load_final_data(run_dir)
    primary = primary_analysis(q)
    subgroups = subgroup_analysis(q, subset)
    unstable = unstable_question_summary(q)
    revisions, revision_details = reflexion_revision_analysis(logs)
    validation = validate_against_dissertation(primary)

    primary.to_csv(output_dir / "primary_paired_statistics.csv", index=False)
    subgroups.to_csv(output_dir / "question_type_subgroups.csv", index=False)
    unstable.to_csv(output_dir / "unstable_question_summary.csv", index=False)
    revisions.to_csv(output_dir / "reflexion_revision_summary.csv", index=False)
    revision_details.to_csv(output_dir / "reflexion_revision_details.csv", index=False)
    validation.to_csv(output_dir / "dissertation_value_validation.csv", index=False)
    write_markdown_summary(
        output_dir / "statistical_validation_summary.md",
        primary,
        subgroups,
        unstable,
        revisions,
        validation,
    )

    print(primary.to_string(index=False))
    print("\nValidation against dissertation Appendix B:")
    print(validation.to_string(index=False))
    print(f"\nAll reported values reproduced: {validation['matches_reported_value'].all()}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
