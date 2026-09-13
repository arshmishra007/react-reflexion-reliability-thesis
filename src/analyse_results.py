from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import config
from src.metrics import consistency_score, pass_at_k, simple_trace_similarity


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def save_run_summary(output_dir: Path, summary: pd.DataFrame, qdf: pd.DataFrame, total_rows: int) -> Path:
    summary_path = output_dir / "run_summary.md"

    blank_predictions = 0
    if "blank_predictions" in summary.columns:
        blank_predictions = int(summary["blank_predictions"].sum())

    latency_summary = ""
    if not qdf.empty and "avg_latency_seconds" in qdf.columns:
        latency_summary = (
            f"- Mean question-level latency: {qdf['avg_latency_seconds'].mean():.3f} seconds\n"
            f"- Maximum question-level latency: {qdf['avg_latency_seconds'].max():.3f} seconds\n"
        )

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Experiment Run Summary\n\n")
        f.write(f"Total logged rows: {total_rows}\n\n")
        f.write(f"Blank predictions: {blank_predictions}\n\n")

        f.write("## Summary Metrics\n\n")
        f.write(dataframe_to_markdown(summary))
        f.write("\n\n")

        f.write("## Runtime Notes\n\n")
        f.write(latency_summary or "- No latency data available.\n")
        f.write("\n")

        f.write("## Manual Review Notes\n\n")
        f.write("- Review blank predictions manually.\n")
        f.write("- Check whether Reflexion improves EM/F1 or mainly consistency.\n")
        f.write("- Compare latency between ReAct and Reflexion.\n")
        f.write("- Inspect low-consistency questions for common failure patterns.\n")

    return summary_path


def analyse_results(input_csv: str | Path) -> None:
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Results file not found: {input_csv}")

    #df = pd.read_csv(input_csv)
    df = pd.read_csv(input_csv, keep_default_na=False)
    output_dir = input_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    question_rows = []
    for (agent_type, question_id), group in df.groupby(["agent_type", "question_id"]):
        answers = group["predicted_answer"].fillna("").tolist()
        traces = group["reasoning_trace"].fillna("").tolist()
        exacts = group["exact_match"].fillna(0).astype(int).tolist()
        question_rows.append({
            "agent_type": agent_type,
            "question_id": question_id,
            "runs": len(group),
            "avg_em": group["exact_match"].mean(),
            "avg_f1": group["f1_score"].mean(),
            "pass_at_k": pass_at_k(exacts),
            "answer_consistency": consistency_score(answers),
            "trace_similarity": simple_trace_similarity(traces),
            "avg_latency_seconds": group["latency_seconds"].mean(),
            "blank_predictions": int((group["predicted_answer"].fillna("").astype(str).str.strip() == "").sum()),
        })

    qdf = pd.DataFrame(question_rows)
    summary = qdf.groupby("agent_type").agg(
        questions=("question_id", "count"),
        mean_em=("avg_em", "mean"),
        mean_f1=("avg_f1", "mean"),
        mean_pass_at_k=("pass_at_k", "mean"),
        mean_answer_consistency=("answer_consistency", "mean"),
        mean_trace_similarity=("trace_similarity", "mean"),
        mean_latency_seconds=("avg_latency_seconds", "mean"),
        blank_predictions=("blank_predictions", "sum"),
    ).reset_index()

    q_out = output_dir / "question_level_metrics.csv"
    s_out = output_dir / "summary_metrics.csv"
    qdf.to_csv(q_out, index=False)
    summary.to_csv(s_out, index=False)
    summary_md = save_run_summary(output_dir, summary, qdf, total_rows=len(df))

    print("\nSummary metrics:")
    print(summary.to_string(index=False))
    print(f"\nSaved question-level metrics: {q_out}")
    print(f"Saved summary metrics: {s_out}")
    print(f"Saved run summary: {summary_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse ReAct vs Reflexion experiment logs.")
    parser.add_argument("--input", default=str(config.OUTPUT_DIR / "experiment_logs.csv"))
    args = parser.parse_args()
    analyse_results(args.input)


if __name__ == "__main__":
    main()
