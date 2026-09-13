from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.agents import ReActAgent, ReflexionAgent
from src.data_loader import build_subset, save_json
from src.llm_client import get_llm_client
from src.metrics import exact_match, f1_score, normalize_answer


def _safe_name(value: str) -> str:
    safe = str(value).strip() or "unknown"
    for old, new in [
        (":", "_"),
        ("/", "_"),
        ("\\", "_"),
        (" ", "-"),
    ]:
        safe = safe.replace(old, new)
    return "".join(ch for ch in safe if ch.isalnum() or ch in {"_", "-", "."})


def create_run_dir(provider: str, model: str, sample_size: int, runs_per_agent: int) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_id = (
        f"{timestamp}_{_safe_name(provider)}_{_safe_name(model)}_"
        f"{sample_size}q_{runs_per_agent}runs"
    )
    run_dir = config.OUTPUT_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_config(run_dir: Path, run_config: dict) -> None:
    config_path = run_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)


def run_experiment(raw_data: str, sample_size: int, runs_per_agent: int, seed: int, context_mode: str) -> Path:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    llm = get_llm_client()
    run_dir = create_run_dir(
        provider=llm.provider,
        model=llm.model,
        sample_size=sample_size,
        runs_per_agent=runs_per_agent,
    )

    subset = build_subset(raw_data, sample_size=sample_size, seed=seed, context_mode=context_mode)
    subset_path = run_dir / f"hotpot_subset_{sample_size}.json"
    save_json(subset, subset_path)

    agents = {
        "react": ReActAgent(llm),
        "reflexion": ReflexionAgent(llm),
    }

    output_path = run_dir / "experiment_logs.csv"
    save_run_config(run_dir, {
        "provider": llm.provider,
        "model": llm.model,
        "raw_data": str(raw_data),
        "sample_size": sample_size,
        "runs_per_agent": runs_per_agent,
        "seed": seed,
        "context_mode": context_mode,
        "agents": list(agents.keys()),
        "temperature": config.TEMPERATURE,
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        "openai_reasoning_effort": config.OPENAI_REASONING_EFFORT,
        "output_dir": str(run_dir),
        "experiment_logs": str(output_path),
        "processed_subset": str(subset_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })

    fieldnames = [
        "timestamp",
        "question_id",
        "agent_type",
        "run_number",
        "question",
        "gold_answer",
        "predicted_answer",
        "normalized_predicted_answer",
        "exact_match",
        "f1_score",
        "raw_output",
        "reasoning_trace",
        "error",
        "latency_seconds",
        "provider",
        "model",
        "temperature",
        "context_mode",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in subset:
            for agent_name, agent in agents.items():
                for run_number in range(1, runs_per_agent + 1):
                    start = time.time()
                    try:
                        result = agent.run(item)
                        error_text = ""
                    except Exception as exc:
                        result = None
                        error_text = f"ERROR: {exc}"
                    latency = round(time.time() - start, 3)

                    predicted = result.predicted_answer if result else ""
                    raw_output = result.raw_output if result else ""
                    trace = result.reasoning_trace if result else error_text
                    logged_error = result.error if result else error_text
                    writer.writerow({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "question_id": item["question_id"],
                        "agent_type": agent_name,
                        "run_number": run_number,
                        "question": item["question"],
                        "gold_answer": item["gold_answer"],
                        "predicted_answer": predicted,
                        "normalized_predicted_answer": normalize_answer(predicted),
                        "exact_match": exact_match(predicted, item["gold_answer"]),
                        "f1_score": f1_score(predicted, item["gold_answer"]),
                        "raw_output": raw_output,
                        "reasoning_trace": trace,
                        "error": logged_error,
                        "latency_seconds": latency,
                        "provider": llm.provider,
                        "model": llm.model,
                        "temperature": config.TEMPERATURE,
                        "context_mode": context_mode,
                    })
                    print(f"Done: {item['question_id']} | {agent_name} | run {run_number} | latency={latency}s")

    print(f"\nRun directory: {run_dir}")
    print(f"Experiment logs: {output_path}")
    print(f"Run config: {run_dir / 'run_config.json'}")
    print(f"Processed subset: {subset_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReAct vs Reflexion repeated-execution experiment.")
    parser.add_argument("--raw_data", required=True, help="Path to HotpotQA dev JSON file.")
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--runs_per_agent", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context_mode", default="supporting_plus_distractors")
    args = parser.parse_args()

    run_experiment(
        raw_data=args.raw_data,
        sample_size=args.sample_size,
        runs_per_agent=args.runs_per_agent,
        seed=args.seed,
        context_mode=args.context_mode,
    )


if __name__ == "__main__":
    main()
