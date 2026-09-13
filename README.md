# Comparative Evaluation of Reliability and Consistency in ReAct and Reflexion LLM Agents for Multi-Hop Reasoning

This repository contains the code and final reproducibility artefacts for an MSc dissertation evaluating whether a two-stage Reflexion-style self-review pipeline improves reliability over a single-pass ReAct-style pipeline on HotpotQA multi-hop question answering.

The project does not train a model or propose a new agent architecture. It performs a controlled empirical comparison under the same model, dataset subset, context construction, answer extraction, metrics, and repeated-run protocol.

## Research Objective

The central question is:

> Does adding an explicit reflection-and-revision step improve repeated-run answer reliability on multi-hop reasoning tasks, and is any improvement large enough to justify the additional computational and latency cost?

The comparison focuses on Exact Match, token-level F1, pass@5, answer consistency, lexical reasoning-trace similarity, latency, blank predictions, and execution errors.

## Dataset

The experiment uses HotpotQA, a public multi-hop question-answering dataset.

Official dataset page: https://hotpotqa.github.io/

The full HotpotQA development file is not included because it is a public source dataset and is larger than needed for dissertation reproducibility. To run new experiments, download the HotpotQA development JSON and place it at:

```text
data/raw/hotpot_dev.json
```

The exact processed final subset used in the dissertation is included in:

```text
outputs/runs/2026_07_18_14_46_17_openai_gpt-5-mini_150q_5runs/hotpot_subset_150.json
```

Final subset:

- 150 questions
- 126 bridge questions
- 24 comparison questions
- all 150 sampled records labelled hard in HotpotQA
- fixed seed `42`
- six context documents per question

## Agent Pipelines

### ReAct-style pipeline

The ReAct-style condition receives the question and selected HotpotQA context, then asks the model to produce a structured response:

```text
Thought:
Action:
Observation:
Final Answer:
```

`ReadContext` is not a live external tool call. The selected context is already provided in the prompt, so this is a controlled ReAct-style reasoning pipeline rather than a full interactive tool-using ReAct agent. Each ReAct execution uses one model request.

### Reflexion-style pipeline

The Reflexion-style condition uses two model requests:

1. Initial answer from the question and context.
2. Reflection and revision using the question, context, and first attempt.

The second request is the architectural intervention under evaluation. ReAct is not given an artificial second call because that would change the architecture being studied.

## Final Experiment Scale

```text
150 questions
x 5 repeated runs
x 2 agent conditions
= 1,500 agent executions
```

Underlying model requests:

```text
ReAct:      150 x 5 x 1 =   750
Reflexion: 150 x 5 x 2 = 1,500
Total:                   2,250
```

Final model configuration:

| Setting | Value |
|---|---|
| Provider | OpenAI |
| Model | `gpt-5-mini` |
| Questions | 150 |
| Runs per agent | 5 |
| Seed | 42 |
| Max output tokens | 1500 |
| Requested reasoning effort | `low` |
| Configured/logged temperature | `0.2` |
| Effective GPT-5 request temperature | `1.0` |

The code records `temperature = 0.2`, but the GPT-5 compatibility logic in `src/llm_client.py` sends `temperature = 1.0` for GPT-5-family requests. Both agents used the same client behaviour.

## Final Results

| Metric | ReAct | Reflexion |
|---|---:|---:|
| Exact Match | 0.6107 | 0.6200 |
| F1 | 0.7937 | 0.8027 |
| pass@5 | 0.6533 | 0.6667 |
| Answer consistency | 0.9347 | 0.9427 |
| Trace similarity | 0.6853 | 0.6307 |
| Mean latency | 2.8596 s | 5.4458 s |
| Blank predictions | 0 | 0 |

At a descriptive level, Reflexion produced slightly higher answer-level scores, while ReAct produced higher lexical trace similarity and substantially lower latency.

Primary paired inferential results:

| Metric | Difference, Reflexion minus ReAct | Primary p-value | Interpretation |
|---|---:|---:|---|
| Exact Match | +0.0093 | 0.511 | No clear architecture difference |
| F1 | +0.0090 | 0.572 | No clear architecture difference |
| pass@5 | +0.0133 | 0.727 | No clear architecture difference |
| Answer consistency | +0.0080 | 0.369 | No clear architecture difference |
| Trace similarity | -0.0546 | < 0.001 | ReAct has higher lexical trace stability |
| Latency | +2.5861 s | < 0.001 | Reflexion is substantially slower |

The final conclusion is a reliability-efficiency trade-off rather than a universal win for either architecture.

## Repository Structure

```text
src/config.py                 Environment and path configuration
src/data_loader.py            HotpotQA loading and context preparation
src/llm_client.py             Mock, Ollama, OpenAI, and Gemini clients
src/agents.py                 ReAct-style and Reflexion-style pipelines
src/metrics.py                Metric implementations
src/run_experiment.py         Experiment runner
src/analyse_results.py        Descriptive aggregation and run summaries
src/statistical_tests.py      Reconstructed inferential analysis stage

tests/                        Dataset/context-selection tests
outputs/runs/...150q_5runs/   Final dissertation run artefacts
outputs/reconstructed_statistics/
                               Validated reconstructed statistical summary
```

## Reproducibility Artefacts

The final dissertation run is stored at:

```text
outputs/runs/2026_07_18_14_46_17_openai_gpt-5-mini_150q_5runs/
```

It contains:

- `experiment_logs.csv`
- `question_level_metrics.csv`
- `summary_metrics.csv`
- `run_config.json`
- `run_summary.md`
- `hotpot_subset_150.json`

The reconstructed statistical validation summary is stored at:

```text
outputs/reconstructed_statistics/statistical_validation_summary.md
```

Provenance note: `src/statistical_tests.py` is a later validated reconstruction of the documented inferential-analysis procedure using the frozen final experiment artefacts. It is not claimed to be the original missing local script. It is included so the dissertation's inferential results can be regenerated and audited from the preserved outputs.

## Setup

Python 3.10 or newer is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a local `.env` file when using hosted providers. Do not commit `.env`.

Mock provider example:

```env
LLM_PROVIDER=mock
TEMPERATURE=0.2
MAX_OUTPUT_TOKENS=700
```

OpenAI example:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
OPENAI_REASONING_EFFORT=low
TEMPERATURE=0.2
MAX_OUTPUT_TOKENS=1500
```

The mock provider is for pipeline validation only and should not be used for accuracy claims.

## Running Experiments

After downloading HotpotQA to `data/raw/hotpot_dev.json`, run a small test experiment:

```bash
python -m src.run_experiment --raw_data data/raw/hotpot_dev.json --sample_size 10 --runs_per_agent 2 --seed 42
```

Run the final-scale experiment:

```bash
python -m src.run_experiment --raw_data data/raw/hotpot_dev.json --sample_size 150 --runs_per_agent 5 --seed 42
```

Analyse a run descriptively:

```bash
python -m src.analyse_results --input outputs/runs/<run-id>/experiment_logs.csv
```

Reproduce the reconstructed inferential analysis for the preserved final run:

```bash
python -m src.statistical_tests --run-dir outputs/runs/2026_07_18_14_46_17_openai_gpt-5-mini_150q_5runs --output-dir outputs/reconstructed_statistics
```

Run tests:

```bash
python -m unittest discover -s tests
```

## Structural Validation

The final output audit confirmed:

- 1,500 execution rows
- 750 ReAct executions
- 750 Reflexion executions
- 150 unique questions
- exactly 5 runs per agent-question pair
- 300 question-agent summaries
- 0 blank predictions
- 0 logged execution errors
- all final statistical validation checks passed

## Limitations

The ReAct implementation uses textual Thought/Action/Observation prompting over supplied context rather than a live external-tool loop.

The Reflexion implementation uses a two-request self-review process without persistent cross-question memory or external evaluator feedback.

All gold supporting documents are included in the supplied context, with distractors. The experiment evaluates reasoning over provided evidence rather than open-world retrieval.

Supporting documents appear before distractors. Both agents receive the same ordering, so the comparison remains controlled, but the setting may be easier than fully randomised context order.

Reasoning-trace similarity is lexical Jaccard similarity. It does not prove semantic equivalence, reasoning faithfulness, or internal model correctness.

Reflexion uses two model requests per execution while ReAct uses one. Latency captures part of this cost difference, but token usage and monetary cost were not logged in a complete reproducible form.

The run used the `gpt-5-mini` model alias rather than a stored versioned model snapshot. Future provider-side model changes could affect exact reruns.

The 150-question sample was selected for experimental feasibility rather than through a formal a priori power analysis, so small architecture effects may be difficult to detect.

## Thesis Conclusion

Reflexion achieved slightly higher descriptive Exact Match, F1, pass@5, and answer consistency than ReAct, but these improvements were small and not statistically decisive. ReAct was substantially faster and showed higher lexical trace consistency. In this controlled HotpotQA setting, the second Reflexion request provided occasional useful corrections but did not establish a decisive answer-level reliability advantage.
