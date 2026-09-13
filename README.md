# ReAct vs Reflexion Architecture Comparison

## Project overview

This project evaluates the **reliability and consistency of two LLM reasoning pipelines — ReAct-style and Reflexion-style — on multi-hop questions from the HotpotQA dataset**.

The study is designed around one central question:

> **Does adding an explicit reflection-and-revision step improve repeated-run answer reliability on multi-hop reasoning tasks, and is any improvement large enough to justify the additional computational and latency cost?**

The project does **not** attempt to invent a new agent architecture or train a new language model. Instead, it performs a controlled comparison of two reasoning pipelines under the same model, dataset, context and evaluation conditions.

The final dissertation title is:

> **Comparative Evaluation of Reliability and Consistency in ReAct and Reflexion LLM Agents for Multi-Hop Reasoning**

---

# Research alignment

The implementation remains aligned with the original research proposal.

The proposal focused on:

- ReAct vs Reflexion comparison
- HotpotQA multi-hop reasoning
- repeated execution of the same questions
- reliability beyond single-run accuracy
- Exact Match, F1, pass@k and answer consistency
- reasoning-trace stability
- controlled experimental conditions
- quantitative analysis rather than model training

The final implementation preserves this design and extends the analysis with:

- end-to-end latency
- paired statistical tests
- 95% paired bootstrap confidence intervals
- Holm multiple-comparison correction
- formal effect sizes
- unstable-question analysis
- Reflexion first-answer vs revised-answer analysis

---

# Why HotpotQA?

HotpotQA is a multi-hop question-answering dataset.

A **multi-hop question** usually requires the model to connect information from more than one document before producing the final answer.

For example:

```text
Document A → identifies a person
Document B → gives information about that person
Question    → requires connecting both facts
```

This makes HotpotQA suitable for studying reasoning reliability rather than simple factual recall.

The final experiment used:

- **150 questions**
- **126 bridge questions**
- **24 comparison questions**
- **all 150 sampled records were labelled hard**
- **no explicit difficulty filter was applied**

---

# Agent architectures

## ReAct-style pipeline

The controlled ReAct-style pipeline receives:

- the question
- the selected HotpotQA context
- a prompt requesting a structured reasoning trace

The model is instructed to produce:

```text
Thought:
...

Action:
ReadContext

Observation:
...

Final Answer:
...
```

### Important implementation detail

`ReadContext` is **not a live external tool call**.

The context is already supplied in the prompt. Therefore, this implementation is best described as a **controlled ReAct-style reasoning pipeline**, not a full external-tool ReAct agent.

The model generates the Thought, Action, Observation and Final Answer in **one LLM request**.

### ReAct execution flow

```text
Question + Context
        ↓
   ReAct prompt
        ↓
    LLM Call 1
        ↓
Thought
Action
Observation
Final Answer
        ↓
   Evaluation
```

---

## Reflexion-style pipeline

The simplified Reflexion-style pipeline adds an explicit self-review stage.

### Call 1 — initial attempt

The model first receives:

```text
Question + Context
```

and produces:

```text
Initial reasoning
+
Initial final answer
```

### Call 2 — reflection and revision

The entire first response is then inserted into a second prompt.

The second request asks the model to:

- review the previous attempt
- check the evidence again
- identify mistakes or unsupported reasoning
- provide a revised final answer

### Reflexion execution flow

```text
Question + Context
        ↓
    LLM Call 1
        ↓
Initial reasoning
Initial answer
        ↓
Previous attempt + Question + Context
        ↓
    LLM Call 2
        ↓
Reflection
Revised reasoning
Final Answer
        ↓
     Evaluation
```

### Why Reflexion uses two calls while ReAct uses one

The second model request is the architectural intervention being tested.

ReAct produces reasoning and an answer in a single attempt.

Reflexion first completes an answer and then receives an additional opportunity to review and revise that completed attempt.

Giving ReAct an artificial second call purely to equalise request count would change the architecture being studied.

The experiment therefore controls the model, data, context, repetitions and evaluation procedure, while allowing the **agent architecture itself** to remain the independent variable.

---

# Experimental design

For a fair comparison, both pipelines used the same:

- HotpotQA subset
- OpenAI provider
- `gpt-5-mini`
- question text
- six-document context
- context-selection process
- answer extraction
- answer normalisation
- evaluation metrics
- number of repeated runs

The independent variable is:

```text
Agent architecture
```

The measured outcomes are:

- Exact Match
- token-level F1
- pass@5
- answer consistency
- reasoning-trace similarity
- latency
- blank predictions

---

# Final experiment scale

The main experiment used:

```text
150 questions
× 5 repeated runs
× 2 agent conditions
= 1,500 agent executions
```

However, the number of underlying model requests differs by architecture.

## ReAct

```text
150 × 5 × 1 model request
= 750 model requests
```

## Reflexion

```text
150 × 5 × 2 model requests
= 1,500 model requests
```

## Total

```text
750 + 1,500
= 2,250 underlying model requests
```

This distinction is important because Reflexion's second request is part of the architecture and directly affects latency and computational cost.

---

# Context construction

The final experiment used:

```text
2 supporting documents
+
4 distractor documents
=
6 documents per question
```

The supporting documents are those referenced by the HotpotQA supporting facts.

Distractors are irrelevant documents included to make the model distinguish useful evidence from unrelated information.

A fixed seed of `42` was used for reproducible sampling and distractor selection.

## Context-order limitation

The final six-document context was **not fully shuffled**.

The implementation placed:

```text
Supporting document 1
Supporting document 2
Distractor document 1
Distractor document 2
Distractor document 3
Distractor document 4
```

Both agents received the same ordering, so the ReAct-vs-Reflexion comparison remains controlled.

However, placing supporting documents first may make evidence identification easier than a fully randomised or retrieval-based setting.

The experiment therefore evaluates:

> reasoning over supplied evidence with distractors

rather than:

> open-world document retrieval

---

# Model configuration

The final run used:

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

## Temperature note

The experiment configuration recorded:

```text
temperature = 0.2
```

However, the GPT-5-specific compatibility logic in `llm_client.py` sent:

```text
temperature = 1.0
```

for GPT-5-family requests.

Both agents used the same client behaviour, so this does not create an asymmetric ReAct-vs-Reflexion comparison.

The thesis reports this distinction explicitly for reproducibility.

## Reasoning-effort note

The client requested:

```text
reasoning_effort = low
```

for GPT-5-family requests.

The client also contained fallback logic that could retry without this parameter if unsupported.

The saved logs record the requested value, but do not prove the effective server-side value for every request.

---

# Metrics

Metrics are first calculated at execution level and then aggregated by question and agent.

The primary inferential comparison uses the **question** as the statistical unit.

Therefore:

```text
1,500 execution rows
→ 300 question-agent summaries
→ 150 paired question-level comparisons
```

## 1. Exact Match — EM

Exact Match is the strictest answer-correctness metric.

It returns:

```text
1 = exact normalized match
0 = otherwise
```

Normalization removes:

- case differences
- punctuation
- articles such as `a`, `an`, `the`
- extra whitespace

Higher is better.

## 2. Token-level F1

F1 measures token overlap between the predicted answer and the gold answer.

It combines:

```text
Precision = how much of the prediction is correct
Recall    = how much of the gold answer was captured
```

The harmonic mean is:

```text
F1 = 2PR / (P + R)
```

F1 is useful because an answer can be partially correct even when Exact Match is `0`.

Higher is better.

## 3. pass@5

Each question is executed five times per agent.

`pass@5 = 1` when at least one of the five repeated runs produces an Exact Match.

Example:

```text
Run 1: incorrect
Run 2: incorrect
Run 3: correct
Run 4: incorrect
Run 5: incorrect

pass@5 = 1
```

This measures **recoverability across repeated attempts**.

Higher is better.

This metric is the project's own `pass@5` definition and should not be confused with tau-bench's `pass^k` reliability metric.

## 4. Answer consistency

Answer consistency measures how often the most common normalized answer appears across repeated runs.

Example:

```text
Paris
Paris
Paris
Paris
Paris
```

produces:

```text
consistency = 1.0
```

Consistency measures **stability**, not correctness.

An agent can be consistently wrong.

Therefore, answer consistency must be interpreted together with EM and F1.

Higher is better when considered jointly with correctness.

## 5. Reasoning-trace similarity

Reasoning-trace similarity uses pairwise **Jaccard similarity** between normalized token sets.

For two token sets `A` and `B`:

```text
J(A, B) = |A ∩ B| / |A ∪ B|
```

Interpretation:

```text
1.0 = identical token sets
0.0 = no shared tokens
```

The final score is the average pairwise similarity across the repeated traces for a question.

### Important limitation

This is a **lexical stability metric**.

It does not prove:

- semantic equivalence
- reasoning correctness
- reasoning faithfulness
- better internal model reasoning

Two correct reasoning processes may use different wording, while two incorrect traces may use very similar wording.

## 6. Latency

Latency measures end-to-end wall-clock time for one complete agent execution.

### ReAct latency

Includes:

```text
one LLM request + answer parsing
```

### Reflexion latency

Includes:

```text
first LLM request
+ construction of the reflection prompt
+ second LLM request
+ answer parsing
```

Lower is better.

## 7. Blank predictions

Blank predictions count cases where no final answer could be extracted.

This helps identify output-format or pipeline failures that accuracy metrics alone may miss.

Lower is better.

The final experiment contained:

```text
0 blank predictions
```

---

# Final aggregate results

| Metric | ReAct | Reflexion |
|---|---:|---:|
| Exact Match | 0.6107 | **0.6200** |
| F1 | 0.7937 | **0.8027** |
| pass@5 | 0.6533 | **0.6667** |
| Answer consistency | 0.9347 | **0.9427** |
| Trace similarity | **0.6853** | 0.6307 |
| Mean latency | **2.8596 s** | 5.4458 s |
| Blank predictions | 0 | 0 |

At a descriptive level, Reflexion produced slightly higher:

- EM
- F1
- pass@5
- answer consistency

ReAct produced:

- higher lexical trace similarity
- substantially lower latency

Reflexion mean latency was approximately:

```text
5.4458 / 2.8596 ≈ 1.90×
```

the ReAct latency.

---

# Question-level paired results

## Exact Match

```text
Reflexion higher: 11
ReAct higher:      7
Tie:             132
```

## F1

```text
Reflexion higher: 14
ReAct higher:     12
Tie:             124
```

## pass@5

```text
Reflexion higher: 5
ReAct higher:     3
Tie:             142
```

## Answer consistency

```text
Reflexion higher: 14
ReAct higher:     10
Tie:             126
```

## Trace similarity

```text
Reflexion higher: 35
ReAct higher:    115
Tie:               0
```

## Latency

```text
Reflexion faster:   0
ReAct faster:     150
Tie:                 0
```

---

# Statistical analysis

The included `src/analyse_results.py` generates:

- question-level metrics
- aggregate summaries
- run summary files

The final dissertation analysis additionally used a separate statistical-analysis
script that is not included in this repository. The inferential results below are
reported for study documentation, but cannot be regenerated by
`src/analyse_results.py` alone.

The primary paired analyses use:

- Wilcoxon signed-rank test for EM
- Wilcoxon signed-rank test for F1
- Exact McNemar test for pass@5
- Wilcoxon signed-rank test for answer consistency
- Wilcoxon signed-rank test for trace similarity
- Wilcoxon signed-rank test for latency
- 10,000 paired bootstrap resamples
- fixed bootstrap seed `42`
- Holm correction across six primary comparisons
- matched rank-biserial correlation for Wilcoxon effect sizes
- discordant-pair odds ratio for pass@5

## Primary results

| Metric | Difference (Reflexion − ReAct) | Primary p-value | Interpretation |
|---|---:|---:|---|
| Exact Match | +0.0093 | 0.511 | No clear architecture difference |
| F1 | +0.0090 | 0.572 | No clear architecture difference |
| pass@5 | +0.0133 | 0.727 | No clear architecture difference |
| Answer consistency | +0.0080 | 0.369 | No clear architecture difference |
| Trace similarity | −0.0546 | < 0.001 | ReAct has higher lexical trace stability |
| Latency | +2.5861 s | < 0.001 | Reflexion is substantially slower |

The small descriptive gains in Reflexion's answer-level metrics are therefore not sufficient to establish a decisive answer-level reliability advantage.

---

# Reflexion revision analysis

Because Reflexion uses two requests, the final logs were also analysed to determine how often the second request actually changed the first answer.

Across **750 Reflexion executions**:

| Revision outcome | Count |
|---|---:|
| Final answer unchanged | 735 |
| Final answer changed | 15 |
| Wrong → Exact Match correction | 9 |
| Exact Match → wrong regression | 0 |
| F1 improved | 12 |
| F1 worsened | 3 |

This means:

- the second request changed the answer in only **15 of 750 executions**
- it corrected **9** initially wrong answers to Exact Match
- there were **0 Exact Match correct-to-incorrect regressions**
- three revisions reduced partial F1 without converting an exact answer into an incorrect one

This helps explain why Reflexion produced only modest aggregate improvements despite using twice as many model requests per execution.

---

# Main interpretation

The final conclusion is **not**:

> Reflexion is universally better than ReAct.

Instead, the experiment shows a **reliability-efficiency trade-off**.

Reflexion produced:

- slightly higher descriptive answer accuracy
- slightly higher pass@5
- slightly higher answer consistency
- occasional useful corrections through the second review step

But ReAct produced:

- much lower latency
- fewer model requests
- higher lexical trace similarity

For this controlled HotpotQA setting, the second Reflexion request therefore provided only modest answer-level gains while almost doubling average response time.

---

# Project structure

```text
data/raw/                     HotpotQA source data
data/processed/               Reusable processed subsets
outputs/runs/                 Timestamped experiment outputs

src/config.py                 Environment and path configuration
src/data_loader.py            Dataset loading and context preparation
src/llm_client.py             Mock, Ollama, OpenAI and Gemini clients
src/agents.py                 ReAct-style and Reflexion-style pipelines
src/metrics.py                Metric implementations
src/run_experiment.py         Experiment runner
src/analyse_results.py        Question-level aggregation and summary generation

tests/                        Dataset and context-selection tests
thesis_notes/                 Methodology and experiment notes
```

---

# Setup

Python 3.10 or newer is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the HotpotQA development JSON at:

```text
data/raw/hotpot_dev.json
```

Create a `.env` file in the project root when you want to override the defaults
or configure a hosted provider. If no provider is configured, the project uses
the deterministic `mock` provider.

---

# Mock-provider pipeline test

The mock provider is intended only for pipeline validation.

Example:

```env
LLM_PROVIDER=mock
TEMPERATURE=0.2
MAX_OUTPUT_TOKENS=700
```

The mock client returns a fixed test response.

It should **not** be used for reporting model accuracy.

---

# OpenAI example configuration

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
OPENAI_REASONING_EFFORT=low
TEMPERATURE=0.2
MAX_OUTPUT_TOKENS=1500
```

Remember that for the final GPT-5 run, the client compatibility logic converted the configured temperature to an effective request temperature of `1.0`.

# Other supported providers

The client also supports local Ollama and Google Gemini models.

Ollama example:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
TEMPERATURE=0.2
MAX_OUTPUT_TOKENS=700
```

Gemini example:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-flash
TEMPERATURE=0.2
MAX_OUTPUT_TOKENS=700
```

The model names above are code defaults, not guarantees that a model is still
available from its provider. Override them in `.env` as needed.

---

# Running an experiment

Example small run:

```text
python -m src.run_experiment --raw_data data/raw/hotpot_dev.json --sample_size 10 --runs_per_agent 2 --seed 42
```

Analyse the produced log:

```text
python -m src.analyse_results --input outputs/runs/<run-id>/experiment_logs.csv
```

Run tests:

```bash
python -m unittest discover -s tests
```

---

# Important command-line options

| Option | Default | Meaning |
|---|---:|---|
| `--raw_data` | required | HotpotQA JSON path |
| `--sample_size` | `10` | Number of sampled questions |
| `--runs_per_agent` | `2` | Repeated runs per question and agent |
| `--seed` | `42` | Sampling and distractor-selection seed |
| `--context_mode` | `supporting_plus_distractors` | Context-construction mode |

---

# Output files

Each experiment is stored under:

```text
outputs/runs/<run-id>/
```

The experiment runner creates:

| File | Purpose |
|---|---|
| `run_config.json` | Saved run configuration |
| `hotpot_subset_<n>.json` | Exact processed subset used |
| `experiment_logs.csv` | One row per question-agent-run execution |

Running `src.analyse_results` then adds:

| File | Purpose |
|---|---|
| `question_level_metrics.csv` | Repeated-run metrics aggregated by question and agent |
| `summary_metrics.csv` | Agent-level aggregate results |
| `run_summary.md` | Human-readable run summary |

---

# Code flow

1. `main()` in `src/run_experiment.py` reads the command-line arguments.
2. `run_experiment()` loads the client, creates the run directory, prepares the data, runs both agents and saves raw logs.
3. `get_llm_client()` selects the configured model provider.
4. `build_subset()` loads HotpotQA, samples questions and preprocesses the records.
5. `select_supporting_plus_distractors()` keeps all required supporting documents and fills remaining slots with seeded distractors.
6. `preprocess_record()` builds the structured question object and records structural context validation.
7. `ReActAgent.run()` builds the ReAct-style prompt and makes one model request.
8. `ReflexionAgent.run()` makes the first attempt, builds the reflection prompt and makes the second model request.
9. `extract_final_answer()` extracts the final answer from the model response.
10. `normalize_answer()`, `exact_match()` and `f1_score()` calculate run-level correctness.
11. `analyse_results()` groups repeated runs by agent and question and calculates repeated-run metrics.
12. `pass_at_k()`, `consistency_score()` and `simple_trace_similarity()` implement repeated-run reliability measures.
13. The dissertation's separate statistical-analysis stage produced the paired tests, bootstrap intervals, Holm correction, effect sizes, unstable-question analysis and Reflexion revision analysis reported above. That analysis script is not included in this repository; `src/analyse_results.py` provides the descriptive aggregation included here.

---

# Structural validation of the final experiment

The final output audit confirmed:

- **1,500 execution rows**
- **750 ReAct executions**
- **750 Reflexion executions**
- **150 unique questions**
- exactly **5 runs** per agent-question pair
- **0 duplicate** question-agent-run rows
- **0 blank predictions**
- **0 blank normalized predictions**
- **0 blank raw outputs**
- **0 blank reasoning traces**
- **0 logged execution errors**
- **0 invalid latency values**
- **6 documents** for every selected question
- **all required supporting titles included**
- **0 missing supporting titles**
- **150/150 context-validation checks passed**

---

# Limitations

## Simplified agent architectures

The ReAct implementation uses textual Thought–Action–Observation prompting rather than an interactive external-tool loop.

The Reflexion implementation uses a two-request self-review process without persistent cross-question memory or external evaluator feedback.

The findings therefore apply to these **controlled architecture variants**, not every possible ReAct or Reflexion implementation.

## Supporting documents are supplied

All gold supporting documents are included in the context.

The experiment evaluates reasoning over evidence with distractors rather than retrieval quality.

## Support-first context ordering

Supporting documents appear before distractors.

This may make evidence identification easier than randomly ordered context.

## Trace similarity is lexical

Jaccard similarity compares word-set overlap.

It does not establish semantic reasoning equivalence or internal reasoning correctness.

## pass@5 depends on the number of attempts

pass@5 should only be compared when both architectures use the same number of repeated runs.

## Consistency can be consistently wrong

A high answer-consistency score must always be interpreted together with EM and F1.

## Compute differs by architecture

Reflexion uses two model requests per execution while ReAct uses one.

Latency captures part of this difference.

However, token usage and monetary cost were not logged in a reproducible form.

## Hosted-model reproducibility

The run used the `gpt-5-mini` model alias rather than a stored versioned model snapshot.

Future provider-side changes could therefore affect an exact rerun.

## Sample size

The 150-question sample was selected primarily for experimental feasibility rather than through a formal a priori power analysis.

Small architecture effects may therefore be difficult to detect.

---

# Thesis conclusion

The completed experiment does not show a decisive universal winner.

Instead:

> **Reflexion produced small descriptive improvements in answer-level reliability, but those gains were not statistically decisive and came with substantially higher latency. ReAct was more computationally efficient and produced more lexically consistent reasoning traces.**

The main result is therefore a:

> **reliability-efficiency trade-off between single-pass ReAct-style reasoning and two-stage Reflexion-style self-review.**

---

# Final experiment summary

```text
Dataset:
HotpotQA development set

Questions:
150

Question types:
126 bridge
24 comparison

Runs per question per agent:
5

Agent executions:
1,500

Underlying model requests:
2,250

Provider:
OpenAI

Model:
gpt-5-mini

Configured temperature:
0.2

Effective GPT-5 request temperature:
1.0

Seed:
42

Documents per question:
6

ReAct:
1 model request per execution

Reflexion:
2 model requests per execution

Primary finding:
Small descriptive Reflexion gains,
but no decisive answer-level reliability superiority;
ReAct is substantially faster.
```

The experiment showed that Reflexion achieved slightly higher Exact Match, F1, pass@5, and answer consistency than ReAct, but these improvements were small and not statistically decisive. The main trade-off was efficiency: Reflexion used two model calls per execution and averaged about 5.45 seconds, while ReAct used one call and averaged about 2.86 seconds. Reflexion occasionally corrected initially wrong answers, but most second-pass revisions did not change the final response. ReAct was therefore much faster and showed higher lexical trace consistency. Overall, the study found no universal winner, but a clear reliability–efficiency trade-off between additional reflection and computational cost.
