# Methodology Notes

This project compares ReAct and Reflexion agents using repeated execution on a HotpotQA subset.

The same dataset subset, base model, temperature, context, and answer extraction rules should be used for both agents. The independent variable is the agent architecture: ReAct or Reflexion.

Main metrics:

- Exact Match
- F1 Score
- pass@k
- Answer Consistency
- Reasoning Trace Similarity
- Latency

Recommended final setup:

- Pilot: 20 questions, 3 runs, 2 agents
- Main experiment: 150 questions, 5 runs, 2 agents
- Robustness test: 30 questions, 10 runs, 2 agents

Important limitation:

Reasoning trace similarity is approximate because traces can express similar reasoning with different wording. The project uses a simple similarity score initially. This can be upgraded to sentence-transformer embeddings for final analysis.
