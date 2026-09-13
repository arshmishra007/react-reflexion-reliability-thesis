# Experiment Run Summary

Total logged rows: 1500

Blank predictions: 0

## Summary Metrics

```text
agent_type  questions  mean_em  mean_f1  mean_pass_at_k  mean_answer_consistency  mean_trace_similarity  mean_latency_seconds  blank_predictions
     react        150 0.610667 0.793685        0.653333                 0.934667               0.685309              2.859632                  0
 reflexion        150 0.620000 0.802652        0.666667                 0.942667               0.630736              5.445755                  0
```

## Runtime Notes

- Mean question-level latency: 4.153 seconds
- Maximum question-level latency: 9.978 seconds

## Manual Review Notes

- Review blank predictions manually.
- Check whether Reflexion improves EM/F1 or mainly consistency.
- Compare latency between ReAct and Reflexion.
- Inspect low-consistency questions for common failure patterns.
