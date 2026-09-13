# Reconstructed Statistical Analysis Validation

> Provenance: this is a reconstructed analysis stage based on the frozen final outputs; it is not represented as the original missing local script.

- Primary statistical unit: 150 paired questions
- Bootstrap: 10,000 paired resamples, seed 42
- Validation against dissertation Appendix B values: **PASS**

## Primary results

| metric             |   react_mean |   reflexion_mean |   difference_reflexion_minus_react |   bootstrap_ci_low |   bootstrap_ci_high | primary_test         |   test_statistic |   primary_p | effect_size_label          |   effect_size |   reflexion_favoured |   react_favoured |   ties |   reflexion_only_pass |   react_only_pass |      holm_p |
|:-------------------|-------------:|-----------------:|-----------------------------------:|-------------------:|--------------------:|:---------------------|-----------------:|------------:|:---------------------------|--------------:|---------------------:|-----------------:|-------:|----------------------:|------------------:|------------:|
| Exact Match        |     0.610667 |         0.62     |                         0.00933333 |        -0.012      |           0.0306667 | Wilcoxon signed-rank |             70.5 | 0.510569    | matched rank-biserial r_rb |      0.175439 |                   11 |                7 |    132 |                   nan |               nan | 1           |
| F1 score           |     0.793685 |         0.802652 |                         0.00896779 |        -0.00651655 |           0.0267366 | Wilcoxon signed-rank |            165.5 | 0.572257    | matched rank-biserial r_rb |      0.131054 |                   14 |               12 |    124 |                   nan |               nan | 1           |
| pass@5             |     0.653333 |         0.666667 |                         0.0133333  |        -0.02       |           0.0533333 | Exact McNemar        |            nan   | 0.726562    | discordant-pair OR         |      1.66667  |                    5 |                3 |    142 |                     5 |                 3 | 1           |
| Answer consistency |     0.934667 |         0.942667 |                         0.008      |        -0.0106667  |           0.0266667 | Wilcoxon signed-rank |            119   | 0.369444    | matched rank-biserial r_rb |      0.206667 |                   14 |               10 |    126 |                   nan |               nan | 1           |
| Trace similarity   |     0.685309 |         0.630736 |                        -0.0545727  |        -0.0673863  |          -0.0419792 | Wilcoxon signed-rank |           1835   | 6.90483e-13 | matched rank-biserial r_rb |     -0.675938 |                   35 |              115 |      0 |                   nan |               nan | 3.45242e-12 |
| Latency (seconds)  |     2.85963  |         5.44575  |                         2.58612    |         2.47546    |           2.70264   | Wilcoxon signed-rank |              0   | 2.29955e-26 | matched rank-biserial r_rb |      1        |                    0 |              150 |      0 |                   nan |               nan | 1.37973e-25 |

## Question-type summaries

| question_type   | agent_type   |   n_questions |       em |       f1 |   pass_at_5 |   consistency |   trace_similarity |   latency |
|:----------------|:-------------|--------------:|---------:|---------:|------------:|--------------:|-------------------:|----------:|
| bridge          | react        |           126 | 0.568254 | 0.7819   |    0.603175 |      0.933333 |           0.690275 |   2.85752 |
| bridge          | reflexion    |           126 | 0.573016 | 0.786226 |    0.626984 |      0.934921 |           0.631925 |   5.38947 |
| comparison      | react        |            24 | 0.833333 | 0.855556 |    0.916667 |      0.941667 |           0.659238 |   2.87072 |
| comparison      | reflexion    |            24 | 0.866667 | 0.888889 |    0.875    |      0.983333 |           0.624494 |   5.74124 |

## Unstable-question summary

| agent_type   |   unstable_questions |   minimum_consistency |
|:-------------|---------------------:|----------------------:|
| react        |                   30 |                   0.4 |
| reflexion    |                   25 |                   0.4 |

## Reflexion revision summary

| outcome                |   count |   percentage |
|:-----------------------|--------:|-------------:|
| Final answer changed   |      15 |          2   |
| Final answer unchanged |     735 |         98   |
| Wrong to Exact Match   |       9 |          1.2 |
| Exact Match to wrong   |       0 |          0   |
| F1 improved            |      12 |          1.6 |
| F1 decreased           |       3 |          0.4 |
| F1 unchanged           |     735 |         98   |

## Validation checks

| metric             | field      |       expected |       observed |   absolute_difference | matches_reported_value   |
|:-------------------|:-----------|---------------:|---------------:|----------------------:|:-------------------------|
| Exact Match        | difference |  0.0093333333  |  0.0093333333  |         1.7347235e-18 | True                     |
| Exact Match        | ci_low     | -0.012         | -0.012         |         0             | True                     |
| Exact Match        | ci_high    |  0.030667      |  0.030666667   |         3.3333333e-07 | True                     |
| Exact Match        | p          |  0.51056917    |  0.51056917    |         0             | True                     |
| F1 score           | difference |  0.0089677948  |  0.0089677948  |         0             | True                     |
| F1 score           | ci_low     | -0.006517      | -0.006516548   |         4.5197515e-07 | True                     |
| F1 score           | ci_high    |  0.026737      |  0.026736566   |         4.344219e-07  | True                     |
| F1 score           | p          |  0.57225736    |  0.57225736    |         0             | True                     |
| pass@5             | difference |  0.013333333   |  0.013333333   |         0             | True                     |
| pass@5             | ci_low     | -0.02          | -0.02          |         0             | True                     |
| pass@5             | ci_high    |  0.053333      |  0.053333333   |         3.3333333e-07 | True                     |
| pass@5             | p          |  0.7265625     |  0.7265625     |         0             | True                     |
| Answer consistency | difference |  0.008         |  0.008         |         1.7347235e-18 | True                     |
| Answer consistency | ci_low     | -0.010667      | -0.010666667   |         3.3333333e-07 | True                     |
| Answer consistency | ci_high    |  0.026667      |  0.026666667   |         3.3333333e-07 | True                     |
| Answer consistency | p          |  0.36944444    |  0.36944444    |         0             | True                     |
| Trace similarity   | difference | -0.054572706   | -0.054572706   |         0             | True                     |
| Trace similarity   | ci_low     | -0.067386      | -0.067386307   |         3.0669167e-07 | True                     |
| Trace similarity   | ci_high    | -0.041979      | -0.041979244   |         2.4406112e-07 | True                     |
| Trace similarity   | p          |  6.9048306e-13 |  6.9048306e-13 |         0             | True                     |
| Latency (seconds)  | difference |  2.5861227     |  2.5861227     |         0             | True                     |
| Latency (seconds)  | ci_low     |  2.47546       |  2.4754604     |         4e-07         | True                     |
| Latency (seconds)  | ci_high    |  2.702645      |  2.7026449     |         6.6666666e-08 | True                     |
| Latency (seconds)  | p          |  2.2995482e-26 |  2.2995482e-26 |         0             | True                     |
