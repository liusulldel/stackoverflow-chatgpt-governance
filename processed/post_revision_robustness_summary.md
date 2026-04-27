# Post-Revision Robustness Checks

## Leave-Two-Tags-Out Jackknife

- Models estimated: `120`
- Negative coefficient share: `0.992`
- p < 0.05 share: `0.433`
- Coefficient range: `-0.0173` to `0.0006`
- Least negative case: drop `javascript` and `python` -> coef `0.0006`, p `0.9355`
- Most negative case: drop `android` and `docker` -> coef `-0.0173`, p `0.0095`

## Dense Placebo Grid

- Placebo months evaluated: `11`
- Pre-shock placebo months evaluated: `9`
- Significant pre-shock placebos (p < 0.05): `2022-04, 2022-10, 2022-11`
- Lowest pre-shock placebo p-value: `0.0055`

## Head Rows

| dropped_tag_a   | dropped_tag_b     |       coef |         se |       pval |   n_clusters |
|:----------------|:------------------|-----------:|-----------:|-----------:|-------------:|
| android         | apache-spark      | -0.0164937 | 0.00605471 | 0.00644735 |           14 |
| android         | bash              | -0.0147748 | 0.00595031 | 0.0130271  |           14 |
| android         | docker            | -0.0172854 | 0.00666934 | 0.00954842 |           14 |
| android         | excel             | -0.0156044 | 0.00580007 | 0.00713706 |           14 |
| android         | firebase          | -0.0128886 | 0.00659213 | 0.0505654  |           14 |
| android         | javascript        | -0.0103302 | 0.00564097 | 0.0670582  |           14 |
| android         | kubernetes        | -0.0111051 | 0.00503568 | 0.0274339  |           14 |
| android         | linux             | -0.0151026 | 0.00674989 | 0.0252566  |           14 |
| android         | memory-management | -0.0146098 | 0.00600309 | 0.0149447  |           14 |
| android         | multithreading    | -0.0144265 | 0.00621535 | 0.0202808  |           14 |

| placebo_month   |        coef |         se |       pval |
|:----------------|------------:|-----------:|-----------:|
| 2022-03         |  0.015478   | 0.00981228 | 0.114701   |
| 2022-04         |  0.0165778  | 0.00804417 | 0.0393178  |
| 2022-05         |  0.00917605 | 0.0050394  | 0.068628   |
| 2022-06         |  0.00316962 | 0.00587286 | 0.5894     |
| 2022-07         |  0.00708132 | 0.00782322 | 0.365377   |
| 2022-08         |  0.00184386 | 0.00695557 | 0.790939   |
| 2022-09         | -0.0093359  | 0.0113619  | 0.411255   |
| 2022-10         | -0.0174399  | 0.00849861 | 0.0401611  |
| 2022-11         | -0.0154362  | 0.00556055 | 0.00550285 |
| 2022-12         | -0.0103544  | 0.00512227 | 0.0432325  |
| 2023-01         | -0.00361529 | 0.00514122 | 0.481933   |
