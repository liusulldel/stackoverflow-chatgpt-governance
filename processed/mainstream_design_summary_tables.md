# Mainstream Tag Design Summary Tables

| model                               |    coef |     se |   pval |   nobs |
|:------------------------------------|--------:|-------:|-------:|-------:|
| accepted_rate_primary               | -0.0104 | 0.0051 | 0.0432 |    240 |
| mean_log_answers_primary            | -0.0228 | 0.011  | 0.0389 |    240 |
| log_questions_secondary             | -0.0483 | 0.0114 | 0      |    240 |
| mean_log_views_secondary            |  0.1872 | 0.0356 | 0      |    240 |
| accepted_question_hc1               | -0.0103 | 0.0049 | 0.0354 | 629022 |
| log_answers_question_hc1            | -0.0228 | 0.0047 | 0      | 629022 |
| accepted_rate_first_focal           | -0.0124 | 0.0042 | 0.003  |    240 |
| mean_log_answers_first_focal        | -0.024  | 0.0107 | 0.0254 |    240 |
| accepted_rate_fractional_all_tag    | -0.0113 | 0.004  | 0.0047 |    240 |
| mean_log_answers_fractional_all_tag | -0.0232 | 0.0099 | 0.0198 |    240 |

## Placebo Tests

| outcome          | placebo_month   |        coef |         se |      pval |
|:-----------------|:----------------|------------:|-----------:|----------:|
| accepted_rate    | 2022-07         |  0.00708132 | 0.00782322 | 0.365377  |
| mean_log_answers | 2022-07         | -0.0020124  | 0.0147867  | 0.891746  |
| log_questions    | 2022-07         |  0.0233927  | 0.0215383  | 0.277436  |
| accepted_rate    | 2022-09         | -0.0093359  | 0.0113619  | 0.411255  |
| mean_log_answers | 2022-09         | -0.0140143  | 0.0100207  | 0.161952  |
| log_questions    | 2022-09         |  0.0621608  | 0.0253622  | 0.0142492 |
| accepted_rate    | 2023-03         | -0.0056342  | 0.0116955  | 0.629991  |
| mean_log_answers | 2023-03         |  0.00539176 | 0.0142451  | 0.705059  |
| log_questions    | 2023-03         |  0.0146033  | 0.0360146  | 0.685123  |

## Wild Cluster Bootstrap

| specification                    |       coef |   cluster_pval |   wild_cluster_pval |   successful_draws |
|:---------------------------------|-----------:|---------------:|--------------------:|-------------------:|
| accepted_rate_primary            | -0.0103544 |     0.0432325  |           0.046046  |                999 |
| mean_log_answers_primary         | -0.0227685 |     0.038932   |           0.0640641 |                999 |
| accepted_rate_first_focal        | -0.0124022 |     0.00298179 |           0.024024  |                999 |
| accepted_rate_fractional_all_tag | -0.0112674 |     0.00466954 |           0.022022  |                999 |

## Leave-One-Out Robustness

| dropped_tag       | dropped_group   |        coef |         se |      pval |
|:------------------|:----------------|------------:|-----------:|----------:|
| android           | low             | -0.0145241  | 0.00592029 | 0.0141563 |
| apache-spark      | low             | -0.0111232  | 0.00542611 | 0.0403695 |
| bash              | high            | -0.0105952  | 0.00515188 | 0.0397274 |
| docker            | low             | -0.0108162  | 0.0057842  | 0.06149   |
| excel             | high            | -0.0114444  | 0.00496083 | 0.0210575 |
| firebase          | low             | -0.00906265 | 0.00493797 | 0.0664613 |
| javascript        | high            | -0.00612004 | 0.00480588 | 0.202859  |
| kubernetes        | low             | -0.00829373 | 0.00432968 | 0.0554222 |
| linux             | low             | -0.0102299  | 0.00539722 | 0.0580403 |
| memory-management | low             | -0.0103708  | 0.00516861 | 0.0448038 |
| multithreading    | low             | -0.0101504  | 0.0051924  | 0.0505997 |
| numpy             | high            | -0.0104349  | 0.0051398  | 0.0423356 |
| pandas            | high            | -0.0108602  | 0.00508413 | 0.0326705 |
| python            | high            | -0.0117716  | 0.00655425 | 0.0724914 |
| regex             | high            | -0.0100171  | 0.0052149  | 0.0547499 |
| sql               | high            | -0.0107594  | 0.00527933 | 0.0415484 |

## Partition Permutation Summary

| anchor_tag   |   n_unique_partitions |   observed_coef |   observed_pval |   coef_percentile_from_bottom |   pval_percentile_from_bottom |   share_negative_coefficients |   share_coefficients_at_least_as_negative_as_observed |
|:-------------|----------------------:|----------------:|----------------:|------------------------------:|------------------------------:|------------------------------:|------------------------------------------------------:|
| bash         |                  6435 |      -0.0103544 |       0.0432325 |                       10.5051 |                       17.7001 |                      0.466667 |                                              0.105051 |