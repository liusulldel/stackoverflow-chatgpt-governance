# Summary Tables

## Descriptive Summary

| group   |   post_chatgpt |   questions |   accepted_rate |   answered_share |   mean_log_answers |   mean_log_views | period       |
|:--------|---------------:|------------:|----------------:|-----------------:|-------------------:|-----------------:|:-------------|
| high    |              0 |        6956 |        0.466869 |         0.638219 |           0.709438 |          5.77067 | Pre-ChatGPT  |
| high    |              1 |        5229 |        0.46314  |         0.651933 |           0.713114 |          5.42731 | Post-ChatGPT |
| low     |              0 |        3336 |        0.373965 |         0.572371 |           0.626501 |          6.21226 | Pre-ChatGPT  |
| low     |              1 |        3415 |        0.331197 |         0.512892 |           0.554185 |          5.79699 | Post-ChatGPT |

## Policy-Window Summary

| group   |   post_policy |   questions |   accepted_rate |   answered_share |   mean_log_answers | period   |
|:--------|--------------:|------------:|----------------:|-----------------:|-------------------:|:---------|
| high    |             0 |         330 |        0.505486 |         0.661373 |           0.712253 | Dec 1-4  |
| high    |             1 |         358 |        0.443352 |         0.626433 |           0.705834 | Dec 5-7  |
| low     |             0 |         216 |        0.342067 |         0.490249 |           0.532843 | Dec 1-4  |
| low     |             1 |         227 |        0.323589 |         0.484556 |           0.60241  | Dec 5-7  |

## Baseline Models

| model                           |    coef |     se |   pval |   nobs |
|:--------------------------------|--------:|-------:|-------:|-------:|
| panel_baseline_mean_log_views   |  0.0107 | 0.0595 | 0.8574 |    192 |
| panel_baseline_log_questions    | -0.3193 | 0.118  | 0.0068 |    192 |
| panel_baseline_accepted_rate    |  0.0189 | 0.0122 | 0.1215 |    192 |
| panel_baseline_answered_share   |  0.0508 | 0.0222 | 0.0223 |    192 |
| panel_baseline_mean_log_answers |  0.0467 | 0.023  | 0.0424 |    192 |
| panel_baseline_novice_share     | -0.0292 | 0.0214 | 0.1725 |    192 |
| question_baseline_log_views     |  0.005  | 0.0383 | 0.8958 |  18936 |
| question_baseline_answered      |  0.0469 | 0.0147 | 0.0014 |  18936 |
| question_baseline_log_answers   |  0.0451 | 0.0123 | 0.0003 |  18936 |

## Preferred Trend-Break Models

| model                          | term           |    coef |     se |   pval |   nobs |
|:-------------------------------|:---------------|--------:|-------:|-------:|-------:|
| panel_segmented_mean_log_views | level_break    | -0.1915 | 0.0966 | 0.0474 |    192 |
| panel_segmented_mean_log_views | post_slope     |  0.0297 | 0.0079 | 0.0002 |    192 |
| panel_segmented_mean_log_views | pretrend_slope |  0.0014 | 0.0083 | 0.8625 |    192 |
| panel_segmented_log_questions  | level_break    | -0.0937 | 0.1702 | 0.582  |    192 |
| panel_segmented_log_questions  | post_slope     | -0.0495 | 0.0098 | 0      |    192 |
| panel_segmented_log_questions  | pretrend_slope |  0.0073 | 0.0163 | 0.6552 |    192 |
| question_segmented_log_views   | level_break    | -0.1927 | 0.0743 | 0.0095 |  18936 |
| question_segmented_log_views   | post_slope     |  0.0268 | 0.0111 | 0.0158 |  18936 |
| question_segmented_log_views   | pretrend_slope |  0.0027 | 0.0085 | 0.7536 |  18936 |

## Event-Study Joint Tests

| outcome        |   pre_joint_pval |   post_joint_pval |
|:---------------|-----------------:|------------------:|
| mean_log_views |           0.1607 |            0      |
| log_questions  |           0.0216 |            0.0002 |

## Placebo Break Tests

| outcome        | shock_month   |   level_coef |   level_se |   level_pval |   slope_coef |   slope_se |   slope_pval |
|:---------------|:--------------|-------------:|-----------:|-------------:|-------------:|-----------:|-------------:|
| mean_log_views | 2022-07       |    0.023215  |  0.0897336 |    0.795859  |    0.0377015 | 0.0199537  |  0.0588322   |
| log_questions  | 2022-07       |    0.10848   |  0.10372   |    0.295612  |   -0.0528365 | 0.0369206  |  0.152407    |
| mean_log_views | 2022-09       |   -0.143065  |  0.0642582 |    0.0259865 |    0.0169122 | 0.012619   |  0.180175    |
| log_questions  | 2022-09       |    0.0498996 |  0.135326  |    0.712325  |   -0.0541072 | 0.0213194  |  0.0111508   |
| mean_log_views | 2022-12       |   -0.19146   |  0.0965505 |    0.0473669 |    0.0297339 | 0.00787805 |  0.000160475 |
| log_questions  | 2022-12       |   -0.0936557 |  0.170155  |    0.582036  |   -0.0494709 | 0.0098401  |  4.96965e-07 |

## Leave-One-Tag-Out Checks

| omitted_tag       |   log_questions_coef |   log_questions_pval |   view_level_coef |   view_level_pval |
|:------------------|---------------------:|---------------------:|------------------:|------------------:|
| bash              |            -0.261566 |           0.018659   |         -0.193436 |         0.0671732 |
| docker            |            -0.245933 |           0.0676216  |         -0.214231 |         0.0995159 |
| excel-formula     |            -0.350628 |           0.00285134 |         -0.21155  |         0.0278503 |
| kubernetes        |            -0.289129 |           0.0860031  |         -0.097352 |         0.0631733 |
| memory-management |            -0.319454 |           0.0102649  |         -0.201604 |         0.0466024 |
| multithreading    |            -0.413541 |           4.6013e-08 |         -0.242795 |         0.0191316 |
| numpy             |            -0.311431 |           0.021755   |         -0.201543 |         0.0553672 |
| powershell        |            -0.343774 |           0.00833155 |         -0.148212 |         0.125286  |

## Randomization Benchmarks

| specification                              |   observed_coef |   abs_randomization_pval |   one_sided_randomization_pval |   num_assignments |
|:-------------------------------------------|----------------:|-------------------------:|-------------------------------:|------------------:|
| panel_baseline_log_questions               |       -0.319262 |                0.0714286 |                      0.0428571 |                70 |
| panel_segmented_mean_log_views_level_break |       -0.19146  |                0.114286  |                      0.0571429 |                70 |

## Wild-Cluster Bootstrap

| specification                              | term                  |       coef |   cluster_se |   cluster_pval |   wild_cluster_pval |   observed_t |   successful_draws |   reps_requested |
|:-------------------------------------------|:----------------------|-----------:|-------------:|---------------:|--------------------:|-------------:|-------------------:|-----------------:|
| panel_segmented_mean_log_views_level_break | high_tag:post_chatgpt | -0.19146   |   0.0965505  |    0.0473669   |           0.0710711 |     -1.98301 |                999 |              999 |
| panel_segmented_mean_log_views_post_slope  | high_tag:time_post    |  0.0297339 |   0.00787805 |    0.000160475 |           0.026026  |      3.77427 |                999 |              999 |
| panel_baseline_log_questions               | high_post             | -0.319262  |   0.117953   |    0.0067959   |           0.0660661 |     -2.70668 |                999 |              999 |
| panel_segmented_log_questions_post_slope   | high_tag:time_post    | -0.0494709 |   0.0098401  |    4.96965e-07 |           0.016016  |     -5.02748 |                999 |              999 |
| policy_accepted_rate                       | high_post_policy      | -0.0663128 |   0.0484498  |    0.171096    |           0.278278  |     -1.36869 |                999 |              999 |