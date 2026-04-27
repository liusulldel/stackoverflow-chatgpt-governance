# Large-Design Summary Tables

## Sample Attrition

|   raw_rows |   analysis_rows |   raw_unique_questions |   analysis_unique_questions |   dropped_multi_selected_rows |   dropped_non_focal_rows |
|-----------:|----------------:|-----------------------:|----------------------------:|------------------------------:|-------------------------:|
|     373665 |          353645 |                 373665 |                      353645 |                         20020 |                    20020 |

## Exposure Ranking

| tag               |   exposure_index | exposure_group   |   pre_questions |   routine_keyword_share |
|:------------------|-----------------:|:-----------------|----------------:|------------------------:|
| linux             |        2.64617   | High Exposure    |           18672 |                0.367341 |
| csv               |        1.4229    | High Exposure    |           11156 |                0.875762 |
| multithreading    |        1.22766   | High Exposure    |           10147 |                0.297526 |
| bash              |        1.08543   | High Exposure    |           13603 |                0.433654 |
| memory-management |        0.519082  | High Exposure    |            1519 |                0.367347 |
| excel-formula     |        0.374377  | High Exposure    |            4722 |                0.626429 |
| numpy             |        0.176744  | Mid Exposure     |           22279 |                0.724763 |
| vba               |        0.0132812 | Mid Exposure     |           23845 |                0.470707 |
| spring            |       -0.0892394 | Mid Exposure     |           22554 |                0.383701 |
| powershell        |       -0.187364  | Mid Exposure     |           19558 |                0.389968 |
| regex             |       -0.231962  | Mid Exposure     |           19178 |                0.79075  |
| apache-spark      |       -0.237201  | Mid Exposure     |           13440 |                0.539062 |
| hibernate         |       -0.768908  | Low Exposure     |            5303 |                0.430134 |
| kubernetes        |       -0.801878  | Low Exposure     |           15694 |                0.298139 |
| android-studio    |       -0.83634   | Low Exposure     |           16873 |                0.387601 |
| entity-framework  |       -0.854882  | Low Exposure     |            5463 |                0.432546 |
| docker            |       -0.987112  | Low Exposure     |           30781 |                0.298366 |
| matplotlib        |       -1.2005    | Low Exposure     |           13504 |                0.555317 |
| firebase          |       -1.27026   | Low Exposure     |           31947 |                0.452124 |

## Descriptive Summary

| exposure_group   |   post_chatgpt |   questions |   accepted_rate |   mean_log_views |   mean_log_answers | period       |
|:-----------------|---------------:|------------:|----------------:|-----------------:|-------------------:|:-------------|
| Low Exposure     |              0 |      124999 |        0.350393 |          5.29369 |           0.558594 | Pre-ChatGPT  |
| Low Exposure     |              1 |       15969 |        0.204649 |          3.56058 |           0.381867 | Post-ChatGPT |
| Mid Exposure     |              0 |      126475 |        0.44098  |          5.00136 |           0.643789 | Pre-ChatGPT  |
| Mid Exposure     |              1 |       15566 |        0.309826 |          3.66007 |           0.516312 | Post-ChatGPT |
| High Exposure    |              0 |       62746 |        0.401109 |          4.83304 |           0.61961  | Pre-ChatGPT  |
| High Exposure    |              1 |        7890 |        0.291535 |          3.65006 |           0.508502 | Post-ChatGPT |

## Policy-Window Summary

|   post_policy |   questions |   accepted_rate |   mean_log_views | period   |
|--------------:|------------:|----------------:|-----------------:|:---------|
|             0 |        1606 |        0.311333 |          4.05294 | Dec 1-4  |
|             1 |        1560 |        0.282051 |          4.11248 | Dec 5-7  |

## Baseline Models

| model                     |    coef |     se |   pval |   nobs |
|:--------------------------|--------:|-------:|-------:|-------:|
| baseline_log_questions    | -0.006  | 0.0265 | 0.8193 |    513 |
| baseline_mean_log_views   |  0.1247 | 0.0901 | 0.1663 |    513 |
| baseline_accepted_rate    |  0.0085 | 0.0071 | 0.228  |    513 |
| baseline_mean_log_answers |  0.0115 | 0.0142 | 0.4166 |    513 |
| question_log_views        |  0.1247 | 0.0861 | 0.1472 | 353645 |
| question_accepted         |  0.0085 | 0.0068 | 0.2069 | 353645 |

## Segmented Models

| model                    | term           |    coef |     se |   pval |   nobs |
|:-------------------------|:---------------|--------:|-------:|-------:|-------:|
| segmented_log_questions  | level_break    |  0.0027 | 0.0304 | 0.9282 |    513 |
| segmented_log_questions  | post_slope     | -0.0036 | 0.0159 | 0.8187 |    513 |
| segmented_log_questions  | pretrend_slope |  0      | 0.0008 | 0.9763 |    513 |
| segmented_mean_log_views | level_break    |  0.0293 | 0.0156 | 0.0606 |    513 |
| segmented_mean_log_views | post_slope     |  0.0248 | 0.0304 | 0.4148 |    513 |
| segmented_mean_log_views | pretrend_slope |  0.0033 | 0.0027 | 0.2229 |    513 |

## Event Study Joint Tests

| outcome        |   pre_joint_pval |   post_joint_pval |
|:---------------|-----------------:|------------------:|
| mean_log_views |              nan |               nan |
| log_questions  |              nan |               nan |

## Placebo Break Tests

| outcome        | placebo_month   |   level_coef |   level_pval |   slope_coef |   slope_pval |
|:---------------|:----------------|-------------:|-------------:|-------------:|-------------:|
| log_questions  | 2022-07         |  -0.00460781 |     0.723987 | -7.11057e-06 |     0.998786 |
| mean_log_views | 2022-07         |  -0.0141662  |     0.409204 |  0.0153485   |     0.180675 |
| log_questions  | 2022-09         |   0.0034206  |     0.835963 | -0.000611293 |     0.903852 |
| mean_log_views | 2022-09         |  -0.0245498  |     0.358597 |  0.0222497   |     0.202803 |
| log_questions  | 2023-03         |  -0.00495581 |     0.823153 | -0.00495581  |     0.823153 |
| mean_log_views | 2023-03         |   0.0478378  |     0.265695 |  0.0478378   |     0.265695 |

## Permutation Tests

| specification                        |   observed_coef |   abs_permutation_pval |   one_sided_pval |   permutation_reps |
|:-------------------------------------|----------------:|-----------------------:|-----------------:|-------------------:|
| baseline_log_questions               |     -0.00604276 |                 0.853  |           0.4352 |               5000 |
| segmented_mean_log_views_level_break |      0.0293017  |                 0.1562 |           0.9306 |               5000 |

## Wild-Cluster Bootstrap

| specification                        | term                        |        coef |   cluster_se |   cluster_pval |   wild_cluster_pval |   successful_draws |   reps_requested |
|:-------------------------------------|:----------------------------|------------:|-------------:|---------------:|--------------------:|-------------------:|-----------------:|
| baseline_log_questions               | exposure_post               | -0.00604276 |    0.0264529 |      0.819308  |            0.815816 |                999 |              999 |
| segmented_mean_log_views_level_break | exposure_index:post_chatgpt |  0.0293017  |    0.015614  |      0.0605693 |            0.184184 |                999 |              999 |
| segmented_mean_log_views_post_slope  | exposure_index:time_post    |  0.0247816  |    0.0303862 |      0.414756  |            0.526527 |                999 |              999 |