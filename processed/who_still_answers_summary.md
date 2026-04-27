# Who Still Answers After ChatGPT? Current Analysis Summary

## Source State

- Question source: `stackexchange_20251231_dump`
- Answer source: `stackexchange_20251231_dump`
- Accepted timing source: `accept_vote_date_daylevel_if_available`
- Current data end: `2025-12-31T23:14:43.730000+00:00`
- 2025 dump watcher status: `completed`
- 2025 dump watcher detail: `Dump parsed and closure panel built.`

## What Is Implemented Now

- `tag_exposure_panel`
- `user_tag_preshock_cohorts`
- `user_tag_month_panel`
- `tag_month_entry_panel`
- `question_closure_panel`
- `identification_profile`
- `trend_break_results`
- `small_sample_inference`
- `leave_two_out`
- `construct_ladders`

## Read on Current Evidence

The contributor-reallocation pipeline is fully implemented on the currently available archival backbone. The canonical `2025Q4` dump backbone is active in this run, so the estimates now reflect the long-window archival population through `2025-12-31`. The headline entrant-side result remains positive and concentrated in `brand_new_platform` entrants, but the timing diagnostics no longer support a ChatGPT-timed break and the downstream accepted-answer outcomes move in the opposite direction of the original weakening hypothesis. The current build hardens the paper with outcome-specific trend-break diagnostics, conservative tag-level inference, leave-two-out stress tests, and construct-validation ladders.

## Model Results

| model                      |        coef |         se |      pval |   nobs |        r2 |
|:---------------------------|------------:|-----------:|----------:|-------:|----------:|
| incumbent_mean_log_answers | -0.334282   | 0.196443   | 0.0888163 |   2304 | 0.542216  |
| incumbent_share_active     | -0.178186   | 0.110743   | 0.107616  |   2304 | 0.640592  |
| expert_answer_share        | -0.0059868  | 0.0162895  | 0.713228  |   1146 | 0.947856  |
| novice_entry_share         |  0.00575991 | 0.00506368 | 0.255331  |   1145 | 0.939771  |
| first_answer_1d            |  0.00312471 | 0.00881093 | 0.722859  |   1147 | 0.944977  |
| accepted_7d                |  0.00896716 | 0.00405151 | 0.0268779 |   1147 | 0.623692  |
| accepted_30d               |  0.0113643  | 0.00452787 | 0.0120783 |   1140 | 0.602554  |
| postshock_presence         |  0.0484549  | 0.0206104  | 0.0187232 |  52363 | 0.0710437 |

## Identification Profile

| specification       | preferred_variant   | preferred_term       |   expected_sign | actual_break_month   |   actual_coef |   actual_se |   actual_pval |   actual_directional_coef |   actual_rank_vs_pre_breaks |   n_candidate_breaks |   n_pre_breaks |   actual_rank_percentile_vs_pre |   significant_pre_breaks |   share_significant_pre_breaks | first_significant_pre_break   |   pre_break_positive_share |
|:--------------------|:--------------------|:---------------------|----------------:|:---------------------|--------------:|------------:|--------------:|--------------------------:|----------------------------:|---------------------:|---------------:|--------------------------------:|-------------------------:|-------------------------------:|:------------------------------|---------------------------:|
| novice_entry_share  | slope_only          | exposure_break_slope |               1 | 2022-12              |  -0.00118342  | 0.000490337 |     0.0158011 |              -0.00118342  |                           2 |                   30 |             29 |                        0.965517 |                       14 |                       0.482759 | 2021-10                       |                   0        |
| expert_answer_share | level_only          | exposure_break_post  |              -1 | 2022-12              |  -0.0210731   | 0.0110312   |     0.0560918 |               0.0210731   |                           1 |                   30 |             29 |                        1        |                        6 |                       0.206897 | 2020-07                       |                   0.448276 |
| accepted_7d_rate    | level_only          | exposure_break_post  |              -1 | 2022-12              |   0.000308224 | 0.0021588   |     0.886467  |              -0.000308224 |                          16 |                   30 |             29 |                        0.482759 |                        0 |                       0        | nan                           |                   0.448276 |

## Small-Sample Inference

| specification              | term                 |        coef |     cr2_se |   cr2_pval |   cr2_tstat |   cr2_df |   wild_cluster_bootstrap_pval |   randomization_pval |
|:---------------------------|:---------------------|------------:|-----------:|-----------:|------------:|---------:|------------------------------:|---------------------:|
| incumbent_mean_log_answers | exposure_post_expert | -0.334282   | 0.203526   |  0.121289  |   -1.64246  |       15 |                    0.24812    |           0.104208   |
| incumbent_share_active     | exposure_post_expert | -0.178186   | 0.115107   |  0.142458  |   -1.548    |       15 |                    0.24812    |           0.122244   |
| expert_answer_share        | exposure_post        | -0.0059868  | 0.0178241  |  0.741612  |   -0.335882 |       15 |                    0.706767   |           0.613226   |
| novice_entry_share         | exposure_post        |  0.00575991 | 0.00557385 |  0.3178    |    1.03338  |       15 |                    0.403509   |           0.54509    |
| first_answer_1d            | exposure_post        |  0.00312471 | 0.00993512 |  0.757464  |    0.314512 |       15 |                    0.734336   |           0.773547   |
| accepted_7d                | exposure_post        |  0.00896716 | 0.00476711 |  0.0795217 |    1.88105  |       15 |                    0.00250627 |           0.00801603 |
| accepted_30d               | exposure_post        |  0.0113643  | 0.00528019 |  0.0480701 |    2.15224  |       15 |                    0          |           0.01002    |

## Tag Exposure Panel

| primary_tag       |   exposure_index |   exposure_rank | exposure_tercile   |   n_questions |   mean_first_answer_1d |   mean_accepted_7d |
|:------------------|-----------------:|----------------:|:-------------------|--------------:|-----------------------:|-------------------:|
| regex             |         1.65731  |               1 | High               |         25971 |               0.863502 |           0.207655 |
| pandas            |         1.1494   |               2 | High               |         20189 |               0.792412 |           0.172024 |
| numpy             |         0.862646 |               3 | High               |          4252 |               0.655691 |           0.191204 |
| sql               |         0.812737 |               4 | High               |        150708 |               0.835928 |           0.178352 |
| apache-spark      |         0.678581 |               5 | High               |         21881 |               0.584114 |           0.185321 |
| excel             |         0.512199 |               6 | Middle             |         72417 |               0.701037 |           0.174545 |
| javascript        |         0.484763 |               7 | Middle             |        618363 |               0.701832 |           0.155    |
| bash              |         0.426737 |               8 | Middle             |         27492 |               0.77528  |           0.193511 |
| python            |         0.370429 |               9 | Middle             |        714087 |               0.668696 |           0.161013 |
| linux             |        -0.348612 |              10 | Middle             |         33433 |               0.53516  |           0.151676 |
| memory-management |        -0.52261  |              11 | Low                |          3082 |               0.644062 |           0.162557 |
| android           |        -0.582472 |              12 | Low                |        196116 |               0.53888  |           0.145939 |
| firebase          |        -0.826932 |              13 | Low                |         40880 |               0.618004 |           0.177299 |
| multithreading    |        -1.01989  |              14 | Low                |         16670 |               0.644511 |           0.184163 |
| kubernetes        |        -1.58677  |              15 | Low                |         30546 |               0.568945 |           0.192628 |
| docker            |        -2.06752  |              16 | Low                |         59798 |               0.501472 |           0.15656  |

## Preshock Cohort Counts

| primary_tag       |   n_preshock_contributors |   n_experts |   n_incumbents |
|:------------------|--------------------------:|------------:|---------------:|
| android           |                     53014 |         915 |           5159 |
| apache-spark      |                      4658 |         109 |            448 |
| bash              |                      7092 |         152 |            513 |
| docker            |                     22732 |         140 |            825 |
| excel             |                     11101 |         276 |            968 |
| firebase          |                     14207 |          96 |            632 |
| javascript        |                    154674 |        3802 |          19604 |
| kubernetes        |                      8863 |         149 |            596 |
| linux             |                     12228 |         122 |            551 |
| memory-management |                      1401 |           7 |             73 |
| multithreading    |                      5722 |          59 |            382 |
| numpy             |                      1784 |          11 |             84 |
| pandas            |                      5321 |         102 |            419 |
| python            |                    151303 |        4168 |          18976 |
| regex             |                      8415 |         137 |            629 |
| sql               |                     29262 |         651 |           2504 |

## Timing / Placebo Snapshot

| specification       | placebo_month   |       coef |         se |       pval |
|:--------------------|:----------------|-----------:|-----------:|-----------:|
| expert_answer_share | 2020-04         | 0.0298374  | 0.0160015  | 0.0622293  |
| novice_entry_share  | 2020-04         | 0.0256162  | 0.0135134  | 0.05801    |
| accepted_7d_rate    | 2020-04         | 0.00579439 | 0.00250066 | 0.0204961  |
| expert_answer_share | 2020-05         | 0.0247601  | 0.0162582  | 0.127775   |
| novice_entry_share  | 2020-05         | 0.0234745  | 0.011886   | 0.0482705  |
| accepted_7d_rate    | 2020-05         | 0.00684912 | 0.00242646 | 0.00476243 |
| expert_answer_share | 2020-06         | 0.0234754  | 0.0164934  | 0.154644   |
| novice_entry_share  | 2020-06         | 0.0219781  | 0.0106729  | 0.03947    |
| accepted_7d_rate    | 2020-06         | 0.00564664 | 0.00232248 | 0.015045   |
| expert_answer_share | 2020-07         | 0.0202551  | 0.016995   | 0.233328   |
| novice_entry_share  | 2020-07         | 0.0191009  | 0.00988807 | 0.0533947  |
| accepted_7d_rate    | 2020-07         | 0.0051701  | 0.00250336 | 0.0388984  |
| expert_answer_share | 2020-08         | 0.0210605  | 0.0170185  | 0.215899   |
| novice_entry_share  | 2020-08         | 0.0172264  | 0.00939698 | 0.0667758  |
| accepted_7d_rate    | 2020-08         | 0.00500559 | 0.00219388 | 0.0225123  |
| expert_answer_share | 2020-09         | 0.0214489  | 0.0168747  | 0.203704   |
| novice_entry_share  | 2020-09         | 0.0155391  | 0.00932881 | 0.0957707  |
| accepted_7d_rate    | 2020-09         | 0.00546197 | 0.00224313 | 0.0148926  |

- Significant pre-shock placebo rows: `34`

## Leave-One-Out Snapshot

| specification       | dropped_tag   |        coef |         se |        pval |
|:--------------------|:--------------|------------:|-----------:|------------:|
| expert_answer_share | android       | -0.0100845  | 0.0185434  | 0.586556    |
| novice_entry_share  | android       |  0.00833512 | 0.00425981 | 0.0503842   |
| accepted_7d_rate    | android       |  0.00778554 | 0.00429884 | 0.0701285   |
| expert_answer_share | apache-spark  | -0.00486717 | 0.0165101  | 0.768146    |
| novice_entry_share  | apache-spark  |  0.00597724 | 0.00510622 | 0.241767    |
| accepted_7d_rate    | apache-spark  |  0.00930635 | 0.00412814 | 0.024173    |
| expert_answer_share | bash          | -0.00731338 | 0.0162447  | 0.652565    |
| novice_entry_share  | bash          |  0.00593664 | 0.00508549 | 0.243062    |
| accepted_7d_rate    | bash          |  0.00885901 | 0.00405995 | 0.029106    |
| expert_answer_share | docker        |  0.0100234  | 0.0190263  | 0.59832     |
| novice_entry_share  | docker        |  0.00356333 | 0.00897707 | 0.691414    |
| accepted_7d_rate    | docker        |  0.0145339  | 0.00287831 | 4.43056e-07 |
| expert_answer_share | excel         | -0.00726695 | 0.0162955  | 0.655635    |
| novice_entry_share  | excel         |  0.0054329  | 0.00522089 | 0.298057    |
| accepted_7d_rate    | excel         |  0.00871224 | 0.00405438 | 0.0316468   |
| expert_answer_share | firebase      | -0.00396887 | 0.0175708  | 0.821296    |
| novice_entry_share  | firebase      |  0.0050698  | 0.00519791 | 0.329384    |
| accepted_7d_rate    | firebase      |  0.00795627 | 0.00386539 | 0.0395581   |
| expert_answer_share | javascript    | -0.00336778 | 0.0173128  | 0.845764    |
| novice_entry_share  | javascript    |  0.00109202 | 0.00390159 | 0.779561    |
| accepted_7d_rate    | javascript    |  0.00898495 | 0.00422196 | 0.0333251   |
| expert_answer_share | kubernetes    | -0.0169133  | 0.0120447  | 0.160254    |
| novice_entry_share  | kubernetes    |  0.00484795 | 0.0055312  | 0.380772    |
| accepted_7d_rate    | kubernetes    |  0.00730739 | 0.00369369 | 0.0478895   |

## Leave-Two-Out Summary

| specification       |   positive_share |   significant_share |
|:--------------------|-----------------:|--------------------:|
| accepted_30d        |         1        |          0.983333   |
| accepted_7d         |         1        |          0.783333   |
| expert_answer_share |         0.158333 |          0.00833333 |
| first_answer_1d     |         0.825    |          0          |
| novice_entry_share  |         0.991667 |          0.0666667  |

## Construct Validation Snapshot

| family                       | variant                       | metric            | label                |           value |    extra_1 |       extra_2 |     extra_3 |
|:-----------------------------|:------------------------------|:------------------|:---------------------|----------------:|-----------:|--------------:|------------:|
| expert_holdout               | early_window_top_decile_min20 | group_mean        | other_holdout_users  |     0.314826    | 0.110563   |      0.173362 |    0.134934 |
| expert_holdout               | early_window_top_decile_min20 | group_mean        | expert_holdout_label |    18.1446      | 8.45211    |      2.91925  |    0.738382 |
| expert_face_validity         | current_frozen_definition     | group_mean        | incumbent_nonexpert  |     8.60788     | 0.341182   |      1.0868   |  997.804    |
| expert_face_validity         | current_frozen_definition     | group_mean        | expert               |    82.1444      | 0.406413   |      1.18843  | 1176.22     |
| expert_threshold_sensitivity | pct_5_min_10                  | expert_share_coef | exposure_post        |     0.000963765 | 0.958751   |  19473        |  nan        |
| expert_threshold_sensitivity | pct_5_min_20                  | expert_share_coef | exposure_post        |    -0.00582385  | 0.720497   |  10856        |  nan        |
| expert_threshold_sensitivity | pct_5_min_30                  | expert_share_coef | exposure_post        |    -0.00852225  | 0.554821   |   6788        |  nan        |
| expert_threshold_sensitivity | pct_9_min_10                  | expert_share_coef | exposure_post        |    -0.00544009  | 0.782865   |  23625        |  nan        |
| expert_threshold_sensitivity | pct_9_min_20                  | expert_share_coef | exposure_post        |    -0.0059868   | 0.713228   |  10896        |  nan        |
| expert_threshold_sensitivity | pct_9_min_30                  | expert_share_coef | exposure_post        |    -0.0085296   | 0.554508   |   6790        |  nan        |
| expert_threshold_sensitivity | pct_15_min_10                 | expert_share_coef | exposure_post        |    -0.00567594  | 0.775981   |  23935        |  nan        |
| expert_threshold_sensitivity | pct_15_min_20                 | expert_share_coef | exposure_post        |    -0.00601264  | 0.712091   |  10903        |  nan        |
| expert_threshold_sensitivity | pct_15_min_30                 | expert_share_coef | exposure_post        |    -0.0085296   | 0.554508   |   6790        |  nan        |
| expert_component_variant     | composite                     | n_experts         | expert_count         | 10896           | 0.0221564  |    nan        |  nan        |
| expert_component_variant     | volume_only                   | n_experts         | expert_count         | 10903           | 0.0221706  |    nan        |  nan        |
| expert_component_variant     | accepted_only                 | n_experts         | expert_count         | 10855           | 0.022073   |    nan        |  nan        |
| expert_component_variant     | score_only                    | n_experts         | expert_count         |   941           | 0.00191347 |    nan        |  nan        |
| entrant_tenure_sensitivity   | tenure_30d                    | novice_entry_coef | exposure_post        |     0.0092115   | 0.0728355  | 232905        |  nan        |

## Guardrail

Do not claim that the dump-backed rerun validates `expert exit + novice entry + weaker closure`. The current long-window run supports a narrower entrant-side paper: entrant share rises more in more exposed domains, expert-share decline is not established, and acceptance-based visible-resolution outcomes improve rather than weaken. The entrant result still does not clear the full conservative-inference stack, and the timing diagnostics do not support a clean ChatGPT break.
