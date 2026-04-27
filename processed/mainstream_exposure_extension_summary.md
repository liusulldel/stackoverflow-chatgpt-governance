# Mainstream Exposure Extension Summary

| model                                  |   coef |     se |   pval |   nobs |
|:---------------------------------------|-------:|-------:|-------:|-------:|
| accepted_rate_continuous_primary       | 0.0054 | 0.0024 | 0.0237 |    240 |
| mean_log_answers_continuous_primary    | 0.0148 | 0.0044 | 0.0008 |    240 |
| accepted_rate_continuous_fractional    | 0.0044 | 0.0019 | 0.0192 |    240 |
| mean_log_answers_continuous_fractional | 0.0119 | 0.004  | 0.0029 |    240 |

## Wild Cluster Bootstrap

| specification                       |       coef |   cluster_pval |   wild_cluster_pval |   successful_draws |
|:------------------------------------|-----------:|---------------:|--------------------:|-------------------:|
| accepted_rate_continuous_primary    | 0.00537751 |      0.0237455 |            0.01001  |                999 |
| accepted_rate_continuous_fractional | 0.0043893  |      0.0192499 |            0.013013 |                999 |

## Exposure Tercile Summary

| exposure_tercile   |   post_chatgpt |   weighted_accepted_rate |   weighted_mean_log_answers | period   |
|:-------------------|---------------:|-------------------------:|----------------------------:|:---------|
| Low                |              0 |                 0.355238 |                    0.586052 | pre      |
| Low                |              1 |                 0.288931 |                    0.499377 | post     |
| Middle             |              0 |                 0.284262 |                    0.484797 | pre      |
| Middle             |              1 |                 0.22969  |                    0.419033 | post     |
| High               |              0 |                 0.319353 |                    0.520057 | pre      |
| High               |              1 |                 0.253948 |                    0.437863 | post     |