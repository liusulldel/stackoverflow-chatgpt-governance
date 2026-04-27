# Who Still Answers: Finish Empirical Upgrades Readout

Date: 2026-04-16

## What changed

This build converts the last high-value empirical upgrades into three reviewer-facing tables.

1. `16-tag external calibration`: every focal tag is shown with its internal exposure rank and mapped JetBrains cluster.
2. `mechanism decomposition`: incumbent activity and role-location margins are separated rather than collapsed into a generic entrant story.
3. `certification conversion ladder`: early answer roles are linked directly to accepted-current conversion and positive-score endorsement.

## Safe interpretation

- The JetBrains layer calibrates the substitution premise at the cluster level; it does not independently verify each tag rank.
- The mechanism decomposition is descriptive panel evidence with the same bounded exposure-post interpretation as the main design.
- The certification ladder is a conversion readout, not a causal mediation design.

## Top exposure rows

| tag          |   exposure_rank | jetbrains_cluster   |   private_public_gap |
|:-------------|----------------:|:--------------------|---------------------:|
| regex        |               1 | SQL / analytics     |            0.0691489 |
| pandas       |               2 | Python / data       |            0.0707351 |
| numpy        |               3 | Python / data       |            0.0707351 |
| sql          |               4 | SQL / analytics     |            0.0691489 |
| apache-spark |               5 | Python / data       |            0.0707351 |

## Strongest decomposition rows

| family                      | margin                       | role           | outcome                  |   coefficient |         se |    p_value |   nobs |   mean_outcome | safe_read                                                                            |
|:----------------------------|:-----------------------------|:---------------|:-------------------------|--------------:|-----------:|-----------:|-------:|---------------:|:-------------------------------------------------------------------------------------|
| incumbent activity          | pre-period expert incumbents | all roles      | mean_answers             |    -0.197816  | 0.0697302  | 0.00455572 |   1152 |       1.43573  | Higher-exposure post-period change among pre-period incumbent contributor-tag pairs. |
| role-location decomposition | recent_entrant_90d_share     | top_score      | recent_entrant_90d_share |     0.0204965 | 0.00820408 | 0.0124778  |   1161 |       0.447515 | Recent entrants become more visible in this answer-pipeline role.                    |
| role-location decomposition | incumbent_365d_share         | first_positive | incumbent_365d_share     |    -0.0600054 | 0.0244101  | 0.0139626  |   1159 |       0.470308 | Pre-existing tag incumbents occupy this answer-pipeline role less often if negative. |
| role-location decomposition | incumbent_365d_share         | top_score      | incumbent_365d_share     |    -0.0500927 | 0.0209577  | 0.0168401  |   1161 |       0.4204   | Pre-existing tag incumbents occupy this answer-pipeline role less often if negative. |
| role-location decomposition | brand_new_platform_share     | first_positive | brand_new_platform_share |    -0.0160252 | 0.00682424 | 0.0188603  |   1159 |       0.113591 | Brand-new platform users occupy this role more often if positive.                    |
| role-location decomposition | recent_entrant_90d_share     | first_positive | recent_entrant_90d_share |     0.0214596 | 0.00922365 | 0.0199875  |   1159 |       0.389939 | Recent entrants become more visible in this answer-pipeline role.                    |
| role-location decomposition | brand_new_platform_share     | top_score      | brand_new_platform_share |    -0.014404  | 0.00619613 | 0.020089   |   1161 |       0.130557 | Brand-new platform users occupy this role more often if positive.                    |
| role-location decomposition | brand_new_platform_share     | first_answer   | brand_new_platform_share |    -0.0139336 | 0.00604787 | 0.0212288  |   1161 |       0.13457  | Brand-new platform users occupy this role more often if positive.                    |

## Strongest certification-ladder rows

| role                  | outcome                                         |   coefficient |         se |     p_value |   nobs |   mean_outcome | safe_read                                                                                                                        |
|:----------------------|:------------------------------------------------|--------------:|-----------:|------------:|-------:|---------------:|:---------------------------------------------------------------------------------------------------------------------------------|
| first positive answer | P(role answer accepted same calendar day)       |   -0.00854015 | 0.0025022  | 0.000642394 |   1159 |       0.290262 | Exposure-post coefficient from tag-month role panel with tag FE, month FE, tag-specific linear trends, and answer-count weights. |
| first answer          | P(role answer accepted same calendar day)       |   -0.0069391  | 0.00212866 | 0.00111472  |   1161 |       0.223058 | Exposure-post coefficient from tag-month role panel with tag FE, month FE, tag-specific linear trends, and answer-count weights. |
| top-scored answer     | P(role answer accepted same calendar day)       |   -0.00770043 | 0.00259244 | 0.00297466  |   1161 |       0.237744 | Exposure-post coefficient from tag-month role panel with tag FE, month FE, tag-specific linear trends, and answer-count weights. |
| top-scored answer     | P(role answer accepted within 7 calendar days)  |   -0.00752327 | 0.00295462 | 0.0108881   |   1161 |       0.405756 | Exposure-post coefficient from tag-month role panel with tag FE, month FE, tag-specific linear trends, and answer-count weights. |
| first answer          | P(role answer accepted within 7 calendar days)  |   -0.00755876 | 0.00307471 | 0.0139572   |   1161 |       0.38003  | Exposure-post coefficient from tag-month role panel with tag FE, month FE, tag-specific linear trends, and answer-count weights. |
| first answer          | P(role answer accepted within 30 calendar days) |   -0.00771128 | 0.00320074 | 0.0159868   |   1161 |       0.405186 | Exposure-post coefficient from tag-month role panel with tag FE, month FE, tag-specific linear trends, and answer-count weights. |
| first positive answer | P(role answer accepted within 7 calendar days)  |   -0.00851443 | 0.00373921 | 0.0227822   |   1159 |       0.487018 | Exposure-post coefficient from tag-month role panel with tag FE, month FE, tag-specific linear trends, and answer-count weights. |
| top-scored answer     | P(role answer accepted within 30 calendar days) |   -0.00740749 | 0.00330601 | 0.025051    |   1161 |       0.432854 | Exposure-post coefficient from tag-month role panel with tag FE, month FE, tag-specific linear trends, and answer-count weights. |
