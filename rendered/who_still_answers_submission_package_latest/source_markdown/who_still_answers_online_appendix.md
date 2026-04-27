# Online Appendix for Private AI, Public Answer Work, and Certification at Stack Overflow

Main manuscript:
- [who_still_answers_option_c_manuscript.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_option_c_manuscript.md)

## A. Canonical Source Regime

The canonical evidence base is the official Stack Overflow `2025Q4` dump covering `2020-01-01` through `2025-12-31`.

Reviewer-facing rule:

- canonical contributor and answer-pipeline evidence comes from the dump-backed long window
- earlier archive runs are design audit trails only
- API timing checks are bounded validation support only

Key control files:

- [who_still_answers_option_c_canonical_lock_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_option_c_canonical_lock_2026-04-04.md)
- [who_still_answers_ws6_completion_memo_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_ws6_completion_memo_2026-04-04.md)

## B. Sample and Linked Objects

The current backbone contains:

- `2,035,885` single-focal-tag questions
- `2,391,883` matched valid answers
- `16` focal technical domains

The main linked objects used by the rebuilt mechanism paper are:

1. durability entrant panel  
   [who_still_answers_durability_entrant_panel.parquet](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_durability_entrant_panel.parquet)
2. answer-role question panel  
   [who_still_answers_answer_role_question_panel.parquet](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_answer_role_question_panel.parquet)
3. answer-role tag-month panel  
   [who_still_answers_answer_role_tag_month_panel.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_answer_role_tag_month_panel.csv)
4. infrastructure bridge panel  
   [who_still_answers_infrastructure_bridge_panel.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_infrastructure_bridge_panel.csv)

### B1. Single-Focal Attrition

The broader focal-question universe contains `2,322,009` questions carrying at least one focal tag. The canonical single-focal backbone keeps `2,035,885` of them (`87.7%`) and excludes `286,124` multi-tag questions (`12.3%`).

Attrition summary:

| Group | Questions | Share | Mean score | Mean views | Mean answers | Accepted-archive rate | Mean tags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kept single-focal questions | `2,035,885` | `87.7%` | `0.693` | `1,161.9` | `1.175` | `0.426` | `3.24` |
| excluded multi-tag questions | `286,124` | `12.3%` | `0.681` | `917.7` | `1.254` | `0.514` | `3.63` |

Source table:
- [who_still_answers_single_focal_attrition_summary.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_single_focal_attrition_summary.csv)

Safe read:

- the single-focal restriction does not drop most of the focal-tag universe
- the excluded set is somewhat more multi-tag, more answer-rich, and more often accepted
- the paper therefore narrows to cleaner exposure assignment rather than claiming to cover the full Stack Overflow question universe

## C. Mechanism Families

### C1. Durability

Readout:
- [who_still_answers_durability_mechanism_readout_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_durability_mechanism_readout_2026-04-04.md)

Core outcomes:

- `return_30d`
- `return_90d`
- `return_180d`
- `return_365d`
- `one_shot_365d`
- `low_repeat_90d`

Headline bounded result:

- `return_365d = -0.0266`
- clustered `p = 0.0004`
- `CR2 p = 0.0047`
- wild `p = 0.0652`
- randomization `p = 0.1242`

Interpretation:

Durability is promoted, but it is not the strongest family after the full `WS6` gate.

### C2. Answer-Role Reallocation

Readout:
- [who_still_answers_answer_role_mechanism_readout_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_answer_role_mechanism_readout_2026-04-04.md)

Core promoted role outcomes on `recent_entrant_90d_share`:

- `first_answer = 0.0245`, `p = 0.0017`
- `first_positive = 0.0281`, `p = 0.0022`
- `top_score = 0.0272`, `p = 0.0003`
- `accepted_current = 0.0169`, `p = 0.0617`

Interpretation:

The safe claim is not that accepted-current certification is unaffected. The safe claim is that the entrant-heavy shift is stronger in early and endorsed answer roles than in accepted-current certification.

### C3. Infrastructure Bridge

Readout:
- [who_still_answers_infrastructure_bridge_readout_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_infrastructure_bridge_readout_2026-04-04.md)

Core bridge variable:

- `recent_gap_first_vs_accepted`

Core outcomes:

- `first_positive_answer_latency_mean = 283.9`, `p = 0.0039`
- `accepted_cond_any_answer_30d_rate = -0.0704`, `p = 0.0509`
- `accepted_vote_30d_rate = -0.0752`, `p = 0.0019`

Interpretation:

The bridge is an independent consequence layer, not a mediation claim.

### C4. Incumbent and Entrant Decomposition

Finish-stage decomposition files:

- [who_still_answers_mechanism_decomposition_table.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_mechanism_decomposition_table.csv)
- [who_still_answers_finish_empirical_upgrades_readout_2026-04-16.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_finish_empirical_upgrades_readout_2026-04-16.md)

The decomposition separates incumbent answer intensity from entrant and role-location margins. The safest read is:

- pre-shock incumbents remain observable, but in more exposed domains expert answer intensity falls after the transition
- entrant-flow evidence shifts toward newer public answerers, but this is not direct proof of AI-assisted entry
- role-location evidence remains about where contributors appear in the public answer pipeline, not why each contributor appeared

### C5. Certification-Conversion Ladder

Finish-stage certification files:

- [who_still_answers_certification_conversion_ladder.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_certification_conversion_ladder.csv)

Main accepted-certification rows:

| Role | `accepted_7d` coefficient | p-value | `accepted_30d` coefficient | p-value |
| --- | ---: | ---: | ---: | ---: |
| first answer | `-0.0076` | `0.0140` | `-0.0077` | `0.0160` |
| first positive answer | `-0.0085` | `0.0228` | `-0.0102` | `0.0331` |
| top-scored answer | `-0.0075` | `0.0109` | `-0.0074` | `0.0251` |

Safe read:

- early visible answers convert into accepted public certification less often in more exposed domains after the transition
- archive-level positive-score measures do not show the same collapse, so the result is about accepted certification rather than generic endorsement
- the ladder is a conversion readout, not a causal mediation design

### C4. Incumbent and Entrant Decomposition

The final build separates incumbent capacity from recent-entrant role location. The strongest decomposition rows are:

| Margin | Role | Coefficient | p-value | Read |
| --- | --- | ---: | ---: | --- |
| expert incumbent mean answers | all roles | `-0.1978` | `0.0046` | thinner expert-incumbent activity |
| incumbent share | first positive | `-0.0600` | `0.0140` | fewer pre-existing tag incumbents in endorsed early role |
| incumbent share | top score | `-0.0501` | `0.0168` | fewer pre-existing tag incumbents in top-scored role |
| recent entrant share | first positive | `+0.0215` | `0.0200` | more recent entrants in endorsed early role |
| recent entrant share | top score | `+0.0205` | `0.0125` | more recent entrants in top-scored role |

Source table:
- [who_still_answers_mechanism_decomposition_table.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_mechanism_decomposition_table.csv)

Safe read:

- this is a decomposition of observable public answer roles, not a direct behavioral model of why each contributor changed behavior
- the evidence supports thinner incumbent answer capacity and broader recent-entrant reallocation
- it does not support a simple `brand_new_platform` takeover story

### C5. Certification-Conversion Ladder

The role-level conversion ladder separates accepted-current conversion rates, positive-score endorsement, and acceptance timing among answers that become accepted.

| Role | Outcome | Coefficient | p-value | Read |
| --- | --- | ---: | ---: | --- |
| first answer | accepted-current conversion | `-0.0052` | `0.1561` | negative but not conventionally significant |
| first positive answer | accepted-current conversion | `-0.0070` | `0.1693` | negative but not conventionally significant |
| top-scored answer | accepted-current conversion | `-0.0047` | `0.1991` | negative but not conventionally significant |
| first answer | acceptance lag among accepted answers | `+2.60` days | `0.0016` | slower conversion into accepted resolution |
| first positive answer | acceptance lag among accepted answers | `+2.66` days | `0.0010` | slower conversion into accepted resolution |
| top-scored answer | acceptance lag among accepted answers | `+2.48` days | `0.0072` | slower conversion into accepted resolution |

Source table:
- [who_still_answers_certification_conversion_ladder.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_certification_conversion_ladder.csv)

Safe read:

- the ladder does not show a clean collapse in role-level acceptance probabilities
- it does show that accepted answers take longer to become certified in more exposed domains after the transition
- the manuscript therefore frames this layer as slower conversion into certified public resolution

## D. WS6 Conservative Inference

Primary gate file:
- [who_still_answers_ws6_small_sample_inference.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_ws6_small_sample_inference.csv)

Key promoted rows:

- `role_first_positive_recent90`: `CR2 p = 0.0108`, wild `p = 0.0276`, RI `p = 0.0100`
- `role_top_score_recent90`: `CR2 p = 0.0042`, wild `p = 0.0175`, RI `p = 0.0080`
- `bridge_accepted_vote_30d_gap`: `CR2 p = 0.0066`, wild `p = 0.0075`
- `bridge_first_positive_latency_gap`: `CR2 p = 0.0124`, wild `p = 0.0100`

Inferential hierarchy used in the manuscript:

1. infrastructure bridge
2. answer-role reallocation
3. durability

## E. WS6 Timing and Construct Sensitivity

Timing audit:
- [who_still_answers_ws6_timing_audit.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_ws6_timing_audit.csv)

Construct sensitivity:
- [who_still_answers_ws6_construct_sensitivity.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_ws6_construct_sensitivity.csv)

Main timing read:

- the paper does not support a pristine release-date shock
- the correct identity is a bounded `ChatGPT-period differential exposure` manuscript

Main construct-sensitivity read:

- role-family evidence is strongest at `90d` and `365d`
- the bridge is strongest for `recent_gap_first_vs_accepted`, not for generic entrant-share variables

## F. Timing Triangulation

Timing triangulation memo:
- [who_still_answers_timing_triangulation_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_timing_triangulation_2026-04-04.md)

Additional timing support files:

- [who_still_answers_external_ai_timing_layer_readout_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_external_ai_timing_layer_readout_2026-04-04.md)
- [who_still_answers_internal_ai_timing_trace_readout_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_internal_ai_timing_trace_readout_2026-04-04.md)
- [who_still_answers_external_ai_timing_results.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_external_ai_timing_results.csv)
- [who_still_answers_internal_ai_timing_results.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_internal_ai_timing_results.csv)

Safe read:

- timing remains bounded
- promoted bridge outcomes are now supported by an external AI-salience layer, a same-setting internal AI-title trace, and a strict question-side AI-ban timing layer
- this strengthens the timing story without converting the branch into a clean discontinuity design
- the strict question-side AI-ban timing layer is now the only canonical reviewer-facing restricted timing layer

Canonical same-setting timing support:

- [who_still_answers_disclosed_ai_corpus_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_disclosed_ai_corpus_2026-04-04.md)
- [who_still_answers_ai_ban_strict_question_side_2026-04-06.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_ai_ban_strict_question_side_2026-04-06.md)
- [who_still_answers_disclosed_ai_counts.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_disclosed_ai_counts.csv)
- [who_still_answers_ai_ban_strict_question_side_results.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_ai_ban_strict_question_side_results.csv)

Earlier broader ban-window and posthistory timing files remain archival scaffolds, not canonical reviewer-facing timing evidence.

### F1. Explicit Pre-Trend Diagnostics

The pre-trend issue is real enough that it should be shown rather than summarized.

Pre-shock slope tests:

| Outcome | High-tag pre-trend slope | p-value | Read |
| --- | ---: | ---: | --- |
| `residual_queue_complexity_index_mean` | `-0.000278` | `0.2694` | no detectable pre-trend |
| `brand_new_platform_share` | `+0.001024` | `<0.001` | meaningful pre-trend |
| `first_answer_1d_rate_closure` | `+0.001313` | `<0.001` | meaningful pre-trend |
| `accepted_vote_30d_rate` | `+0.000283` | `0.0038` | meaningful pre-trend |

Source table:
- [who_still_answers_pretrend_diagnostics_table.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_pretrend_diagnostics_table.csv)

Safe read:

- broad-window exposure comparisons should not be sold as clean parallel-trends DiD
- the long window identifies a persistent exposure gradient
- timing layers are used to ask whether the same part of the pipeline moves when AI becomes more salient, not to rescue a one-date treatment claim

## G. Exposure Validation Ladder

Exposure-validation memo:
- [who_still_answers_exposure_validation_ladder_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_exposure_validation_ladder_2026-04-04.md)

Supporting files:

- [who_still_answers_jetbrains_calibration_readout_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_jetbrains_calibration_readout_2026-04-04.md)
- [who_still_answers_jetbrains_calibration_table_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_jetbrains_calibration_table_2026-04-04.md)
- [who_still_answers_jetbrains_calibration_table.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_jetbrains_calibration_table.csv)

Safe read:

- exposure remains indirect in the main Stack Overflow setting
- the branch now supports that proxy through a ladder:
  - pre-shock tag-level exposure index
  - within-tag question-level residualization
- same-setting disclosed-AI thread corpus
- strict question-side disclosed-AI subset visible at posting time
- external JetBrains developer-survey calibration

### G1. Tag-Level Exposure Ranking

The paper now reports the full tag ordering directly.

| Rank | Tag | Tercile | Exposure index | Routine hits | Context hits | Pre-shock questions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `regex` | High | `1.657` | `1.303` | `0.012` | `21,515` |
| `2` | `pandas` | High | `1.149` | `0.960` | `0.019` | `17,115` |
| `3` | `numpy` | High | `0.863` | `0.810` | `0.063` | `3,554` |
| `4` | `sql` | High | `0.813` | `0.771` | `0.066` | `123,789` |
| `5` | `apache-spark` | High | `0.679` | `0.737` | `0.107` | `18,428` |
| `6` | `excel` | Middle | `0.512` | `0.520` | `0.023` | `54,913` |
| `7` | `javascript` | Middle | `0.485` | `0.532` | `0.051` | `529,061` |
| `8` | `bash` | Middle | `0.427` | `0.520` | `0.075` | `21,874` |
| `9` | `python` | Middle | `0.370` | `0.501` | `0.094` | `589,797` |
| `10` | `linux` | Middle | `-0.349` | `0.268` | `0.339` | `27,622` |
| `11` | `memory-management` | Low | `-0.523` | `0.323` | `0.542` | `2,329` |
| `12` | `android` | Low | `-0.582` | `0.393` | `0.629` | `162,195` |
| `13` | `firebase` | Low | `-0.827` | `0.358` | `0.780` | `34,085` |
| `14` | `multithreading` | Low | `-1.020` | `0.238` | `0.802` | `13,035` |
| `15` | `kubernetes` | Low | `-1.587` | `0.218` | `1.144` | `25,272` |
| `16` | `docker` | Low | `-2.068` | `0.176` | `1.415` | `45,761` |

Source table:
- [who_still_answers_exposure_ranking_table.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_exposure_ranking_table.csv)
- [who_still_answers_16tag_external_calibration_table.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_16tag_external_calibration_table.csv)

Safe read:

- readers can now judge the face validity of the exposure ordering directly
- every focal tag is also mapped to a JetBrains developer-survey cluster with a positive private-public answer-source gap
- the paper does not ask the ranking to be perfect
- it asks only that the ordering capture a plausible exposure gradient, which is then stress-tested through question-level residualization, leave-two-out checks, and external calibration

### G2. Sixteen-Tag External Calibration Surface

The final build maps every focal tag to its matched JetBrains calibration cluster. This is a transparency table, not an independent tag-by-tag validation.

| Rank | Tag | Exposure index | JetBrains cluster | Private-public answers gap |
| --- | --- | ---: | --- | ---: |
| `1` | `regex` | `1.657` | SQL / analytics | `0.069` |
| `2` | `pandas` | `1.149` | Python / data | `0.071` |
| `3` | `numpy` | `0.863` | Python / data | `0.071` |
| `4` | `sql` | `0.813` | SQL / analytics | `0.069` |
| `5` | `apache-spark` | `0.679` | Python / data | `0.071` |
| `6` | `excel` | `0.512` | SQL / analytics | `0.069` |
| `7` | `javascript` | `0.485` | JavaScript / web | `0.098` |
| `8` | `bash` | `0.427` | Shell / infra / cloud | `0.067` |
| `9` | `python` | `0.370` | Python / data | `0.071` |
| `10` | `linux` | `-0.349` | Shell / infra / cloud | `0.067` |
| `11` | `memory-management` | `-0.523` | Android / mobile | `0.080` |
| `12` | `android` | `-0.582` | Android / mobile | `0.080` |
| `13` | `firebase` | `-0.827` | JavaScript / web | `0.098` |
| `14` | `multithreading` | `-1.020` | Android / mobile | `0.080` |
| `15` | `kubernetes` | `-1.587` | Shell / infra / cloud | `0.067` |
| `16` | `docker` | `-2.068` | Shell / infra / cloud | `0.067` |

Source table:
- [who_still_answers_16tag_external_calibration_table.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_16tag_external_calibration_table.csv)

Safe read:

- JetBrains supports the public-private substitution premise in all matched technical clusters
- it does not mechanically validate exact tag ordering inside each cluster
- the manuscript therefore uses this table as construct calibration, not as identification

## H. Direct-AI-Use External Prototype

Positioning memo:
- [who_still_answers_aidev_positioning_note_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_aidev_positioning_note_2026-04-04.md)

Supporting files:

- [who_still_answers_aidev_second_setting_prototype_readout_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_aidev_second_setting_prototype_readout_2026-04-04.md)
- [who_still_answers_aidev_fe_results.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_aidev_fe_results.csv)
- [who_still_answers_aidev_strict_overlap_fe_results.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_aidev_strict_overlap_fe_results.csv)
- [who_still_answers_aidev_domain_overlap_upgrade_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_aidev_domain_overlap_upgrade_2026-04-04.md)
- [who_still_answers_aidev_domain_overlap_results.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_aidev_domain_overlap_results.csv)
- [who_still_answers_devgpt_source_map_2026-04-05.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_devgpt_source_map_2026-04-05.md)
- [who_still_answers_devgpt_positioning_2026-04-05.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_devgpt_positioning_2026-04-05.md)
- [who_still_answers_devgpt_sidecar_readout_2026-04-05.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_devgpt_sidecar_readout_2026-04-05.md)
- [who_still_answers_devgpt_sidecar_results.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_devgpt_sidecar_results.csv)

Safe read:

- `DevGPT` adds a thinner but more question-like direct-AI troubleshooting validation setting
- `AIDev` remains the strongest certification-focused answer to the `no direct AI-use external setting` critique
- neither is a one-for-one replication of Stack Overflow
- together they are stronger than either alone because they split the external job into a question-like problem-thread validation and a certification-focused public-review validation

## I. Display Packet and Figures

Display guide:
- [who_still_answers_option_c_display_packet.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_option_c_display_packet.md)
- [who_still_answers_rendered_tables_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_rendered_tables_2026-04-04.md)
- [who_still_answers_rendered_figures_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_rendered_figures_2026-04-04.md)

Core figures:

- [who_still_answers_durability_mechanism.png](D:/AI alignment/projects/stackoverflow_chatgpt_governance/figures/who_still_answers_durability_mechanism.png)
- [who_still_answers_answer_role_mechanism.png](D:/AI alignment/projects/stackoverflow_chatgpt_governance/figures/who_still_answers_answer_role_mechanism.png)
- [who_still_answers_infrastructure_bridge.png](D:/AI alignment/projects/stackoverflow_chatgpt_governance/figures/who_still_answers_infrastructure_bridge.png)
- [who_still_answers_ws6_conservative_inference.png](D:/AI alignment/projects/stackoverflow_chatgpt_governance/figures/who_still_answers_ws6_conservative_inference.png)
- [who_still_answers_ws6_timing_rank_panel.png](D:/AI alignment/projects/stackoverflow_chatgpt_governance/figures/who_still_answers_ws6_timing_rank_panel.png)

## J. Residual Queue Complexity Summary

The queue-composition layer relies on observable shape and complexity proxies. Summary statistics for the main inputs are:

| Variable | Mean | SD | Median |
| --- | ---: | ---: | ---: |
| `body_word_count` | `178.0` | `169.3` | `134.0` |
| `code_block_count` | `1.512` | `1.418` | `1.0` |
| `code_char_count` | `1,838.6` | `3,467.1` | `828.0` |
| `error_keyword_density` | `0.0000` | `0.0000` | `0.0000` |
| `tag_count_full` | `3.291` | `1.168` | `3.0` |
| `has_edit` | `0.542` | `0.498` | `1.0` |
| `residual_queue_complexity_index_mean` | `0.074` | `0.195` | `0.055` |

Source table:
- [who_still_answers_residual_queue_complexity_summary_stats.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/who_still_answers_residual_queue_complexity_summary_stats.csv)

## K. Interpretation Boundaries

This appendix supports the same four manuscript boundaries:

- no complete direct AI-use claim
- no clean one-date treatment story
- no claim that durability alone carries the paper
- no claim that the bridge proves mediation or generalized platform decline
- no claim that `DevGPT` or `AIDev` alone is a one-for-one Stack Overflow replacement

## L. Moderation / Closure Context Only

The project did inspect closure-side archival traces, but that line is not promoted as a main empirical pillar.

Supporting files:

- [build_who_still_answers_moderation_summary.json](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/build_who_still_answers_moderation_summary.json)
- [build_who_still_answers_moderation_panel.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/build_who_still_answers_moderation_panel.csv)
- [build_who_still_answers_moderation_layer.py](D:/AI alignment/projects/stackoverflow_chatgpt_governance/scripts/build_who_still_answers_moderation_layer.py)

Safe read:

- this build observes current closure states, not full historical moderation paths
- deletion events are not reconstructed here
- the line is therefore retained only as appendix context on policy-side content handling, not as a reviewer-facing identification layer
