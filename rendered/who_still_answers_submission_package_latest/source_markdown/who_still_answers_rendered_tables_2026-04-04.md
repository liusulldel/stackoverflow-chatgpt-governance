# Rendered Tables: Who Still Answers Option C

Date: April 4, 2026

Canonical manuscript:
- [who_still_answers_option_c_manuscript.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_option_c_manuscript.md)

## Table 1. Sample and Design Architecture

| Component | Unit | Count / window | Role |
| --- | --- | ---: | --- |
| Canonical Stack Overflow backbone | Questions | `2,035,885` | Main-setting evidence base |
| Canonical Stack Overflow backbone | Valid answers | `2,391,883` | Role and certification outcomes |
| Focal domains | Tags | `16` | Exposure contrast |
| Durability family | Contributor-entry windows | `30/90/180/365d` | Family A |
| Role family | Question-level roles | `first_answer`, `first_positive`, `top_score`, `accepted_current` | Family B |
| Bridge family | Tag-month outcomes | latency and certification | Family C |
| External timing validation | Tag-month layer | external salience + internal AI-title trace | Timing ladder |
| Exposure validation | Question-level + survey | within-tag residualization + JetBrains clusters | Construct ladder |
| Direct-AI-use external setting | Pull requests | overlap repos and strict overlap repo-months | Second-setting prototype |

## Table 2. Promoted Mechanism Results

| Family | Outcome / estimand | Coefficient | Cluster p | CR2 p | Wild / RI read |
| --- | --- | ---: | ---: | ---: | --- |
| Bridge | `accepted_vote_30d_rate ~ recent_gap_first_vs_accepted` | `-0.0752` | `0.0019` | `0.0066` | wild `0.0075` |
| Bridge | `first_positive_answer_latency_mean ~ recent_gap_first_vs_accepted` | `+283.9` min | `0.0039` | `0.0124` | wild `0.0100` |
| Role | `first_positive recent_entrant_90d_share` | `+0.0281` | `0.0022` | `0.0108` | wild `0.0276`, RI `0.0100` |
| Role | `top_score recent_entrant_90d_share` | `+0.0272` | `0.0003` | `0.0042` | wild `0.0175`, RI `0.0080` |
| Durability | `return_365d` | `-0.0266` | `0.0004` | `0.0047` | wild `0.0652`, RI `0.1242` |
| Durability | `one_shot_365d` | `+0.0266` | `0.0004` | `0.0047` | wild `0.0802`, RI `0.0982` |

Interpretation note:
- The promoted hierarchy remains `bridge -> role -> durability`.
- Durability survives clustered and CR2 inference but stays more bounded under the strictest resampling checks.

## Table 3. Timing and Exposure Validation Ladder

### Panel A. Timing Triangulation

| Layer | Outcome | Coefficient | Cluster p | Permutation p |
| --- | --- | ---: | ---: | ---: |
| External AI salience | `recent_gap_first_vs_accepted` | `+0.0036` | `0.0123` | `0.0336` |
| External AI salience | `first_positive_answer_latency_mean` | `+24.0` min | `0.0033` | `0.0138` |
| Internal AI-title trace | `recent_gap_first_vs_accepted` | `+0.0065` | `0.0103` | `0.1090` |
| Internal AI-title trace | `first_positive_answer_latency_mean` | `+48.1` min | `0.0006` | `0.0088` |

### Panel B. Question-Level Exposure Validation

| Layer | Outcome | Coefficient | Cluster p |
| --- | --- | ---: | ---: |
| Queue residualization | `pred_mean`, `high_tag x post` | `-0.0656` | `0.0564` |
| Within-tag continuous high tags | `accepted_30d_rate`, `post x exp_mean` | `+0.0140` | `<0.001` |
| Within-tag continuous high tags | `response_cert_gap`, `post x exp_mean` | `-0.0123` | `<0.001` |
| Within-tag continuous high tags | `accepted_given_fast_rate`, `post x exp_mean` | `+0.0108` | `<0.001` |

### Panel C. JetBrains External Calibration

| Cluster | ChatGPT / AI answers share | Stack Overflow answers share | Private-public gap |
| --- | ---: | ---: | ---: |
| `SQL / analytics` | `0.206` | `0.137` | `0.069` |
| `Python / data` | `0.218` | `0.147` | `0.071` |
| `JavaScript / web` | `0.227` | `0.128` | `0.098` |
| `Android / mobile` | `0.221` | `0.141` | `0.080` |
| `Shell / infra / cloud` | `0.214` | `0.147` | `0.067` |

Interpretation note:
- The ladder now runs `tag-level exposure -> question-level residualization -> external developer-survey calibration`.
- This materially narrows the proxy critique without turning main-setting exposure into direct AI-use measurement.

## Table 4. Direct-AI-Use External Prototype (`AIDev`)

### Panel A. Overlap Repositories

| Outcome | AI-authored PR coefficient | SE | p-value |
| --- | ---: | ---: | ---: |
| `first_review_7d` | `+0.4075` | `0.0297` | `<0.001` |
| `approved_30d` | `+0.2414` | `0.0300` | `<0.001` |
| `changes_requested_30d` | `+0.0472` | `0.0062` | `<0.001` |
| `merged_30d` | `-0.2237` | `0.0307` | `<0.001` |

### Panel B. Strict Overlap Repo-Month

| Outcome | AI-authored PR coefficient | SE | p-value |
| --- | ---: | ---: | ---: |
| `first_review_7d` | `+0.4203` | `0.0361` | `<0.001` |
| `approved_30d` | `+0.2444` | `0.0388` | `<0.001` |
| `changes_requested_30d` | `+0.0412` | `0.0065` | `<0.001` |
| `merged_30d` | `-0.2056` | `0.0377` | `<0.001` |
| `log_first_review_hours` | `+0.2227` | `0.1010` | `0.0275` |

Interpretation note:
- `AIDev` is a direct-AI-use external prototype, not a one-for-one Stack Overflow replication.
- Its value is the analogous certification pattern: faster visible review, but weaker later merge certification.
