# Staged Public Resolution in the Generative-AI Era: Evidence from Stack Overflow

Replication code, processed tag-month panels, figures, and the manuscript / appendix / cover-letter PDF bundle for:

> *Staged Public Resolution in the Generative-AI Era: Evidence from Stack Overflow*
> (working title; an alternate, longer title in the manuscript file is *Private AI, Public Answer Work, and Certification at Stack Overflow*).

## Abstract

Private generative AI creates a new outside option for technical problem solving. Its effects on public knowledge platforms appear less in raw activity counts than in *which* questions still reach the public archive, *who* performs visible answer work inside the residual queue, and *how* that work becomes certified public resolution. Using the official Stack Overflow `2025Q4` dump (`N = 2,322,009` focal-tag questions across 16 technical domains, 2020-01-01 through 2025-12-31), the design contrasts higher- vs. lower-substitutability domains. The headline result is a **staged-resolution split**: rapid public response weakens, while late acceptance does not deteriorate in the same direction. Newer entrants become more visible in early-answer and endorsement roles than in accepted-answer certification, and tag-months with larger early-vs.-accepted role gaps display weaker downstream certification. The paper reframes platform deterioration not as a uniform traffic collapse but as a *staged* unbundling of public resolution.

## Headline Results (canonical 3-rung ladder)

| Rung | Outcome | Coefficient | Clustered p | Wild-cluster p |
|---|---|---:|---:|---:|
| **1. Rapid response weakens** | `first_answer_1d` | **−3.25 pp** | 0.0019 | 0.0451 |
| **2. Stage contrast** | `any_answer_7d` | **−2.67 pp** | — | — |
| **3. Late acceptance does not move with response** | `accepted_vote_30d` | **+0.91 pp** | 0.0016 | 0.0501 |

Sample: `2,322,009` focal-tag questions, `16` technical domains (`bash`, `excel`, `javascript`, `numpy`, `pandas`, `python`, `regex`, `sql`, `apache-spark`, `android`, `docker`, `firebase`, `kubernetes`, `linux`, `memory-management`, `multithreading`), `2020-01-01` through `2025-12-31`. Window-trajectory persistence: the first-answer-1d effect is `−2.04 pp` through `2023-02`, `−2.71 pp` through `2023-12`, and `−3.25 pp` over the full 2025 window — the slowdown emerges early and persists.

## Repository Layout

```
.
├── README.md
├── LICENSE                       (MIT)
├── scripts/                      87 build / analysis scripts (Python + PowerShell)
├── processed/                    Tag-month panels, model results, validation outputs
│                                  (CSV / JSON; question-level dumps excluded — see below)
├── figures/                      43 PNG figures from the submission package
├── rendered/
│   └── who_still_answers_submission_package_latest/
│                                 18 rendered PDFs + source markdown for the
│                                 full multi-agent submission bundle
└── isr_submission_package/
    ├── 01_Manuscript.pdf
    ├── 02_Online_Appendix.pdf
    └── 03_Cover_Letter.pdf
```

## Data Availability

The empirical pipeline starts from the **official Stack Exchange data dump**, which is published under the Creative Commons CC BY-SA license at <https://archive.org/details/stackexchange>. The relevant snapshots for this paper are the periodic `stackexchange_2025-12-31` (and earlier) dumps containing `Posts.xml`, `Users.xml`, `Comments.xml`, `Votes.xml`, `Tags.xml`, and `PostHistory.xml` for `stackoverflow.com` and the smaller second-setting sites used for boundary checks (`unix.stackexchange.com`, `dba.stackexchange.com`, `superuser.com`).

**This repository does not redistribute the raw Stack Exchange archives.** To replicate the pipeline:

1. Download the relevant dump from `archive.org/details/stackexchange`.
2. Place the extracted parquet/XML files under a local `raw/` directory (see `scripts/build_stackoverflow_2025_dump_extension.py` and similar build scripts for expected paths).
3. Run the build scripts in `scripts/` in roughly the order suggested by their filenames (`build_stackoverflow_2025_dump_extension.py` → `build_question_level_exposure_index_api.py` → role-mechanism / closure-ladder builders → `build_extended_sample_results.py` / `build_harmonized_integration_table.py`).
4. Tag-month aggregated outputs in `processed/` should reproduce within numerical tolerance.

Per-question and per-user panels are intentionally **excluded** from this repository because (a) they are direct derivatives of CC BY-SA Stack Exchange content and the canonical redistribution venue is `archive.org`, and (b) several files exceed GitHub's per-file and per-repo size limits.

## What is in `processed/`

Tag-month panels, model coefficient tables, validation summaries, placebo tests, permutation tests, and event-study aggregates — all of which are computed *from* the raw Stack Exchange dump but contain only aggregated, non-reidentifying data. Examples:

- `closure_ladder_model_results.csv`, `closure_ladder_results.json`
- `extended_sample_model_results.csv`, `extended_sample_wild_cluster_bootstrap.csv`
- `harmonized_integration_model_results.csv`, `harmonized_integration_results.json`
- `large_design_paper_stats.json`, `large_design_permutation_tests.csv`, `large_design_placebo_break_tests.csv`
- `event_study_questions.csv`, `event_study_views.csv`
- `who_still_answers_*.csv` and `*.json` — role-mechanism, durability, certification-bridge results

## What is in `rendered/who_still_answers_submission_package_latest/`

The full multi-PDF submission bundle plus its underlying source markdown:

```
00_Submission_Package_Index.pdf
01_Manuscript.pdf
02_Online_Appendix.pdf
03_Cover_Letter.pdf
04_Reviewer_Memo.pdf
05_Reviewer_Question_Bank.pdf
06_Strict_ISR_Rescore.pdf
07_Multi_Agent_Synthesis.pdf
08_Display_Packet.pdf
09_Rendered_Tables.pdf
10_Rendered_Figures.pdf
11_Disclosed_AI_Corpus.pdf
12_AI_Ban_Strict_Question_Side.pdf
13_Coauthor_Strategy.pdf
14_AIDev_Domain_Overlap_Upgrade.pdf
15_Causal_Mining_Synthesis.pdf
16_DevGPT_Sidecar.pdf
17_Finish_Empirical_Upgrades.pdf
18_Final_Finish_Recommendation.pdf
source_markdown/                  raw markdown for all of the above
```

The compact 3-PDF `isr_submission_package/` directory contains only the manuscript, appendix, and cover letter for readers who only want the journal-bound version.

## Citation

```bibtex
@unpublished{liu2026staged,
  author = {Liu, Sully},
  title  = {Staged Public Resolution in the Generative-AI Era: Evidence from Stack Overflow},
  year   = {2026},
  note   = {Working paper. Replication: \url{https://github.com/liusulldel/stackoverflow-chatgpt-governance}}
}
```

## License

Code, scripts, and processed outputs in this repository are released under the MIT License (see `LICENSE`). The underlying Stack Exchange dump remains under **CC BY-SA 4.0** at the source; any derivative panels you reconstruct from it inherit those terms.
