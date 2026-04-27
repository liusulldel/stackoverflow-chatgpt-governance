# Causal-Mining Synthesis

Date: 2026-04-06

This memo records what came back from the three causal-mining tracks launched after the latest referee-style attack. The goal was not to turn the paper into a clean causal design. The goal was narrower: identify whether any one of these lines materially improves reviewer confidence in the paper's main claim.

## Bottom Line

Only one line is clearly worth keeping in the paper-facing package:

1. `AI-ban restricted design upgrade`: keep, but only as a bounded timing consistency check.

The other two lines are much weaker:

2. `Moderation / deletion / closure policy layer`: keep only if needed for appendix context; do not promote.
3. `Contributor fixed-effects around first disclosure`: do not promote; likely cut unless a much stricter disclosure definition is rebuilt.

## 1. AI-Ban Restricted Design Upgrade

Worker lines: `Volta`, `Sartre`

Files:

- `processed/who_still_answers_ai_ban_results.csv`
- `processed/who_still_answers_ai_ban_diagnostics.csv`
- `processed/who_still_answers_ai_ban_summary.json`
- `scripts/build_who_still_answers_ai_ban_restricted_timing.py`

What improved:

- Smaller event windows (`7` to `35` days)
- Smaller donuts (`0`, `1`, `3`, `5`)
- Exact-match strata on `primary_tag x exposure_bin`
- Trimmed comparison sets with both disclosed and non-disclosed threads

Best usable result:

- `7-day window`, `1-day donut`, `matched_all_tags`, `first_answer_1d`
- Coefficient: `-0.4674`
- `p = 0.0233`
- `n_disclosed = 49`

Why this is still not a clean restricted design:

- Results are unstable across windows, donuts, and outcomes.
- Comparison balance improves on exposure but remains imperfect on timing and post-ban share.
- The stricter question-side sample is very small and highly concentrated.
- Placebo and few-cluster concerns remain.

Paper recommendation:

- Keep as a bounded same-setting timing layer.
- Do not sell as a clean discontinuity.
- Use it to support the claim that the main pattern is not driven only by a broad post-period dummy.

## 2. Moderation / Deletion / Closure Policy Layer

Worker lines: `Chandrasekhar`, `Russell`

Files:

- `processed/build_who_still_answers_moderation_question_level.parquet`
- `processed/build_who_still_answers_moderation_panel.csv`
- `processed/build_who_still_answers_moderation_summary.json`
- `scripts/build_who_still_answers_moderation_layer.py`

What the current build actually measures:

- Current closure states via `ClosedDate` in `Posts.xml`
- `closed_archive`
- `closed_7d`
- `closed_30d`

Topline diff-in-diff readouts:

- `closed_7d`: `+0.0060`
- `closed_30d`: `+0.0065`
- `closed_archive`: `+0.0061`

Why this line is weak:

- Deletion events are not available in this build.
- Closure is measured only from current `ClosedDate`, so it misses historical reversals and does not reconstruct full moderation trajectories.
- There is no formal inferential architecture yet; the current outputs are descriptive summaries.
- Even substantively, this line is not obviously aligned with the paper's central certification story.

Paper recommendation:

- At most, use as appendix context on policy-side content handling.
- Do not present as a causal governance result.
- Do not let it distract from the stronger bridge and role-reallocation story.

## 3. Contributor Fixed Effects Around First Disclosure

Worker lines: `Carson`, `Dalton`

Files:

- `processed/who_still_answers_disclosure_fe_disclosures.csv`
- `processed/who_still_answers_disclosure_fe_user_month_panel.csv`
- `scripts/build_who_still_answers_disclosure_fe_event_study.py`

What came back:

- A usable disclosure panel was built.
- User-month panel size: `49,848`
- Users: `3,133`
- Disclosure events: `19,969` across `12,922` users in the raw disclosure file

Why this line is currently not paper-safe:

- The disclosure detector still produces many false positives.
- There are hits before the ChatGPT period, including early `gemini`, `copilot`, `claude`, and `openai` mentions that clearly reflect generic-word noise rather than true GenAI disclosure.
- Example: pre-`2022-11-30` hits total `2,999`.
- Because the treatment event itself is noisy, the FE design is not a reliable mechanism test.
- The contributor-linked sample is already selective, and the answer-side FE design is thinner still.

Paper recommendation:

- Do not promote in current form.
- If revisited, rebuild around a much stricter disclosure definition:
  - question-side or answer-side explicit self-disclosure,
  - post-`2022-11-30`,
  - tool-name patterns that exclude generic-word collisions.
- Until then, this line is not worth main-text space.

## Ranking by Value Added

1. `AI-ban restricted design upgrade`
   Useful. Keep as a bounded timing check.

2. `Moderation / deletion / closure policy layer`
   Weak but not useless. Appendix-only if needed.

3. `Contributor fixed effects around first disclosure`
   Not worth keeping in present form.

## Implication for Revision Order

The causal-mining exercise does not overturn the existing paper identity. It sharpens it.

The paper is still best understood as:

- a careful reallocation paper,
- with one stronger same-setting restricted timing layer,
- and without a clean adoption-level causal design.

So the right next move is not to keep opening more causal sidecars. The right move is to integrate the one keeper (`AI-ban`) carefully, mention the moderation line only if useful, and cut or quarantine the FE line.
