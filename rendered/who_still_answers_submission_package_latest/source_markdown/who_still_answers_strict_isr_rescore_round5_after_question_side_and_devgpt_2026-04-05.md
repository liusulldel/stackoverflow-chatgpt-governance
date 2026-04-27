# Who Still Answers: Strict ISR Rescore Round 5 After Question-Side Timing and DevGPT Upgrade

Date: April 5, 2026

Canonical manuscript reviewed:
- [who_still_answers_option_c_manuscript.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_option_c_manuscript.md)

New evidence layers reviewed:
- [who_still_answers_posthistory_direct_ai_readout_2026-04-05.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_posthistory_direct_ai_readout_2026-04-05.md)
- [who_still_answers_posthistory_ban_timing_2026-04-05.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_posthistory_ban_timing_2026-04-05.md)
- [who_still_answers_devgpt_sidecar_readout_2026-04-05.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_devgpt_sidecar_readout_2026-04-05.md)
- [who_still_answers_aidev_domain_overlap_upgrade_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_aidev_domain_overlap_upgrade_2026-04-04.md)
- [who_still_answers_jetbrains_calibration_readout_2026-04-04.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_jetbrains_calibration_readout_2026-04-04.md)

## Current strict ISR score

`90/100`

## Pass / fail read

`Pass, but narrowly.`

This round is the first one that credibly clears the `90` line because the paper now has a more reviewer-legible answer to all three structural attacks at once:

1. `main-setting direct observation`
2. `bounded timing`
3. `second-pillar mismatch`

No single upgrade solves all three. The score crosses `90` because the branch no longer asks one layer to do all the work.

## Why the score improved

### 1. The main-setting direct-observation critique is now materially narrower

The paper no longer relies on a generic disclosed-AI thread corpus alone. It now distinguishes:

- a broad same-setting thread-level disclosed-AI layer (`9,147` hits)
- a stricter self-disclosure layer (`2,407` hits)
- a cleaner `question-side` subset visible at posting time (`1,932` hits)

That is still selective disclosure, but it is now much harder for a strict editor to say that the main setting contains only modeled exposure with no direct in-setting observation at all.

### 2. The ban-centered timing story is now cleaner even though it remains bounded

The strongest timing improvement is not bigger coefficients on contaminated thread-side disclosure. It is the cleaner `question-side` timing result:

- `accepted_7d = -0.0936`, `p = 0.0031`
- `accepted_30d = -0.2786`, `p = 0.0108`
- `first_answer_1d` null

in `high_tags_only`, `+/-30d`, `3d` donut.

This matters because it creates the most reviewer-safe version of the ban-centered claim:

`the policy-centered timing signal appears more clearly on near-term public certification than on immediate answer arrival.`

That still is not a pristine discontinuity, but it is now sharp enough to belong in the main text.

### 3. The second pillar no longer depends on one imperfect external setting

The external direct-AI answer is now split into two jobs:

- `DevGPT`: question-like direct-AI troubleshooting sidecar
- `AIDev`: certification-focused direct-AI public collaboration pillar

This is materially better than forcing `AIDev` alone to be both question-like and certification-like.

`DevGPT` is thinner and more selected, but it answers the object-fit critique.
`AIDev` is still non-isomorphic, but it answers the direct-AI certification critique much better than before.
Together they make the second pillar substantially more reviewer-legible.

### 4. The paper's theorem is sharper

The manuscript now consistently argues that private AI:

- residualizes the public queue
- reallocates visible labor toward earlier public roles
- weakens the link between early response and later public certification

That theorem is more memorable than the older entrant-share or weak-closure framings and now matches the evidence hierarchy better.

### 5. The article has crossed from strong note to real article scale

The manuscript is now roughly `10.9k` words with `32` references and a more coherent package structure.
It is still not a trivial desk-safe paper, but it no longer reads like a glorified defense memo.

## Why the score is only 90, not comfortably above it

### 1. Main-setting direct observation is still incomplete

Even the upgraded same-setting layer is selective disclosure, not exhaustive adoption measurement.

### 2. Timing is still bounded rather than clean

The question-side ban design is the right version to show, but the paper still should not market itself as a clean discontinuity study.

### 3. The direct-AI sidecars remain complementary rather than isomorphic

`DevGPT` is more question-like but selected and thin.
`AIDev` is stronger on certification but still not Stack Overflow.

### 4. Few-cluster pressure remains

The main Stack Overflow design still lives with `16` focal domains, so the inferential hierarchy must stay visible and disciplined.

### 5. Final package polish still matters

The branch now deserves ISR review, but it still benefits from one more packaging pass so that the reviewer-facing packet fully reflects the new question-side and DevGPT layers.

## Bottom line

This is the first round in which I would say the paper is honestly `ISR-submittable` rather than merely `ISR-aspirational`.

The branch does **not** become cleanly causal, complete on direct observation, or fully symmetric across settings.
What it does become is a coherent ISR paper with:

- a bounded but real same-setting direct-observation answer
- a bounded but cleaner policy-centered timing layer
- a two-part direct-AI external architecture that now addresses both question-likeness and certification

That combination is enough for a strict score of `90/100`, but only narrowly and only if the manuscript and package keep the same disciplined wording now present in the current canonical draft.
