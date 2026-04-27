# Reviewer Memo: Option C Anticipated Objections and Prepared Responses

Date: April 5, 2026

## Purpose

This memo is the active reviewer-response bank for the `Option C` manuscript:

- [who_still_answers_option_c_manuscript.md](D:/AI alignment/projects/stackoverflow_chatgpt_governance/paper/who_still_answers_option_c_manuscript.md)

It supersedes the earlier entrant-share response bank.

## Objection 1: "This is just another Stack Overflow plus ChatGPT paper."

**Prepared response**

The manuscript studies a different object from the nearest Stack Overflow / GenAI papers. The question is not whether activity or answer text changed. The question is whether the residual public queue, visible answer roles, and later certification moved together after private AI became a credible outside option. The evidence is organized around residual-queue recomposition, incumbent/recent-entrant role reallocation, certification consequences, and bounded durability support.

## Objection 2: "You do not observe AI use."

**Prepared response**

Correct: we do not observe AI use exhaustively in Stack Overflow. We observe selective thread-level disclosure.

We handle this in three ways. First, we keep the main design as a pre-shock differential exposure architecture. Second, we add same-setting thread-level disclosure evidence, including a strict question-side subset visible when the question is posted. Third, we add external evidence: JetBrains microdata calibrate the substitution premise across matched technical clusters, and two direct-AI sidecars (`DevGPT` and `AIDev`) show a similar response-versus-certification split in settings where AI use is directly observed.

## Objection 3: "Timing is too dirty for ChatGPT causality."

**Prepared response**

We agree. This is not a pristine one-date causal design.

We therefore keep the timing claim narrow. We show that the mechanism strengthens when AI is more salient (external pageviews and an internal AI-title trace). We also show one canonical restricted timing check: around `2022-12-05`, in a strict question-side disclosed-AI subset, accepted-window certification outcomes weaken more clearly than immediate answer arrival. This is a restricted timing result, not a discontinuity claim. Broader moderation and closure traces are appendix-only descriptive context because the dump does not recover full historical enforcement trajectories.

## Objection 4: "With only 16 clusters, the main results are not trustworthy."

**Prepared response**

The manuscript foregrounds the few-cluster problem and keeps an explicit evidence hierarchy. We report clustered inference, `CR2`, wild-cluster bootstrap, and randomization inference where appropriate. We do not claim that every mechanism family is equally strong. The bridge is strongest. Role reallocation is next. Durability is supportive and bounded under the harshest resampling.

## Objection 5: "Durability still seems over-claimed."

**Prepared response**

The manuscript already narrows durability language. Durability is part of the mechanism stack. It is not the headline.

## Objection 6: "Your answer-role story collapses because accepted-current is not clearly zero."

**Prepared response**

We do not claim that accepted-current roles are flat. We claim an asymmetry: the entrant-heavy shift is larger in early and endorsed roles than in accepted-current certification. That is exactly what the role results show. We also treat the sharpest asymmetry term as secondary under conservative inference.

## Objection 6b: "Is this just AI-augmented novice entry?"

**Prepared response**

No. The final decomposition explicitly rules against a simple brand-new-user takeover. The brand-new-platform share is negative in the main role decompositions. The stronger same-setting read is thinner incumbent capacity and broader recent-entrant role reallocation: expert incumbent mean answer activity falls, incumbent shares fall in the main answer roles, and recent entrants become more visible in early and endorsed roles.

## Objection 7: "The bridge is just mediation language without real identification."

**Prepared response**

We do not claim mediation. The bridge is a consequence layer. The finding is that tag-months with larger first-vs-accepted role gaps have slower endorsed resolution and weaker public certification. That is meaningful even without a mediation design.

## Objection 8: "This still sounds too narrow for a full article."

**Prepared response**

The paper is not a single entrant-share coefficient. It offers a specific reallocation pattern in public answer supply and links that pattern to downstream certification. It shows where in the pipeline entrant visibility shifts. It is explicit about conservative inference and timing limits.

Outlet risk is now mostly editorial: keeping the hierarchy clear and not overselling identification.
