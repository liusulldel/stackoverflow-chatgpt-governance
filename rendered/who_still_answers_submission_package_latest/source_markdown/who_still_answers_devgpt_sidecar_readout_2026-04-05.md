# Who Still Answers DevGPT Sidecar Readout

Date: 2026-04-05

## Feasibility Verdict

Feasible. DevGPT is locally downloadable and includes issue/discussion threads with direct ChatGPT-sharing traces.
This makes it more question-like than AIDev PR-only data for a sidecar focused on public response and closure/certification logic.

## Data Built

- Thread panel (issues + discussions, deduped to latest snapshot): `321` rows
- Issue rows: `284`; discussion rows: `37`
- PR support panel (deduped): `164` rows

## Sidecar Logic

Question-like layer: `updated_7d` (early public response proxy) vs `closed_30d` (formalized closure).
Divergence metric: `response_close_gap_30d = updated_7d - closed_30d`.
Direct-AI feature layer: prompt/answer tokens, multi-turn chat, and context-heaviness parsed from shared prompts.

## Topline Rates

- Thread updated within 7d: `0.617`
- Thread closed within 30d: `0.343`
- Thread closed within 90d: `0.389`
- Mean response-close gap (30d): `0.274`

## Context-Heavy Split (Descriptive)

- low-context: n=`160`, updated_7d=`0.619`, closed_30d=`0.306`, gap=`0.312`
- high-context: n=`161`, updated_7d=`0.615`, closed_30d=`0.379`, gap=`0.236`

## Clustered LPM Results

- `devgpt_thread` / `issue_discussion_all` / `updated_7d` / `context_heavy`: coef `0.0685`, clustered `p = 0.2364` (nobs `321`, repos `258`)
- `devgpt_thread` / `issue_discussion_all` / `updated_7d` / `multi_turn`: coef `0.0821`, clustered `p = 0.1861` (nobs `321`, repos `258`)
- `devgpt_thread` / `issue_discussion_all` / `updated_7d` / `log_ai_tokens`: coef `-0.0405`, clustered `p = 0.0022` (nobs `321`, repos `258`)
- `devgpt_thread` / `issue_discussion_all` / `closed_30d` / `context_heavy`: coef `0.1102`, clustered `p = 0.0949` (nobs `321`, repos `258`)
- `devgpt_thread` / `issue_discussion_all` / `closed_30d` / `multi_turn`: coef `-0.0393`, clustered `p = 0.6012` (nobs `321`, repos `258`)
- `devgpt_thread` / `issue_discussion_all` / `closed_30d` / `log_ai_tokens`: coef `-0.0214`, clustered `p = 0.1821` (nobs `321`, repos `258`)
- `devgpt_thread` / `issue_discussion_all` / `response_close_gap_30d` / `context_heavy`: coef `-0.0417`, clustered `p = 0.6303` (nobs `321`, repos `258`)
- `devgpt_thread` / `issue_discussion_all` / `response_close_gap_30d` / `multi_turn`: coef `0.1215`, clustered `p = 0.2194` (nobs `321`, repos `258`)
- `devgpt_thread` / `issue_discussion_all` / `response_close_gap_30d` / `log_ai_tokens`: coef `-0.0190`, clustered `p = 0.3951` (nobs `321`, repos `258`)
- `devgpt_thread` / `issue_only_updated` / `closed_30d_given_updated` / `context_heavy`: coef `0.1372`, clustered `p = 0.1560` (nobs `170`, repos `141`)
- `devgpt_thread` / `issue_only_updated` / `closed_30d_given_updated` / `multi_turn`: coef `0.0387`, clustered `p = 0.6952` (nobs `170`, repos `141`)
- `devgpt_thread` / `issue_only_updated` / `closed_30d_given_updated` / `log_ai_tokens`: coef `-0.0286`, clustered `p = 0.1909` (nobs `170`, repos `141`)
- `devgpt_pr_support` / `pr_all` / `merged_30d` / `context_heavy`: coef `0.2291`, clustered `p = 0.0268` (nobs `164`, repos `129`)
- `devgpt_pr_support` / `pr_all` / `merged_30d` / `multi_turn`: coef `-0.1255`, clustered `p = 0.1768` (nobs `164`, repos `129`)
- `devgpt_pr_support` / `pr_all` / `merged_30d` / `log_ai_tokens`: coef `0.0030`, clustered `p = 0.8989` (nobs `164`, repos `129`)
- `devgpt_pr_support` / `pr_all` / `merged_7d` / `context_heavy`: coef `0.1804`, clustered `p = 0.0997` (nobs `164`, repos `129`)
- `devgpt_pr_support` / `pr_all` / `merged_7d` / `multi_turn`: coef `-0.2248`, clustered `p = 0.0113` (nobs `164`, repos `129`)
- `devgpt_pr_support` / `pr_all` / `merged_7d` / `log_ai_tokens`: coef `0.0106`, clustered `p = 0.5959` (nobs `164`, repos `129`)

## What This Adds (and What It Does Not)

Adds: a direct-AI-use sidecar with question-like threads and an explicit response-versus-closure decomposition.
Does not add: complete direct observation of all AI use on GitHub (still a selective self-sharing sample).
Therefore this is a mechanism-validating sidecar, not a standalone causal replacement for the main Stack Overflow design.
