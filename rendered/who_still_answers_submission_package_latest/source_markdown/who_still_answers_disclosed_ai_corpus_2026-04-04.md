# Same-Setting Disclosed-AI Corpus

Date: April 4, 2026

## Design

This build expands the earlier title-only disclosed-AI prototype into a thread-level corpus using the local `2025Q4` Stack Overflow dump.
The detector now scans question titles, question bodies, answer bodies, question comments, and answer comments.
Pattern matching was also tightened to reduce obvious false positives from non-AI technical terms.

## Topline

- total focal questions scanned: `2,322,009`
- disclosed-AI questions: `9,996`
- disclosed-AI share: `0.4305%`
- question-title hits: `1,244`
- question-body hits: `7,229`
- answer-body hits: `2,340`
- question-comment hits: `1,613`
- answer-comment hits: `1,365`
- non-title thread additions beyond the title-only prototype: `8,752`
- pre-ChatGPT hits: `1,001`
- post-ChatGPT hits: `8,995`
- pre-ban hits: `1,021`
- post-ban hits: `8,975`
- comments available locally: `True`

## Read

This is still not a full same-setting direct-AI measurement layer, because disclosure remains selective.
It is materially stronger than the earlier build because it now works at the question-thread level and includes comment-layer disclosures.
That makes it a stronger base for reviewer-facing timing and direct-observation discussion than the earlier title-plus-body corpus.

## Files

- hits parquet: `D:\AI alignment\projects\stackoverflow_chatgpt_governance\processed\who_still_answers_disclosed_ai_question_hits.parquet`
- hits csv: `D:\AI alignment\projects\stackoverflow_chatgpt_governance\processed\who_still_answers_disclosed_ai_question_hits.csv`
- counts csv: `D:\AI alignment\projects\stackoverflow_chatgpt_governance\processed\who_still_answers_disclosed_ai_counts.csv`
- tag-month counts: `D:\AI alignment\projects\stackoverflow_chatgpt_governance\processed\who_still_answers_disclosed_ai_tag_month_counts.csv`
- summary json: `D:\AI alignment\projects\stackoverflow_chatgpt_governance\processed\who_still_answers_disclosed_ai_summary.json`
