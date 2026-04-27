# Strict Question-Side AI-Ban Upgrade

Date: April 6, 2026

## Goal

This build re-estimates the ban-centered timing check using the stricter question-side disclosure layer rather than the older, broader disclosed-AI hit table.
The idea is simple: keep only disclosures visible when the question is posted, then ask whether accepted-window outcomes weaken after the Stack Overflow AI ban.

## Window counts

- `all_tags`, `+/-30d`, donut `0d`: `662` questions, hits `40`, pre/post = `10/30`
- `all_tags`, `+/-30d`, donut `3d`: `590` questions, hits `37`, pre/post = `7/30`
- `all_tags`, `+/-45d`, donut `0d`: `1029` questions, hits `60`, pre/post = `11/49`
- `all_tags`, `+/-45d`, donut `3d`: `957` questions, hits `57`, pre/post = `8/49`
- `all_tags`, `+/-45d`, donut `7d`: `865` questions, hits `49`, pre/post = `7/42`
- `all_tags`, `+/-60d`, donut `0d`: `1399` questions, hits `89`, pre/post = `14/75`
- `all_tags`, `+/-60d`, donut `3d`: `1327` questions, hits `86`, pre/post = `11/75`
- `all_tags`, `+/-60d`, donut `7d`: `1235` questions, hits `78`, pre/post = `10/68`
- `high_tags_only`, `+/-30d`, donut `0d`: `404` questions, hits `35`, pre/post = `8/27`
- `high_tags_only`, `+/-30d`, donut `3d`: `363` questions, hits `32`, pre/post = `5/27`
- `high_tags_only`, `+/-45d`, donut `0d`: `625` questions, hits `53`, pre/post = `9/44`
- `high_tags_only`, `+/-45d`, donut `3d`: `584` questions, hits `50`, pre/post = `6/44`
- `high_tags_only`, `+/-45d`, donut `7d`: `515` questions, hits `42`, pre/post = `5/37`
- `high_tags_only`, `+/-60d`, donut `0d`: `860` questions, hits `77`, pre/post = `12/65`
- `high_tags_only`, `+/-60d`, donut `3d`: `819` questions, hits `74`, pre/post = `9/65`
- `high_tags_only`, `+/-60d`, donut `7d`: `750` questions, hits `66`, pre/post = `8/58`

## Best rows

- `first_answer_1d`, `+/-45d`, donut `0d`: coef `-0.2952`, `p = 0.0128`, hits `53`
- `accepted_30d`, `+/-60d`, donut `3d`: coef `-0.2393`, `p = 0.0212`, hits `74`
- `accepted_30d`, `+/-45d`, donut `3d`: coef `-0.2633`, `p = 0.0408`, hits `50`
- `accepted_30d`, `+/-60d`, donut `7d`: coef `-0.3049`, `p = 0.0409`, hits `66`
- `accepted_30d`, `+/-30d`, donut `3d`: coef `-0.3297`, `p = 0.0612`, hits `32`
- `accepted_30d`, `+/-60d`, donut `0d`: coef `-0.2233`, `p = 0.0698`, hits `77`
- `accepted_7d`, `+/-60d`, donut `7d`: coef `-0.2670`, `p = 0.0700`, hits `66`
- `first_answer_1d`, `+/-60d`, donut `0d`: coef `-0.2059`, `p = 0.0731`, hits `77`

## Safe read

This stricter version is cleaner than the earlier question-side prototype because it relies on the posthistory direct-AI build's strict question-side disclosure flag.
The strongest pattern is no longer about immediate answer arrival. It is about accepted-window outcomes.
In high-exposure tags, `accepted_30d` is negative in every promoted window, and it is conventionally significant in the `+/-45d`, donut `3d`; `+/-60d`, donut `3d`; and `+/-60d`, donut `7d` rows.
By contrast, `first_answer_1d` is mostly null once the donut is applied. That is exactly the direction the paper wants: the ban window looks more like weakened certification than reduced immediate answer arrival.

## Honest ceiling

This still does not become a pristine discontinuity.
What it does provide is a cleaner and more coherent restricted timing result than the broader disclosed-hit build.
If the paper keeps one AI-ban timing layer, this is now the best reviewer-facing version.
