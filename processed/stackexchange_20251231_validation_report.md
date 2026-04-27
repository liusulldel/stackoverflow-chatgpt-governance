# stackexchange_20251231 Validation Report

## Metadata
- `n_questions`: `2322009`
- `n_answers`: `2750564`
- `n_accept_votes`: `1013245`
- `n_question_level_rows`: `2322009`
- `n_primary_panel_rows`: `1147`
- `n_fractional_panel_rows`: `1152`
- `min_question_created_at`: `2020-01-01T00:01:28.670000+00:00`
- `max_question_created_at`: `2025-12-31T23:14:43.730000+00:00`
- `observation_cutoff_at`: `2025-12-31T23:14:43.730000+00:00`
- `n_primary_tags`: `16`

## Checks
- `PASS` manifest_has_summary: Manifest includes a non-null summary payload.
- `PASS` date_range_starts_in_2020: Question-level min date is 2020-01-01T00:01:28.670000+00:00.
- `PASS` date_range_reaches_2025: Question-level max date reaches 2025-12-31T23:14:43.730000+00:00.
- `PASS` question_ids_unique: Question-level data have unique question_id rows.
- `PASS` primary_panel_unique_tag_month: Primary panel has unique tag-month rows.
- `PASS` fractional_panel_unique_tag_month: Fractional panel has unique tag-month rows.
- `PASS` tag_set_matches_selected_tags: Observed tag set matches the 16 selected tags.
- `PASS` panel_summary_has_observation_cutoff: Panel summary includes observation_cutoff=2025-12-31T23:14:43.730000+00:00.
- `PASS` accepted_7d_le_accepted_30d_question_level: Question-level accepted_7d never exceeds accepted_30d.
- `PASS` first_answer_1d_le_first_answer_7d_question_level: Question-level first_answer_1d never exceeds first_answer_7d.
- `PASS` accepted_7d_le_accepted_30d_primary_panel: Primary panel preserves accepted_7d_rate <= accepted_30d_rate whenever the two rates share the same risk set.
- `PASS` first_answer_1d_le_first_answer_7d_primary_panel: Primary panel preserves first_answer_1d_rate <= first_answer_7d_rate whenever the two rates share the same risk set.
- `PASS` accepted_7d_le_accepted_30d_fractional_panel: Fractional panel preserves accepted_7d_rate <= accepted_30d_rate whenever the two rates share the same risk set.
- `PASS` first_answer_1d_le_first_answer_7d_fractional_panel: Fractional panel preserves first_answer_1d_rate <= first_answer_7d_rate whenever the two rates share the same risk set.
- `PASS` primary_panel_has_denominators: Primary panel includes denominator columns for eligibility-aware short-horizon rates.
- `PASS` fractional_panel_has_denominators: Fractional panel includes denominator columns for eligibility-aware short-horizon rates.
- `PASS` question_level_has_eligibility_flags: Question-level data include eligibility flags for all short-horizon outcomes.
- `PASS` eligibility_flags_respect_cutoff: Eligibility flags respect the observation cutoff for all short-horizon outcomes.
- `PASS` late_2025_primary_denominators_drop_below_total: Late-window censoring is visible in the primary panel denominators.
- `PASS` focal_summary_matches_question_rows: Focal summary n_questions matches parquet row count (2322009).
- `PASS` panel_summary_matches_question_level_rows: Panel summary n_question_level_rows matches parquet row count (2322009).
