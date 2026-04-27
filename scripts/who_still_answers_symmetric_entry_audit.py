from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import build_who_still_answers_analysis as mod


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
PAPER_DIR = BASE_DIR / "paper"

TAG_MONTH_PANEL_CSV = PROCESSED_DIR / "who_still_answers_tag_month_entry_panel.csv"

MODEL_RESULTS_CSV = PROCESSED_DIR / "who_still_answers_symmetric_entry_model_results.csv"
MODEL_RESULTS_JSON = PROCESSED_DIR / "who_still_answers_symmetric_entry_results.json"
PLACEBO_CSV = PROCESSED_DIR / "who_still_answers_symmetric_entry_placebo_grid.csv"
LEAVE_TWO_OUT_CSV = PROCESSED_DIR / "who_still_answers_symmetric_entry_leave_two_out.csv"
SMALL_SAMPLE_CSV = PROCESSED_DIR / "who_still_answers_symmetric_entry_small_sample_inference.csv"
RANDOMIZATION_CSV = PROCESSED_DIR / "who_still_answers_symmetric_entry_randomization.csv"
IDENTIFICATION_CSV = PROCESSED_DIR / "who_still_answers_symmetric_entry_identification_profile.csv"
TREND_BREAK_CSV = PROCESSED_DIR / "who_still_answers_symmetric_entry_trend_break_results.csv"
EVENT_STUDY_CSV = PROCESSED_DIR / "who_still_answers_symmetric_entry_event_study.csv"
SUMMARY_MD = PAPER_DIR / "who_still_answers_symmetric_entry_audit.md"


def configure_output_paths() -> None:
    mod.MODEL_RESULTS_CSV = MODEL_RESULTS_CSV
    mod.MODEL_RESULTS_JSON = MODEL_RESULTS_JSON
    mod.PLACEBO_CSV = PLACEBO_CSV
    mod.LEAVE_TWO_OUT_CSV = LEAVE_TWO_OUT_CSV
    mod.SMALL_SAMPLE_INFERENCE_CSV = SMALL_SAMPLE_CSV
    mod.EXPOSURE_RANDOMIZATION_CSV = RANDOMIZATION_CSV
    mod.IDENTIFICATION_PROFILE_CSV = IDENTIFICATION_CSV
    mod.TREND_BREAK_RESULTS_CSV = TREND_BREAK_CSV


def build_spec(tag_month_panel: pd.DataFrame) -> mod.ModelSpec:
    frame = tag_month_panel.dropna(subset=["novice_entry_share"]).copy()
    frame = frame.loc[frame["n_new_answerers"].fillna(0) > 0].copy()
    return mod.ModelSpec(
        name="novice_entry_share",
        frame=frame,
        outcome="novice_entry_share",
        weight_col="n_new_answerers",
        term="exposure_post",
        formula="novice_entry_share ~ exposure_post + C(primary_tag) + C(month_id)",
        cluster_col="primary_tag",
    )


def summarize_outputs(
    fit_results: dict,
    small_sample: pd.DataFrame,
    identification: pd.DataFrame,
    leave_two_out: pd.DataFrame,
) -> str:
    summary = fit_results["novice_entry_share"]["summary"]
    small = small_sample.loc[small_sample["specification"] == "novice_entry_share"].iloc[0]
    trend = identification.loc[identification["specification"] == "novice_entry_share"].iloc[0]
    leave = leave_two_out.loc[leave_two_out["specification"] == "novice_entry_share"].copy()

    lines = [
        "# Symmetric Entrant Audit",
        "",
        "This audit reuses the updated `who_still_answers_tag_month_entry_panel.csv` after redefining the headline entrant outcome symmetrically across all months.",
        "",
        "## Baseline Entrant Result",
        "",
        f"- coefficient: `{summary['coef']:.6f}`",
        f"- clustered p-value: `{summary['pval']:.6f}`",
        "",
        "## Conservative Inference",
        "",
        f"- CR2 p-value: `{small['cr2_pval']:.6f}`",
        f"- wild bootstrap p-value: `{small['wild_cluster_bootstrap_pval']:.6f}`",
        f"- randomization p-value: `{small['randomization_pval']:.6f}`",
        "",
        "## Timing Read",
        "",
        f"- 2022-12 slope-break coefficient: `{trend['actual_coef']:.6f}`",
        f"- 2022-12 slope-break p-value: `{trend['actual_pval']:.6f}`",
        f"- 2022-12 rank versus pre-break candidates: `{int(trend['actual_rank_vs_pre_breaks'])} / {int(trend['n_pre_breaks'])}`",
        f"- share significant pre-break slopes: `{trend['share_significant_pre_breaks']:.6f}`",
        "",
        "## Leave-Two-Out",
        "",
        f"- coefficient minimum: `{leave['coef'].min():.6f}`",
        f"- coefficient median: `{leave['coef'].median():.6f}`",
        f"- coefficient maximum: `{leave['coef'].max():.6f}`",
        f"- positive share: `{(leave['coef'] > 0).mean():.6f}`",
        f"- p < 0.05 share: `{(leave['pval'] < 0.05).mean():.6f}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    configure_output_paths()
    tag_month_panel = pd.read_csv(TAG_MONTH_PANEL_CSV)
    spec = build_spec(tag_month_panel)

    fit_results = mod.fit_models([spec])
    mod.event_study(spec.frame, spec.outcome, spec.weight_col, EVENT_STUDY_CSV)
    placebo = mod.placebo_grid(tag_month_panel)
    leave_two = mod.leave_two_out([spec], {"novice_entry_share"})
    _, identification = mod.trend_break_diagnostics(tag_month_panel)
    small_sample, randomization = mod.small_sample_inference([spec], {"novice_entry_share"})

    SUMMARY_MD.write_text(
        summarize_outputs(fit_results, small_sample, identification, leave_two),
        encoding="utf-8",
    )

    print(SUMMARY_MD)
    print(MODEL_RESULTS_CSV)
    print(SMALL_SAMPLE_CSV)


if __name__ == "__main__":
    main()
