from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
PAPER_DIR = BASE_DIR / "paper"

TAG_MONTH_ENTRY_PANEL_CSV = PROCESSED_DIR / "who_still_answers_tag_month_entry_panel.csv"
QUESTION_CLOSURE_PANEL_PARQUET = PROCESSED_DIR / "who_still_answers_question_closure_panel.parquet"

OUTPUT_CSV = PROCESSED_DIR / "harmonized_integration_model_results.csv"
OUTPUT_JSON = PROCESSED_DIR / "harmonized_integration_results.json"
OUTPUT_MD = PAPER_DIR / "harmonized_integration_readout.md"

EXPOSURE_FAMILIES = {
    "continuous": "exposure_post",
    "binary": "high_post",
}

OUTCOME_SPECS = {
    "entrant_composition": ("novice_entry_share", "n_new_answerers"),
    "first_answer_1d": ("first_answer_1d_rate", "first_answer_1d_denom"),
    "accepted_30d": ("accepted_30d_rate", "accepted_30d_denom"),
}


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(TAG_MONTH_ENTRY_PANEL_CSV)
    panel["month_start"] = pd.to_datetime(panel["month_id"] + "-01", utc=True)
    panel["post_chatgpt"] = panel["post_chatgpt"].astype(int)
    panel["high_tag"] = panel["high_tag"].astype(int)
    panel["high_post"] = panel["high_tag"] * panel["post_chatgpt"]

    # Rebuild a clean, monotone time index so every spec uses the same trend control.
    ordered_months = sorted(panel["month_id"].unique())
    month_index = {month_id: idx + 1 for idx, month_id in enumerate(ordered_months)}
    panel["time_index_harmonized"] = panel["month_id"].map(month_index).astype(int)
    return panel


def fit_clustered_wls(formula: str, frame: pd.DataFrame, weight_col: str):
    sample = frame.loc[frame[weight_col].fillna(0) > 0].copy()
    y, x = patsy.dmatrices(formula, sample, return_type="dataframe", NA_action="drop")
    sample = sample.loc[y.index].copy()
    weights = sample[weight_col].astype(float)
    model = sm.WLS(y, x, weights=weights).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample["primary_tag"], "use_correction": True, "df_correction": True},
    )
    return model, sample


def build_formula(outcome: str, exposure_term: str) -> str:
    return (
        f"{outcome} ~ {exposure_term} + C(primary_tag):time_index_harmonized + "
        "C(primary_tag) + C(month_id)"
    )


def summarize_sample(question_closure: pd.DataFrame, panel: pd.DataFrame) -> dict:
    primary_questions = question_closure.loc[question_closure["primary_tag"].notna()].copy()
    return {
        "n_question_level_rows": int(len(primary_questions)),
        "n_tag_month_rows": int(len(panel)),
        "min_question_created_at": str(primary_questions["question_created_at"].min()),
        "max_question_created_at": str(primary_questions["question_created_at"].max()),
        "n_tags": int(primary_questions["primary_tag"].nunique()),
        "construct_stable_entrant_note": (
            "entrant_composition uses the symmetric low-tenure entrant share already stored in "
            "who_still_answers_tag_month_entry_panel.csv"
        ),
    }


def write_readout(model_df: pd.DataFrame, sample_summary: dict) -> None:
    lines = [
        "# Harmonized Integration Readout",
        "",
        "This readout aligns the merged-paper test to one shared tag-month universe, one shared trend-control structure, and the construct-stable entrant definition from the symmetric entrant audit.",
        "",
        "## Common Sample",
        "",
        f"- Question-level rows: `{sample_summary['n_question_level_rows']}`",
        f"- Tag-month rows: `{sample_summary['n_tag_month_rows']}`",
        f"- Tags: `{sample_summary['n_tags']}`",
        f"- Window start: `{sample_summary['min_question_created_at']}`",
        f"- Window end: `{sample_summary['max_question_created_at']}`",
        "",
        "## Specification",
        "",
        "- Shared unit: `tag-month`",
        "- Shared trend control: `C(primary_tag):time_index_harmonized + C(primary_tag) + C(month_id)`",
        "- Shared entrant construct: symmetric low-tenure entrant share (`novice_entry_share`)",
        "- Exposure families shown side by side: `continuous` and `binary`",
        "",
        "## Main Table",
        "",
        model_df[
            [
                "exposure_family",
                "outcome_family",
                "outcome",
                "weight_col",
                "term",
                "coef",
                "se",
                "pval",
                "nobs",
                "formula",
            ]
        ].to_markdown(index=False),
        "",
        "## Read",
        "",
        "- Use this table to judge whether the merger can survive one common empirical skeleton.",
        "- If entrant composition moves only under one exposure family, the merged paper must say that explicitly.",
        "- If `first_answer_1d` and `accepted_30d` continue to diverge under the same skeleton, the staged-resolution story survives harmonization better than the labor-reallocation story.",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    panel = load_panel()
    question_closure = pd.read_parquet(QUESTION_CLOSURE_PANEL_PARQUET)
    sample_summary = summarize_sample(question_closure, panel)

    rows: list[dict] = []
    for exposure_family, exposure_term in EXPOSURE_FAMILIES.items():
        for outcome_family, (outcome, weight_col) in OUTCOME_SPECS.items():
            frame = panel.dropna(subset=[outcome]).copy()
            formula = build_formula(outcome, exposure_term)
            model, sample = fit_clustered_wls(formula, frame, weight_col)
            rows.append(
                {
                    "exposure_family": exposure_family,
                    "outcome_family": outcome_family,
                    "outcome": outcome,
                    "weight_col": weight_col,
                    "term": exposure_term,
                    "coef": float(model.params.get(exposure_term, np.nan)),
                    "se": float(model.bse.get(exposure_term, np.nan)),
                    "pval": float(model.pvalues.get(exposure_term, np.nan)),
                    "nobs": int(model.nobs),
                    "formula": formula,
                    "mean_outcome": float(sample[outcome].mean()),
                }
            )

    model_df = pd.DataFrame(rows)
    model_df.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "sample_summary": sample_summary,
                "model_results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_readout(model_df, sample_summary)

    print(OUTPUT_CSV)
    print(OUTPUT_JSON)
    print(OUTPUT_MD)


if __name__ == "__main__":
    main()
