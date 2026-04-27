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

ENTRY_PANEL_CSV = PROCESSED_DIR / "who_still_answers_tag_month_entry_panel.csv"
CLOSURE_PANEL_CSV = PROCESSED_DIR / "closure_ladder_primary_panel.csv"
ENTRANT_PROFILES_CSV = PROCESSED_DIR / "who_still_answers_entrant_profiles.csv"

HARMONIZED_PANEL_CSV = PROCESSED_DIR / "p1_p2_harmonized_tag_month_panel.csv"
RESULTS_CSV = PROCESSED_DIR / "p1_p2_harmonized_chain_results.csv"
RESULTS_JSON = PROCESSED_DIR / "p1_p2_harmonized_chain_results.json"
SUMMARY_MD = PAPER_DIR / "p1_p2_harmonization_readout.md"
SUBTYPE_RESULTS_CSV = PROCESSED_DIR / "p1_p2_harmonized_subtype_results.csv"
SUBTYPE_SUMMARY_MD = PAPER_DIR / "p1_p2_subtype_harmonization_readout.md"


def load_and_merge() -> pd.DataFrame:
    entry = pd.read_csv(ENTRY_PANEL_CSV)
    closure = pd.read_csv(CLOSURE_PANEL_CSV).rename(columns={"tag": "primary_tag"})

    merged = entry.merge(
        closure[
            [
                "primary_tag",
                "month_id",
                "n_questions",
                "high_tag",
                "exposure_index",
                "post_chatgpt",
                "time_index",
                "first_answer_1d_rate",
                "first_answer_1d_denom",
                "accepted_vote_30d_rate",
                "accepted_vote_30d_denom",
            ]
        ].rename(
            columns={
                "n_questions": "n_questions_closure",
                "high_tag": "high_tag_closure",
                "exposure_index": "exposure_index_closure",
                "post_chatgpt": "post_chatgpt_closure",
                "time_index": "time_index_closure",
                "first_answer_1d_rate": "first_answer_1d_rate_closure",
                "first_answer_1d_denom": "first_answer_1d_denom_closure",
            }
        ),
        on=["primary_tag", "month_id"],
        how="inner",
    )

    checks = {
        "n_questions": np.allclose(merged["n_questions"], merged["n_questions_closure"]),
        "exposure_index": np.allclose(merged["exposure_index"], merged["exposure_index_closure"]),
        "post_chatgpt": (merged["post_chatgpt"] == merged["post_chatgpt_closure"]).all(),
        "first_answer_1d_rate": np.allclose(
            merged["first_answer_1d_rate"],
            merged["first_answer_1d_rate_closure"],
            atol=2e-3,
        ),
        "first_answer_1d_denom": np.allclose(
            merged["first_answer_1d_denom"],
            merged["first_answer_1d_denom_closure"],
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Panel harmonization checks failed for: {failed}")

    # For the first harmonization run, explicitly adopt the P2 closure-ladder
    # binary threshold and trend index so the stage outcomes are not silently
    # redefined while we test the mechanism chain.
    merged["high_tag_p1_median"] = merged["high_tag"]
    merged["high_tag"] = merged["high_tag_closure"]
    merged["time_index_p1"] = merged["time_index"]
    merged["time_index"] = merged["time_index_closure"].astype(float)
    merged["first_answer_1d_rate"] = merged["first_answer_1d_rate_closure"]
    merged["first_answer_1d_denom"] = merged["first_answer_1d_denom_closure"]
    merged["high_post"] = merged["high_tag"] * merged["post_chatgpt"]
    return merged.sort_values(["primary_tag", "month_id"]).reset_index(drop=True)


def fit_weighted(formula: str, data: pd.DataFrame, weight_col: str):
    frame = data.loc[data[weight_col].fillna(0) > 0].copy()
    y, x = patsy.dmatrices(formula, data=frame, return_type="dataframe", NA_action="drop")
    fit_frame = frame.loc[y.index].copy()
    groups = fit_frame["primary_tag"]
    weights = fit_frame[weight_col].astype(float)
    return sm.WLS(y, x, weights=weights).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True, "df_correction": True},
    )


def build_specs(panel: pd.DataFrame) -> list[dict]:
    same_trend_tail = " + C(primary_tag):time_index + C(primary_tag) + C(month_id)"
    specs = [
        {
            "family": "binary",
            "outcome": "novice_entry_share",
            "weight_col": "n_new_answerers",
            "term": "high_post",
            "formula": "novice_entry_share ~ high_post" + same_trend_tail,
        },
        {
            "family": "binary",
            "outcome": "first_answer_1d_rate",
            "weight_col": "first_answer_1d_denom",
            "term": "high_post",
            "formula": "first_answer_1d_rate ~ high_post" + same_trend_tail,
        },
        {
            "family": "binary",
            "outcome": "accepted_30d_rate",
            "weight_col": "accepted_30d_denom",
            "term": "high_post",
            "formula": "accepted_30d_rate ~ high_post" + same_trend_tail,
        },
        {
            "family": "continuous",
            "outcome": "novice_entry_share",
            "weight_col": "n_new_answerers",
            "term": "exposure_post",
            "formula": "novice_entry_share ~ exposure_post" + same_trend_tail,
        },
        {
            "family": "continuous",
            "outcome": "first_answer_1d_rate",
            "weight_col": "first_answer_1d_denom",
            "term": "exposure_post",
            "formula": "first_answer_1d_rate ~ exposure_post" + same_trend_tail,
        },
        {
            "family": "continuous",
            "outcome": "accepted_30d_rate",
            "weight_col": "accepted_30d_denom",
            "term": "exposure_post",
            "formula": "accepted_30d_rate ~ exposure_post" + same_trend_tail,
        },
    ]
    return specs


def build_subtype_panel(panel: pd.DataFrame) -> pd.DataFrame:
    entrant_profiles = pd.read_csv(ENTRANT_PROFILES_CSV)
    subtype_monthly = (
        entrant_profiles.groupby(["primary_tag", "entry_month"], as_index=False)
        .agg(
            n_new_answerers=("answerer_user_id", "size"),
            brand_new_platform_count=("entrant_type", lambda s: int((s == "brand_new_platform").sum())),
            low_tenure_existing_count=("entrant_type", lambda s: int((s == "low_tenure_existing").sum())),
            established_cross_tag_count=("entrant_type", lambda s: int((s == "established_cross_tag").sum())),
        )
        .rename(columns={"entry_month": "month_id"})
    )
    merged = panel.merge(
        subtype_monthly,
        on=["primary_tag", "month_id"],
        how="left",
        suffixes=("", "_profiles"),
    )
    for col in [
        "n_new_answerers_profiles",
        "brand_new_platform_count",
        "low_tenure_existing_count",
        "established_cross_tag_count",
    ]:
        merged[col] = merged[col].fillna(0.0)

    subtype_denominator = merged["n_new_answerers_profiles"].where(
        merged["n_new_answerers_profiles"] > 0,
        np.nan,
    )
    merged["brand_new_platform_share"] = merged["brand_new_platform_count"] / subtype_denominator
    merged["low_tenure_existing_share"] = merged["low_tenure_existing_count"] / subtype_denominator
    merged["established_cross_tag_share"] = merged["established_cross_tag_count"] / subtype_denominator
    return merged


def run_specs(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in build_specs(panel):
        fit = fit_weighted(spec["formula"], panel, spec["weight_col"])
        coef = float(fit.params.get(spec["term"], np.nan))
        se = float(fit.bse.get(spec["term"], np.nan))
        pval = float(fit.pvalues.get(spec["term"], np.nan))
        rows.append(
            {
                "family": spec["family"],
                "outcome": spec["outcome"],
                "term": spec["term"],
                "coef": coef,
                "se": se,
                "pval": pval,
                "nobs": int(fit.nobs),
                "weight_col": spec["weight_col"],
                "formula": spec["formula"],
            }
        )
    return pd.DataFrame(rows)


def run_subtype_specs(panel: pd.DataFrame) -> pd.DataFrame:
    same_trend_tail = " + C(primary_tag):time_index + C(primary_tag) + C(month_id)"
    rows: list[dict] = []
    subtype_outcomes = [
        "brand_new_platform_share",
        "low_tenure_existing_share",
        "established_cross_tag_share",
    ]
    for family, term in [("binary", "high_post"), ("continuous", "exposure_post")]:
        for outcome in subtype_outcomes:
            formula = f"{outcome} ~ {term}" + same_trend_tail
            fit = fit_weighted(formula, panel, "n_new_answerers_profiles")
            rows.append(
                {
                    "family": family,
                    "outcome": outcome,
                    "term": term,
                    "coef": float(fit.params.get(term, np.nan)),
                    "se": float(fit.bse.get(term, np.nan)),
                    "pval": float(fit.pvalues.get(term, np.nan)),
                    "nobs": int(fit.nobs),
                    "weight_col": "n_new_answerers_profiles",
                    "formula": formula,
                }
            )
    return pd.DataFrame(rows)


def write_summary(panel: pd.DataFrame, results: pd.DataFrame) -> None:
    entry_binary = results.loc[(results["family"] == "binary") & (results["outcome"] == "novice_entry_share")].iloc[0]
    speed_binary = results.loc[(results["family"] == "binary") & (results["outcome"] == "first_answer_1d_rate")].iloc[0]
    accept_binary = results.loc[(results["family"] == "binary") & (results["outcome"] == "accepted_30d_rate")].iloc[0]
    entry_cont = results.loc[(results["family"] == "continuous") & (results["outcome"] == "novice_entry_share")].iloc[0]
    speed_cont = results.loc[(results["family"] == "continuous") & (results["outcome"] == "first_answer_1d_rate")].iloc[0]
    accept_cont = results.loc[(results["family"] == "continuous") & (results["outcome"] == "accepted_30d_rate")].iloc[0]

    lines = [
        "# P1/P2 Harmonization Readout",
        "",
        "## Unified Design",
        "",
        "- Common sample: exact tag-month intersection of `who_still_answers_tag_month_entry_panel.csv` and `closure_ladder_primary_panel.csv`.",
        f"- Rows: `{len(panel)}` tag-month observations.",
        f"- Tags: `{panel['primary_tag'].nunique()}`.",
        f"- Months: `{panel['month_id'].nunique()}`.",
        "- Entrant definition: symmetric low-tenure entrant share (`novice_entry_share`) from the updated P1 panel.",
        "- Trend controls: identical across all outcomes via `C(primary_tag):time_index + C(primary_tag) + C(month_id)`.",
        "",
        "## Binary Harmonization",
        "",
        f"- entrant composition (`high_post`): coef `{entry_binary['coef']:.6f}`, p `{entry_binary['pval']:.6f}`",
        f"- first_answer_1d (`high_post`): coef `{speed_binary['coef']:.6f}`, p `{speed_binary['pval']:.6f}`",
        f"- accepted_30d (`high_post`): coef `{accept_binary['coef']:.6f}`, p `{accept_binary['pval']:.6f}`",
        "",
        "## Continuous Harmonization",
        "",
        f"- entrant composition (`exposure_post`): coef `{entry_cont['coef']:.6f}`, p `{entry_cont['pval']:.6f}`",
        f"- first_answer_1d (`exposure_post`): coef `{speed_cont['coef']:.6f}`, p `{speed_cont['pval']:.6f}`",
        f"- accepted_30d (`exposure_post`): coef `{accept_cont['coef']:.6f}`, p `{accept_cont['pval']:.6f}`",
        "",
        "## Read",
        "",
        "- This table isolates whether the cross-paper conflict is being driven mainly by exposure family once sample, entrant construction, and trend controls are harmonized.",
        "- It does not yet settle the final journal narrative; it is a design-adjudication run.",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_subtype_summary(panel: pd.DataFrame, results: pd.DataFrame) -> None:
    lines = [
        "# P1/P2 Subtype Harmonization Readout",
        "",
        "## Unified Design",
        "",
        "- Common sample: exact tag-month intersection used in the first harmonization run.",
        "- Entrant denominator: all first-time focal-tag answerers in a tag-month from `who_still_answers_entrant_profiles.csv`.",
        "- Trend controls: identical across all subtype outcomes via `C(primary_tag):time_index + C(primary_tag) + C(month_id)`.",
        "",
        "## Results",
        "",
        results.to_markdown(index=False),
        "",
        "## Read",
        "",
        "- This table tests whether the failed aggregate entrant bridge is hiding a subtype-specific bridge.",
        "- A positive `brand_new_platform_share` under the unified spec would keep a narrower mechanism path alive.",
    ]
    SUBTYPE_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    panel = load_and_merge()
    panel.to_csv(HARMONIZED_PANEL_CSV, index=False)
    results = run_specs(panel)
    subtype_panel = build_subtype_panel(panel)
    subtype_results = run_subtype_specs(subtype_panel)
    results.to_csv(RESULTS_CSV, index=False)
    subtype_results.to_csv(SUBTYPE_RESULTS_CSV, index=False)
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "n_rows": int(len(panel)),
                "n_tags": int(panel["primary_tag"].nunique()),
                "n_months": int(panel["month_id"].nunique()),
                "results": results.to_dict(orient="records"),
                "subtype_results": subtype_results.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(panel, results)
    write_subtype_summary(subtype_panel, subtype_results)
    print(HARMONIZED_PANEL_CSV)
    print(RESULTS_CSV)
    print(SUBTYPE_RESULTS_CSV)
    print(SUMMARY_MD)
    print(SUBTYPE_SUMMARY_MD)


if __name__ == "__main__":
    main()
