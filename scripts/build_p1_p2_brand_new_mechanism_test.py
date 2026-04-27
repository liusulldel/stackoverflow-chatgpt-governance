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

HARMONIZED_PANEL_CSV = PROCESSED_DIR / "p1_p2_harmonized_tag_month_panel.csv"
ENTRANT_PROFILES_CSV = PROCESSED_DIR / "who_still_answers_entrant_profiles.csv"

SUBTYPE_PANEL_CSV = PROCESSED_DIR / "p1_p2_harmonized_subtype_panel.csv"
RESULTS_CSV = PROCESSED_DIR / "p1_p2_brand_new_mechanism_results.csv"
RESULTS_JSON = PROCESSED_DIR / "p1_p2_brand_new_mechanism_results.json"
SUMMARY_MD = PAPER_DIR / "p1_p2_brand_new_mechanism_readout.md"


def fit_weighted(formula: str, data: pd.DataFrame, weight_col: str):
    frame = data.loc[data[weight_col].fillna(0) > 0].copy()
    y, x = patsy.dmatrices(formula, data=frame, return_type="dataframe", NA_action="drop")
    fit_frame = frame.loc[y.index].copy()
    weights = fit_frame[weight_col].astype(float)
    groups = fit_frame["primary_tag"]
    return sm.WLS(y, x, weights=weights).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True, "df_correction": True},
    )


def build_subtype_panel() -> pd.DataFrame:
    panel = pd.read_csv(HARMONIZED_PANEL_CSV)
    entrant_profiles = pd.read_csv(ENTRANT_PROFILES_CSV)
    subtype_monthly = (
        entrant_profiles.groupby(["primary_tag", "entry_month"], as_index=False)
        .agg(
            n_new_answerers_profiles=("answerer_user_id", "size"),
            brand_new_platform_count=("entrant_type", lambda s: int((s == "brand_new_platform").sum())),
            low_tenure_existing_count=("entrant_type", lambda s: int((s == "low_tenure_existing").sum())),
            established_cross_tag_count=("entrant_type", lambda s: int((s == "established_cross_tag").sum())),
        )
        .rename(columns={"entry_month": "month_id"})
    )
    merged = panel.merge(subtype_monthly, on=["primary_tag", "month_id"], how="left")
    for col in [
        "n_new_answerers_profiles",
        "brand_new_platform_count",
        "low_tenure_existing_count",
        "established_cross_tag_count",
    ]:
        merged[col] = merged[col].fillna(0.0)

    denom = merged["n_new_answerers_profiles"].where(merged["n_new_answerers_profiles"] > 0, np.nan)
    merged["brand_new_platform_share"] = merged["brand_new_platform_count"] / denom
    merged["low_tenure_existing_share"] = merged["low_tenure_existing_count"] / denom
    merged["established_cross_tag_share"] = merged["established_cross_tag_count"] / denom
    return merged


def build_specs() -> list[dict]:
    controls = " + C(primary_tag):time_index + C(primary_tag) + C(month_id)"
    rows: list[dict] = []
    base_outcomes = [
        ("first_answer_1d_rate", "first_answer_1d_denom"),
        ("accepted_30d_rate", "accepted_30d_denom"),
    ]
    for family, term in [("binary", "high_post"), ("continuous", "exposure_post")]:
        for outcome, weight_col in base_outcomes:
            rows.append(
                {
                    "family": family,
                    "model": "baseline",
                    "outcome": outcome,
                    "weight_col": weight_col,
                    "formula": f"{outcome} ~ {term}" + controls,
                    "headline_term": term,
                    "focus_terms": [term],
                }
            )
            rows.append(
                {
                    "family": family,
                    "model": "brand_new_only",
                    "outcome": outcome,
                    "weight_col": weight_col,
                    "formula": f"{outcome} ~ {term} + brand_new_platform_share" + controls,
                    "headline_term": term,
                    "focus_terms": [term, "brand_new_platform_share"],
                }
            )
            rows.append(
                {
                    "family": family,
                    "model": "composition_two_share",
                    "outcome": outcome,
                    "weight_col": weight_col,
                    "formula": f"{outcome} ~ {term} + brand_new_platform_share + low_tenure_existing_share" + controls,
                    "headline_term": term,
                    "focus_terms": [term, "brand_new_platform_share", "low_tenure_existing_share"],
                }
            )
    return rows


def run_specs(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in build_specs():
        fit = fit_weighted(spec["formula"], panel, spec["weight_col"])
        record = {
            "family": spec["family"],
            "model": spec["model"],
            "outcome": spec["outcome"],
            "weight_col": spec["weight_col"],
            "formula": spec["formula"],
            "nobs": int(fit.nobs),
        }
        for term in spec["focus_terms"]:
            record[f"{term}_coef"] = float(fit.params.get(term, np.nan))
            record[f"{term}_se"] = float(fit.bse.get(term, np.nan))
            record[f"{term}_pval"] = float(fit.pvalues.get(term, np.nan))
        rows.append(record)
    return pd.DataFrame(rows)


def write_summary(results: pd.DataFrame) -> None:
    def pick(family: str, model: str, outcome: str) -> pd.Series:
        return results.loc[
            (results["family"] == family)
            & (results["model"] == model)
            & (results["outcome"] == outcome)
        ].iloc[0]

    binary_base = pick("binary", "baseline", "first_answer_1d_rate")
    binary_brand = pick("binary", "brand_new_only", "first_answer_1d_rate")
    binary_comp = pick("binary", "composition_two_share", "first_answer_1d_rate")
    binary_accept = pick("binary", "brand_new_only", "accepted_30d_rate")

    cont_base = pick("continuous", "baseline", "first_answer_1d_rate")
    cont_brand = pick("continuous", "brand_new_only", "first_answer_1d_rate")
    cont_comp = pick("continuous", "composition_two_share", "first_answer_1d_rate")
    cont_accept = pick("continuous", "brand_new_only", "accepted_30d_rate")

    lines = [
        "# Brand-New Entrant Mechanism Test",
        "",
        "## Unified Setup",
        "",
        "- Common panel: `p1_p2_harmonized_tag_month_panel.csv` plus subtype shares built from `who_still_answers_entrant_profiles.csv`.",
        "- Mechanism probe: does `brand_new_platform_share` help explain `first_answer_1d_rate` under the unified design?",
        "- Harder composition probe: include both `brand_new_platform_share` and `low_tenure_existing_share`, leaving `established_cross_tag_share` as the omitted reference share.",
        "",
        "## Binary Read",
        "",
        f"- baseline `first_answer_1d high_post`: `{binary_base['high_post_coef']:.6f}` (p `{binary_base['high_post_pval']:.6f}`)",
        f"- add `brand_new_platform_share`: `high_post = {binary_brand['high_post_coef']:.6f}` (p `{binary_brand['high_post_pval']:.6f}`), `brand_new_platform_share = {binary_brand['brand_new_platform_share_coef']:.6f}` (p `{binary_brand['brand_new_platform_share_pval']:.6f}`)",
        f"- add two-share composition: `high_post = {binary_comp['high_post_coef']:.6f}` (p `{binary_comp['high_post_pval']:.6f}`), `brand_new_platform_share = {binary_comp['brand_new_platform_share_coef']:.6f}` (p `{binary_comp['brand_new_platform_share_pval']:.6f}`), `low_tenure_existing_share = {binary_comp['low_tenure_existing_share_coef']:.6f}` (p `{binary_comp['low_tenure_existing_share_pval']:.6f}`)",
        f"- `accepted_30d` with `brand_new_platform_share`: `high_post = {binary_accept['high_post_coef']:.6f}` (p `{binary_accept['high_post_pval']:.6f}`), `brand_new_platform_share = {binary_accept['brand_new_platform_share_coef']:.6f}` (p `{binary_accept['brand_new_platform_share_pval']:.6f}`)",
        "",
        "## Continuous Read",
        "",
        f"- baseline `first_answer_1d exposure_post`: `{cont_base['exposure_post_coef']:.6f}` (p `{cont_base['exposure_post_pval']:.6f}`)",
        f"- add `brand_new_platform_share`: `exposure_post = {cont_brand['exposure_post_coef']:.6f}` (p `{cont_brand['exposure_post_pval']:.6f}`), `brand_new_platform_share = {cont_brand['brand_new_platform_share_coef']:.6f}` (p `{cont_brand['brand_new_platform_share_pval']:.6f}`)",
        f"- add two-share composition: `exposure_post = {cont_comp['exposure_post_coef']:.6f}` (p `{cont_comp['exposure_post_pval']:.6f}`), `brand_new_platform_share = {cont_comp['brand_new_platform_share_coef']:.6f}` (p `{cont_comp['brand_new_platform_share_pval']:.6f}`), `low_tenure_existing_share = {cont_comp['low_tenure_existing_share_coef']:.6f}` (p `{cont_comp['low_tenure_existing_share_pval']:.6f}`)",
        f"- `accepted_30d` with `brand_new_platform_share`: `exposure_post = {cont_accept['exposure_post_coef']:.6f}` (p `{cont_accept['exposure_post_pval']:.6f}`), `brand_new_platform_share = {cont_accept['brand_new_platform_share_coef']:.6f}` (p `{cont_accept['brand_new_platform_share_pval']:.6f}`)",
        "",
        "## Read",
        "",
        "- A convincing mechanism bridge would normally show both: rising brand-new-platform share in exposed domains, and a negative conditional association between that share and first-answer speed that materially attenuates the exposure term.",
        "- This output is a hard probe, not a causal mediation design.",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    panel = build_subtype_panel()
    panel.to_csv(SUBTYPE_PANEL_CSV, index=False)
    results = run_specs(panel)
    results.to_csv(RESULTS_CSV, index=False)
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "n_rows": int(len(panel)),
                "results": results.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(results)
    print(SUBTYPE_PANEL_CSV)
    print(RESULTS_CSV)
    print(SUMMARY_MD)


if __name__ == "__main__":
    main()
