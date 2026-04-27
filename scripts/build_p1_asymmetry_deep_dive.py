from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

SUBTYPE_CONSEQUENCE_PANEL_CSV = PROCESSED / "p1_jmis_subtype_consequence_panel.csv"
RESIDUAL_QUEUE_PANEL_CSV = PROCESSED / "p1_jmis_residual_queue_panel.csv"
VALIDATION_SAMPLE_PARQUET = PROCESSED / "entrant_first_question_validation_sample.parquet"
TAG_FAMILY_MAP_CSV = PROCESSED / "p1_tag_family_map.csv"

ESTABLISHED_INTERACTION_RESULTS_CSV = PROCESSED / "p1_established_interaction_results.csv"
BRAND_NEW_COMPLEXITY_GAPS_CSV = PROCESSED / "p1_brand_new_complexity_gaps.csv"
FAMILY_CONSEQUENCE_RESULTS_CSV = PROCESSED / "p1_family_consequence_results.csv"
ASYMMETRY_DEEP_DIVE_MEMO_MD = PAPER / "p1_asymmetry_deep_dive_memo.md"


def fit_wls(formula: str, data: pd.DataFrame, weight_col: str):
    frame = data.dropna(subset=[weight_col]).copy()
    base = smf.wls(formula, data=frame, weights=frame[weight_col], missing="drop").fit()
    used_groups = frame.loc[base.model.data.row_labels, "primary_tag"]
    robust = base.get_robustcov_results(
        cov_type="cluster",
        groups=used_groups,
    )
    names = base.model.exog_names
    return {
        "params": pd.Series(robust.params, index=names),
        "bse": pd.Series(robust.bse, index=names),
        "pvalues": pd.Series(robust.pvalues, index=names),
        "nobs": int(base.nobs),
    }


def build_established_interactions() -> pd.DataFrame:
    panel = pd.read_csv(SUBTYPE_CONSEQUENCE_PANEL_CSV)
    residual = pd.read_csv(RESIDUAL_QUEUE_PANEL_CSV)[
        ["primary_tag", "month_id", "residual_queue_complexity_index_mean"]
    ]
    panel = panel.merge(residual, on=["primary_tag", "month_id"], how="left")
    rows: list[dict[str, object]] = []
    for family, term in [("binary", "high_post"), ("continuous", "exposure_post")]:
        for outcome, weight_col in [
            ("any_answer_7d_rate", "any_answer_7d_denom"),
            ("first_positive_answer_latency_mean", "first_positive_answer_latency_denom"),
            ("accepted_vote_30d_rate", "accepted_vote_30d_denom"),
        ]:
            formula = (
                f"{outcome} ~ {term} + established_cross_tag_share + residual_queue_complexity_index_mean + "
                "established_cross_tag_share:residual_queue_complexity_index_mean + "
                "C(primary_tag):time_index + C(primary_tag) + C(month_id)"
            )
            model = fit_wls(formula, panel.dropna(subset=[outcome]), weight_col)
            for coef_term in [
                term,
                "established_cross_tag_share",
                "residual_queue_complexity_index_mean",
                "established_cross_tag_share:residual_queue_complexity_index_mean",
            ]:
                if coef_term not in model["params"].index:
                    continue
                rows.append(
                    {
                        "family": family,
                        "outcome": outcome,
                        "formula": formula,
                        "term": coef_term,
                        "coef": float(model["params"][coef_term]),
                        "se": float(model["bse"][coef_term]),
                        "pval": float(model["pvalues"][coef_term]),
                        "nobs": model["nobs"],
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(ESTABLISHED_INTERACTION_RESULTS_CSV, index=False)
    return out


def build_brand_new_complexity_gaps() -> pd.DataFrame:
    sample = pd.read_parquet(VALIDATION_SAMPLE_PARQUET)
    sample = sample[sample["entrant_type"].isin(["brand_new_platform", "established_cross_tag"])].copy()
    sample["is_brand_new"] = (sample["entrant_type"] == "brand_new_platform").astype(int)
    sample["complexity_tercile"] = pd.qcut(sample["direct_complexity_index"], 3, labels=["low", "mid", "high"])
    rows: list[dict[str, object]] = []
    for tercile, frame in sample.groupby("complexity_tercile"):
        for outcome in ["first_answer_1d", "score", "is_current_accepted_answer", "accepted_30d"]:
            model = smf.ols(f"{outcome} ~ is_brand_new + C(primary_tag) + C(entry_month)", data=frame).fit()
            rows.append(
                {
                    "complexity_tercile": str(tercile),
                    "outcome": outcome,
                    "coef_brand_new_vs_established": float(model.params.get("is_brand_new", np.nan)),
                    "se": float(model.bse.get("is_brand_new", np.nan)),
                    "pval": float(model.pvalues.get("is_brand_new", np.nan)),
                    "nobs": int(model.nobs),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(BRAND_NEW_COMPLEXITY_GAPS_CSV, index=False)
    return out


def build_family_consequence_results() -> pd.DataFrame:
    tag_map = pd.read_csv(TAG_FAMILY_MAP_CSV)
    panel = pd.read_csv(SUBTYPE_CONSEQUENCE_PANEL_CSV).merge(tag_map, on="primary_tag", how="left")
    rows: list[dict[str, object]] = []
    for tag_family, frame in panel.groupby("tag_family"):
        for family, term in [("binary", "high_post"), ("continuous", "exposure_post")]:
            for outcome, weight_col, subtype_term in [
                ("any_answer_7d_rate", "any_answer_7d_denom", "established_cross_tag_share"),
                ("first_positive_answer_latency_mean", "first_positive_answer_latency_denom", "established_cross_tag_share"),
                ("accepted_vote_30d_rate", "accepted_vote_30d_denom", "brand_new_platform_share"),
                ("first_answer_1d_rate_closure", "first_answer_1d_denom_closure", "brand_new_platform_share"),
            ]:
                model = fit_wls(
                    f"{outcome} ~ {term} + {subtype_term} + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
                    frame.dropna(subset=[outcome]),
                    weight_col,
                )
                for coef_term in [term, subtype_term]:
                    if coef_term not in model["params"].index:
                        continue
                    rows.append(
                        {
                            "tag_family": tag_family,
                            "family": family,
                            "outcome": outcome,
                            "term": coef_term,
                            "coef": float(model["params"][coef_term]),
                            "se": float(model["bse"][coef_term]),
                            "pval": float(model["pvalues"][coef_term]),
                            "nobs": model["nobs"],
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(FAMILY_CONSEQUENCE_RESULTS_CSV, index=False)
    return out


def write_memo(
    established_results: pd.DataFrame,
    brand_new_complexity: pd.DataFrame,
    family_results: pd.DataFrame,
) -> None:
    def pick(df: pd.DataFrame, **filters):
        temp = df.copy()
        for key, value in filters.items():
            temp = temp[temp[key] == value]
        if temp.empty:
            return None
        return temp.iloc[0]

    est_any_binary = pick(
        established_results,
        family="binary",
        outcome="any_answer_7d_rate",
        term="established_cross_tag_share:residual_queue_complexity_index_mean",
    )
    est_latency_binary = pick(
        established_results,
        family="binary",
        outcome="first_positive_answer_latency_mean",
        term="established_cross_tag_share:residual_queue_complexity_index_mean",
    )
    low_speed = pick(brand_new_complexity, complexity_tercile="low", outcome="first_answer_1d")
    mid_speed = pick(brand_new_complexity, complexity_tercile="mid", outcome="first_answer_1d")
    high_speed = pick(brand_new_complexity, complexity_tercile="high", outcome="first_answer_1d")
    low_accept = pick(brand_new_complexity, complexity_tercile="low", outcome="accepted_30d")
    mid_accept = pick(brand_new_complexity, complexity_tercile="mid", outcome="accepted_30d")
    high_accept = pick(brand_new_complexity, complexity_tercile="high", outcome="accepted_30d")

    fam_rows = family_results[
        ((family_results["outcome"] == "any_answer_7d_rate") & (family_results["term"] == "established_cross_tag_share"))
        | ((family_results["outcome"] == "first_answer_1d_rate_closure") & (family_results["term"] == "brand_new_platform_share"))
        | ((family_results["outcome"] == "accepted_vote_30d_rate") & (family_results["term"] == "brand_new_platform_share"))
    ].sort_values(["tag_family", "outcome", "family"])

    lines = [
        "# P1 Asymmetry Deep Dive",
        "",
        "## Established Cross-Tag",
        "",
        "The deeper interaction question is whether `established_cross_tag` becomes especially adverse when the residual queue is more complex.",
        "",
        f"- Binary interaction on `any_answer_7d_rate`: coef `{est_any_binary['coef']:.4f}`, p `{est_any_binary['pval']:.4g}`." if est_any_binary is not None else "- Binary any-answer interaction unavailable.",
        f"- Binary interaction on `first_positive_answer_latency_mean`: coef `{est_latency_binary['coef']:.4f}`, p `{est_latency_binary['pval']:.4g}`." if est_latency_binary is not None else "- Binary latency interaction unavailable.",
        "",
        "Read:",
        "",
        "- If these interactions are weak, the safer story remains fallback dependence rather than a complexity-triggered mechanism.",
        "- If they are strong, the established-cross-tag adverse role is most visible when the local queue becomes harder and local replenishment is thin.",
        "",
        "## Brand-New Asymmetry",
        "",
        "The sellable asymmetry is `response speed versus certification`, not `brand-new entrants cause slowdown`.",
        "",
        f"- Low-complexity tercile, brand-new vs established on `first_answer_1d`: `{low_speed['coef_brand_new_vs_established']:.4f}`, p `{low_speed['pval']:.4g}`." if low_speed is not None else "- Low-complexity speed gap unavailable.",
        f"- Mid-complexity tercile, brand-new vs established on `first_answer_1d`: `{mid_speed['coef_brand_new_vs_established']:.4f}`, p `{mid_speed['pval']:.4g}`." if mid_speed is not None else "- Mid-complexity speed gap unavailable.",
        f"- High-complexity tercile, brand-new vs established on `first_answer_1d`: `{high_speed['coef_brand_new_vs_established']:.4f}`, p `{high_speed['pval']:.4g}`." if high_speed is not None else "- High-complexity speed gap unavailable.",
        f"- Low-complexity tercile, brand-new vs established on `accepted_30d`: `{low_accept['coef_brand_new_vs_established']:.4f}`, p `{low_accept['pval']:.4g}`." if low_accept is not None else "- Low-complexity certification gap unavailable.",
        f"- Mid-complexity tercile, brand-new vs established on `accepted_30d`: `{mid_accept['coef_brand_new_vs_established']:.4f}`, p `{mid_accept['pval']:.4g}`." if mid_accept is not None else "- Mid-complexity certification gap unavailable.",
        f"- High-complexity tercile, brand-new vs established on `accepted_30d`: `{high_accept['coef_brand_new_vs_established']:.4f}`, p `{high_accept['pval']:.4g}`." if high_accept is not None else "- High-complexity certification gap unavailable.",
        "",
        "Read:",
        "",
        "- If the speed benefit survives mainly in low/mid complexity while the certification penalty persists in higher complexity, the governance story becomes much sharper.",
        "- That is sellable because it says platforms can gain responsiveness from new entrants without gaining equivalent verification capacity.",
        "",
        "## Family Concentration",
        "",
        "Family-specific consequence models show where these subtype roles are economically strongest.",
        "",
    ]
    for _, row in fam_rows.iterrows():
        lines.append(
            f"- Family `{row['tag_family']}`, {row['family']} model, `{row['outcome']}` on `{row['term']}`: coef `{row['coef']:.4f}`, p `{row['pval']:.4g}`."
        )
    lines += [
        "",
        "Read:",
        "",
        "- The subtype pattern is broad, but the practical meaning differs by family.",
        "- The most sellable version is likely a paper where family context moderates how entrant-type re-sorting translates into speed versus certification tradeoffs.",
    ]
    ASYMMETRY_DEEP_DIVE_MEMO_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    established_results = build_established_interactions()
    brand_new_complexity = build_brand_new_complexity_gaps()
    family_results = build_family_consequence_results()
    write_memo(established_results, brand_new_complexity, family_results)


if __name__ == "__main__":
    main()
