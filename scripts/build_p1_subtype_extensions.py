from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

SUBTYPE_PANEL_CSV = PROCESSED / "p1_p2_harmonized_subtype_panel.csv"
VALIDATION_SAMPLE_PARQUET = PROCESSED / "entrant_first_question_validation_sample.parquet"
PROFILE_SUMMARY_CSV = PROCESSED / "p1_entrant_type_profile_summary.csv"
SUBTYPE_CONSEQUENCE_RESULTS_CSV = PROCESSED / "p1_jmis_subtype_consequence_results.csv"

TAG_FAMILY_MAP_CSV = PROCESSED / "p1_tag_family_map.csv"
TAG_FAMILY_SUMMARY_CSV = PROCESSED / "p1_tag_family_summary.csv"
TAG_LEVEL_BRAND_NEW_GAPS_CSV = PROCESSED / "p1_brand_new_tag_level_gaps.csv"
TAG_FAMILY_BRAND_NEW_GAPS_CSV = PROCESSED / "p1_brand_new_family_gaps.csv"
SUBTYPE_EXTENSION_MEMO_MD = PAPER / "p1_subtype_extensions_memo.md"


TAG_FAMILY_MAP = {
    "bash": "data_scripting",
    "excel": "data_scripting",
    "numpy": "data_scripting",
    "pandas": "data_scripting",
    "python": "data_scripting",
    "regex": "data_scripting",
    "sql": "data_scripting",
    "javascript": "application_framework",
    "android": "application_framework",
    "firebase": "application_framework",
    "docker": "systems_infra",
    "kubernetes": "systems_infra",
    "linux": "systems_infra",
    "memory-management": "systems_infra",
    "multithreading": "systems_infra",
    "apache-spark": "systems_infra",
}


def build_tag_family_map() -> pd.DataFrame:
    out = pd.DataFrame(
        [{"primary_tag": tag, "tag_family": family} for tag, family in TAG_FAMILY_MAP.items()]
    ).sort_values(["tag_family", "primary_tag"])
    out.to_csv(TAG_FAMILY_MAP_CSV, index=False)
    return out


def build_tag_family_summary(tag_map: pd.DataFrame) -> pd.DataFrame:
    panel = pd.read_csv(SUBTYPE_PANEL_CSV)
    agg = panel.groupby(["primary_tag", "post_chatgpt"], as_index=False).agg(
        brand_new_platform_share=("brand_new_platform_share", "mean"),
        low_tenure_existing_share=("low_tenure_existing_share", "mean"),
        established_cross_tag_share=("established_cross_tag_share", "mean"),
        first_answer_1d_rate=("first_answer_1d_rate_closure", "mean"),
        accepted_vote_30d_rate=("accepted_vote_30d_rate", "mean"),
        exposure_index=("exposure_index", "first"),
        high_tag=("high_tag", "first"),
    )
    wide = agg.pivot(index="primary_tag", columns="post_chatgpt")
    wide.columns = ["_".join([col, str(int(flag))]) for col, flag in wide.columns]
    wide = wide.reset_index()
    for var in [
        "brand_new_platform_share",
        "low_tenure_existing_share",
        "established_cross_tag_share",
        "first_answer_1d_rate",
        "accepted_vote_30d_rate",
    ]:
        wide[f"{var}_delta"] = wide[f"{var}_1"] - wide[f"{var}_0"]
    wide = wide.merge(tag_map, on="primary_tag", how="left")
    family_summary = (
        wide.groupby("tag_family", as_index=False)
        .agg(
            n_tags=("primary_tag", "size"),
            mean_exposure_index=("exposure_index_0", "mean"),
            mean_brand_new_delta=("brand_new_platform_share_delta", "mean"),
            mean_low_tenure_delta=("low_tenure_existing_share_delta", "mean"),
            mean_established_delta=("established_cross_tag_share_delta", "mean"),
            mean_first_answer_delta=("first_answer_1d_rate_delta", "mean"),
            mean_accepted_vote_delta=("accepted_vote_30d_rate_delta", "mean"),
        )
        .sort_values("mean_exposure_index", ascending=False)
    )
    family_summary.to_csv(TAG_FAMILY_SUMMARY_CSV, index=False)
    return family_summary


def build_brand_new_gaps(tag_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = pd.read_parquet(VALIDATION_SAMPLE_PARQUET)
    sample = sample[sample["entrant_type"].isin(["brand_new_platform", "established_cross_tag"])].copy()
    sample = sample.merge(tag_map, on="primary_tag", how="left")
    sample["is_brand_new"] = (sample["entrant_type"] == "brand_new_platform").astype(int)

    rows = []
    for tag, frame in sample.groupby("primary_tag"):
        if frame["entrant_type"].nunique() < 2 or len(frame) < 100:
            continue
        for outcome in ["score", "is_current_accepted_answer", "accepted_30d", "first_answer_1d"]:
            model = smf.ols(f"{outcome} ~ is_brand_new + C(entry_month)", data=frame).fit()
            rows.append(
                {
                    "scope": "tag",
                    "primary_tag": tag,
                    "tag_family": frame["tag_family"].iloc[0],
                    "outcome": outcome,
                    "coef_brand_new_vs_established": float(model.params.get("is_brand_new", float("nan"))),
                    "pval": float(model.pvalues.get("is_brand_new", float("nan"))),
                    "nobs": int(model.nobs),
                }
            )
    tag_out = pd.DataFrame(rows).sort_values(["outcome", "coef_brand_new_vs_established"])
    tag_out.to_csv(TAG_LEVEL_BRAND_NEW_GAPS_CSV, index=False)

    family_rows = []
    for family, frame in sample.groupby("tag_family"):
        if frame["entrant_type"].nunique() < 2 or len(frame) < 100:
            continue
        for outcome in ["score", "is_current_accepted_answer", "accepted_30d", "first_answer_1d"]:
            model = smf.ols(f"{outcome} ~ is_brand_new + C(primary_tag) + C(entry_month)", data=frame).fit()
            family_rows.append(
                {
                    "tag_family": family,
                    "outcome": outcome,
                    "coef_brand_new_vs_established": float(model.params.get("is_brand_new", float("nan"))),
                    "pval": float(model.pvalues.get("is_brand_new", float("nan"))),
                    "nobs": int(model.nobs),
                }
            )
    family_out = pd.DataFrame(family_rows).sort_values(["outcome", "coef_brand_new_vs_established"])
    family_out.to_csv(TAG_FAMILY_BRAND_NEW_GAPS_CSV, index=False)
    return tag_out, family_out


def write_memo(
    family_summary: pd.DataFrame,
    profile_summary: pd.DataFrame,
    consequence_results: pd.DataFrame,
    family_gaps: pd.DataFrame,
    tag_gaps: pd.DataFrame,
) -> None:
    est_any_answer = consequence_results[
        (consequence_results["model"] == "established_only")
        & (consequence_results["outcome"] == "any_answer_7d_rate")
        & (consequence_results["term"] == "established_cross_tag_share")
    ].sort_values("family")
    est_latency = consequence_results[
        (consequence_results["model"] == "established_only")
        & (consequence_results["outcome"] == "first_positive_answer_latency_mean")
        & (consequence_results["term"] == "established_cross_tag_share")
    ].sort_values("family")
    brand_family_score = family_gaps[
        (family_gaps["outcome"] == "score")
    ].sort_values("coef_brand_new_vs_established")
    brand_family_accept = family_gaps[
        (family_gaps["outcome"] == "accepted_30d")
    ].sort_values("coef_brand_new_vs_established")
    brand_tag_score = tag_gaps[tag_gaps["outcome"] == "score"].sort_values("coef_brand_new_vs_established")

    lines = [
        "# P1 Subtype Extensions",
        "",
        "## 1. Established Cross-Tag Mechanism: What Can Be Said Now",
        "",
        "The current evidence does not support a `surge` story for `established_cross_tag`. Its post-period share declines almost everywhere in raw deltas, and the cross-tag-share delta itself is not meaningfully exposure-graded at the tag level.",
        "",
        "What does stand is a different candidate mechanism:",
        "",
        "- `established_cross_tag` is the dominant post-shock entrant pool by volume and experience, not by novice status.",
        "- Its members are highly experienced elsewhere on the platform, with long tenure and substantial prior answer histories.",
        "- Yet tag-months with relatively higher `established_cross_tag_share` have worse front-end and deeper-progression outcomes.",
        "",
        f"- Post-shock profile: `mean_tenure_days = {profile_summary.loc[profile_summary['entrant_type']=='established_cross_tag','mean_tenure_days'].iloc[0]:.1f}`, `mean_prior_answers = {profile_summary.loc[profile_summary['entrant_type']=='established_cross_tag','mean_prior_answers'].iloc[0]:.1f}`.",
    ]
    for _, row in est_any_answer.iterrows():
        lines.append(
            f"- {row['family']} established-only consequence on `any_answer_7d_rate`: coef `{row['coef']:.4f}`, p `{row['pval']:.4g}`."
        )
    for _, row in est_latency.iterrows():
        lines.append(
            f"- {row['family']} established-only consequence on `first_positive_answer_latency_mean`: coef `{row['coef']:.1f}` hours, p `{row['pval']:.4g}`."
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "The safest current mechanism candidate is not that cross-tag migrants are individually low quality. It is that when local replenishment is thin, the answer pool becomes more dependent on imported cross-tag incumbents who appear less able to maintain fast broad coverage, even if they individually certify better than brand-new entrants.",
        "",
        "## 2. Brand-New Platform Entrants: Governance Interpretation",
        "",
        "The strongest governance reading is now `fast response is not the same thing as certified resolution`.",
        "",
    ]
    for _, row in brand_family_score.iterrows():
        lines.append(
            f"- Family `{row['tag_family']}` brand-new vs established score gap: coef `{row['coef_brand_new_vs_established']:.4f}`, p `{row['pval']:.4g}`."
        )
    for _, row in brand_family_accept.iterrows():
        lines.append(
            f"- Family `{row['tag_family']}` brand-new vs established accepted-30d gap: coef `{row['coef_brand_new_vs_established']:.4f}`, p `{row['pval']:.4g}`."
        )
    lines += [
        "",
        "Brand-new entrants appear to help with responsiveness, but they underperform on downstream certification signals. That supports platform-governance implications such as:",
        "",
        "- separate response-speed dashboards from certification/acceptance dashboards",
        "- build triage or escalation paths that route hard residual questions away from pure first-response metrics",
        "- treat newcomer participation as throughput-enhancing but not as a substitute for deeper verification capacity",
        "",
        "## 3. Tag-Family Concentration",
        "",
        "The subtype pattern is not driven by only one or two tags; brand-new share rises and low-tenure share falls in all families. But the economic meaning of that shift is not uniform across families.",
        "",
    ]
    for _, row in family_summary.iterrows():
        lines.append(
            f"- Family `{row['tag_family']}`: exposure mean `{row['mean_exposure_index']:.3f}`, brand-new delta `{row['mean_brand_new_delta']:.3f}`, low-tenure delta `{row['mean_low_tenure_delta']:.3f}`, established delta `{row['mean_established_delta']:.3f}`, first-answer delta `{row['mean_first_answer_delta']:.3f}`."
        )
    top_tags = brand_tag_score.head(6)
    lines += [
        "",
        "The brand-new certification disadvantage is strongest in these tags:",
        "",
    ]
    for _, row in top_tags.iterrows():
        lines.append(
            f"- `{row['primary_tag']}` ({row['tag_family']}): score gap `{row['coef_brand_new_vs_established']:.4f}`, p `{row['pval']:.4g}`."
        )
    lines += [
        "",
        "Read:",
        "",
        "- The entrant re-sorting is widespread, not a one-tag artifact.",
        "- The governance stakes appear especially legible in data/scripting and application-framework tags, where brand-new participation rises sharply but certification gaps remain visible.",
        "- The adverse `established_cross_tag` consequence pattern is better read as a fallback-dependence story than as a raw volume-surge story.",
    ]
    SUBTYPE_EXTENSION_MEMO_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    tag_map = build_tag_family_map()
    family_summary = build_tag_family_summary(tag_map)
    profile_summary = pd.read_csv(PROFILE_SUMMARY_CSV)
    consequence_results = pd.read_csv(SUBTYPE_CONSEQUENCE_RESULTS_CSV)
    tag_gaps, family_gaps = build_brand_new_gaps(tag_map)
    write_memo(
        family_summary=family_summary,
        profile_summary=profile_summary,
        consequence_results=consequence_results,
        family_gaps=family_gaps,
        tag_gaps=tag_gaps,
    )


if __name__ == "__main__":
    main()
