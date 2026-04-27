from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(r"D:\AI alignment\projects\stackoverflow_chatgpt_governance")
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

EXPOSURE_RANKING = PROCESSED / "who_still_answers_exposure_ranking_table.csv"
JETBRAINS_CLUSTER = PROCESSED / "who_still_answers_jetbrains_calibration_table.csv"
INCUMBENT_PANEL = PROCESSED / "who_still_answers_incumbent_cohort_panel.csv"
ROLE_Q_PANEL = PROCESSED / "who_still_answers_answer_role_question_panel.parquet"
ACCEPT_VOTES = PROCESSED / "stackexchange_20251231_focal_accept_votes.parquet"

OUT_16TAG = PROCESSED / "who_still_answers_16tag_external_calibration_table.csv"
OUT_MECH = PROCESSED / "who_still_answers_mechanism_decomposition_table.csv"
OUT_CERT = PROCESSED / "who_still_answers_certification_conversion_ladder.csv"
OUT_SUMMARY = PROCESSED / "who_still_answers_finish_empirical_upgrades_summary.json"
OUT_READOUT = PAPER / "who_still_answers_finish_empirical_upgrades_readout_2026-04-16.md"


TAG_TO_JETBRAINS_CLUSTER = {
    "sql": "SQL / analytics",
    "excel": "SQL / analytics",
    "regex": "SQL / analytics",
    "python": "Python / data",
    "pandas": "Python / data",
    "numpy": "Python / data",
    "apache-spark": "Python / data",
    "javascript": "JavaScript / web",
    "firebase": "JavaScript / web",
    "android": "Android / mobile",
    "multithreading": "Android / mobile",
    "memory-management": "Android / mobile",
    "bash": "Shell / infra / cloud",
    "linux": "Shell / infra / cloud",
    "docker": "Shell / infra / cloud",
    "kubernetes": "Shell / infra / cloud",
}


def add_time_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    month = pd.PeriodIndex(out["month_id"].astype(str), freq="M")
    out["time_index"] = (month.year - month.year.min()) * 12 + month.month - month.month.min() + 1
    out["exposure_post"] = out["exposure_index"] * out["post_chatgpt"]
    return out


def fit_panel(
    df: pd.DataFrame,
    outcome: str,
    weight_col: str | None = None,
) -> dict[str, float | int | str]:
    cols = ["primary_tag", "month_id", "time_index", "exposure_post", outcome]
    if weight_col:
        cols.append(weight_col)
    use = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    formula = f"{outcome} ~ exposure_post + C(primary_tag) + C(month_id) + C(primary_tag):time_index"
    if weight_col:
        mod = smf.wls(formula, data=use, weights=use[weight_col])
    else:
        mod = smf.ols(formula, data=use)
    res = mod.fit(cov_type="cluster", cov_kwds={"groups": use["primary_tag"]})
    return {
        "coef": float(res.params.get("exposure_post", np.nan)),
        "se": float(res.bse.get("exposure_post", np.nan)),
        "pval": float(res.pvalues.get("exposure_post", np.nan)),
        "nobs": int(res.nobs),
        "mean_outcome": float(use[outcome].mean()),
    }


def build_16tag_external_calibration() -> pd.DataFrame:
    tags = pd.read_csv(EXPOSURE_RANKING)
    jets = pd.read_csv(JETBRAINS_CLUSTER)
    tags["jetbrains_cluster"] = tags["tag"].map(TAG_TO_JETBRAINS_CLUSTER)
    merged = tags.merge(
        jets[
            [
                "cluster",
                "respondent_n",
                "chatgpt_answers_share",
                "stackoverflow_answers_share",
                "private_public_gap",
                "ai_search_saving_share",
                "ai_debugging_share",
            ]
        ],
        left_on="jetbrains_cluster",
        right_on="cluster",
        how="left",
    )
    merged["external_calibration_level"] = "JetBrains cluster mapped to focal tag"
    merged["safe_interpretation"] = (
        "External developer-survey cluster supports the private-vs-public substitution premise; "
        "it does not independently validate this tag's exact exposure rank."
    )
    keep = [
        "exposure_rank",
        "tag",
        "exposure_tercile",
        "exposure_index",
        "manual_high",
        "pre_questions",
        "jetbrains_cluster",
        "respondent_n",
        "chatgpt_answers_share",
        "stackoverflow_answers_share",
        "private_public_gap",
        "ai_search_saving_share",
        "ai_debugging_share",
        "external_calibration_level",
        "safe_interpretation",
    ]
    out = merged[keep].sort_values("exposure_rank").reset_index(drop=True)
    out.to_csv(OUT_16TAG, index=False)
    return out


def build_mechanism_decomposition() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    inc = add_time_index(pd.read_csv(INCUMBENT_PANEL))
    for is_expert, label in [(1, "pre-period expert incumbents"), (0, "pre-period nonexpert incumbents")]:
        sub = inc[inc["is_expert"] == is_expert].copy()
        for outcome in ["share_active", "mean_answers"]:
            est = fit_panel(sub, outcome, weight_col="n_user_tag_pairs")
            rows.append(
                {
                    "family": "incumbent activity",
                    "margin": label,
                    "role": "all roles",
                    "outcome": outcome,
                    "coefficient": est["coef"],
                    "se": est["se"],
                    "p_value": est["pval"],
                    "nobs": est["nobs"],
                    "mean_outcome": est["mean_outcome"],
                    "safe_read": "Higher-exposure post-period change among pre-period incumbent contributor-tag pairs.",
                }
            )

    role = pd.read_parquet(ROLE_Q_PANEL)
    role["is_positive_score"] = role["score"].fillna(0).gt(0).astype(float)
    role["answer_count"] = 1
    group_cols = ["primary_tag", "month_id", "role", "post_chatgpt", "exposure_index"]
    role_tm = (
        role.groupby(group_cols, observed=True)
        .agg(
            n_answers=("answer_count", "sum"),
            recent_entrant_90d_share=("recent_entrant_90d", "mean"),
            incumbent_365d_share=("incumbent_365d", "mean"),
            brand_new_platform_share=("brand_new_platform", "mean"),
            established_cross_tag_share=("established_cross_tag", "mean"),
            accepted_current_share=("is_current_accepted_answer", "mean"),
            positive_score_share=("is_positive_score", "mean"),
        )
        .reset_index()
    )
    role_tm = add_time_index(role_tm)
    for role_name in ["first_answer", "first_positive", "top_score", "accepted_current"]:
        sub = role_tm[role_tm["role"] == role_name].copy()
        for outcome, read in [
            ("recent_entrant_90d_share", "Recent entrants become more visible in this answer-pipeline role."),
            ("incumbent_365d_share", "Pre-existing tag incumbents occupy this answer-pipeline role less often if negative."),
            ("brand_new_platform_share", "Brand-new platform users occupy this role more often if positive."),
            ("established_cross_tag_share", "Established cross-tag users provide fallback capacity if positive."),
        ]:
            est = fit_panel(sub, outcome, weight_col="n_answers")
            rows.append(
                {
                    "family": "role-location decomposition",
                    "margin": outcome,
                    "role": role_name,
                    "outcome": outcome,
                    "coefficient": est["coef"],
                    "se": est["se"],
                    "p_value": est["pval"],
                    "nobs": est["nobs"],
                    "mean_outcome": est["mean_outcome"],
                    "safe_read": read,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_MECH, index=False)
    return out


def build_certification_ladder() -> pd.DataFrame:
    role = pd.read_parquet(ROLE_Q_PANEL)
    accepts = pd.read_parquet(ACCEPT_VOTES)
    accepts = accepts[["answer_id", "accept_vote_date"]].drop_duplicates("answer_id")
    role = role.merge(accepts, on="answer_id", how="left")
    role["answer_created_at"] = pd.to_datetime(role["answer_created_at"], utc=True, errors="coerce")
    role["accept_vote_date"] = pd.to_datetime(role["accept_vote_date"], utc=True, errors="coerce")
    role["accepted_current"] = role["is_current_accepted_answer"].fillna(0).astype(float)
    role["positive_score"] = role["score"].fillna(0).gt(0).astype(float)
    # Accept-vote dates in the public dump are calendar dates, not exact timestamps.
    # Calendar-day windows avoid treating same-day accepted answers as negative-lag events.
    role["accept_lag_calendar_days"] = (
        role["accept_vote_date"].dt.normalize() - role["answer_created_at"].dt.normalize()
    ).dt.days
    role.loc[role["accept_lag_calendar_days"] < 0, "accept_lag_calendar_days"] = np.nan
    role["accepted_any_archive"] = role["accept_vote_date"].notna().astype(float)
    role["accepted_0d"] = role["accept_lag_calendar_days"].eq(0).astype(float)
    role["accepted_7d"] = role["accept_lag_calendar_days"].between(0, 7, inclusive="both").astype(float)
    role["accepted_30d"] = role["accept_lag_calendar_days"].between(0, 30, inclusive="both").astype(float)
    role["answer_count"] = 1

    group_cols = ["primary_tag", "month_id", "role", "post_chatgpt", "exposure_index"]
    tm = (
        role.groupby(group_cols, observed=True)
        .agg(
            n_answers=("answer_count", "sum"),
            accepted_0d_rate=("accepted_0d", "mean"),
            accepted_7d_rate=("accepted_7d", "mean"),
            accepted_30d_rate=("accepted_30d", "mean"),
            accepted_any_archive_rate=("accepted_any_archive", "mean"),
            current_accepted_archive_rate=("accepted_current", "mean"),
            positive_score_rate=("positive_score", "mean"),
            mean_score=("score", "mean"),
        )
        .reset_index()
    )
    tm = add_time_index(tm)

    rows: list[dict[str, object]] = []
    for role_name, role_label in [
        ("first_answer", "first answer"),
        ("first_positive", "first positive answer"),
        ("top_score", "top-scored answer"),
    ]:
        sub = tm[tm["role"] == role_name].copy()
        for outcome, label in [
            ("accepted_0d_rate", "P(role answer accepted same calendar day)"),
            ("accepted_7d_rate", "P(role answer accepted within 7 calendar days)"),
            ("accepted_30d_rate", "P(role answer accepted within 30 calendar days)"),
            ("accepted_any_archive_rate", "P(role answer ever accepted in archive)"),
            ("current_accepted_archive_rate", "P(role answer is current accepted answer)"),
            ("positive_score_rate", "P(role answer receives positive score)"),
            ("mean_score", "archive answer score"),
        ]:
            if outcome == "positive_score_rate" and role_name == "first_positive":
                continue
            est = fit_panel(sub, outcome, weight_col="n_answers")
            rows.append(
                {
                    "role": role_label,
                    "outcome": label,
                    "coefficient": est["coef"],
                    "se": est["se"],
                    "p_value": est["pval"],
                    "nobs": est["nobs"],
                    "mean_outcome": est["mean_outcome"],
                    "safe_read": (
                        "Exposure-post coefficient from tag-month role panel with tag FE, month FE, "
                        "tag-specific linear trends, and answer-count weights."
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CERT, index=False)
    return out


def write_readout(cal: pd.DataFrame, mech: pd.DataFrame, cert: pd.DataFrame) -> None:
    top_cal = cal[["tag", "exposure_rank", "jetbrains_cluster", "private_public_gap"]].head(5)
    mech_keep = mech.sort_values("p_value").head(8)
    cert_keep = cert.sort_values("p_value").head(8)
    lines = [
        "# Who Still Answers: Finish Empirical Upgrades Readout",
        "",
        "Date: 2026-04-16",
        "",
        "## What changed",
        "",
        "This build converts the last high-value empirical upgrades into three reviewer-facing tables.",
        "",
        "1. `16-tag external calibration`: every focal tag is shown with its internal exposure rank and mapped JetBrains cluster.",
        "2. `mechanism decomposition`: incumbent activity and role-location margins are separated rather than collapsed into a generic entrant story.",
        "3. `certification conversion ladder`: early answer roles are linked directly to accepted-current conversion and positive-score endorsement.",
        "",
        "## Safe interpretation",
        "",
        "- The JetBrains layer calibrates the substitution premise at the cluster level; it does not independently verify each tag rank.",
        "- The mechanism decomposition is descriptive panel evidence with the same bounded exposure-post interpretation as the main design.",
        "- The certification ladder is a conversion readout, not a causal mediation design.",
        "",
        "## Top exposure rows",
        "",
        top_cal.to_markdown(index=False),
        "",
        "## Strongest decomposition rows",
        "",
        mech_keep.to_markdown(index=False),
        "",
        "## Strongest certification-ladder rows",
        "",
        cert_keep.to_markdown(index=False),
        "",
    ]
    OUT_READOUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cal = build_16tag_external_calibration()
    mech = build_mechanism_decomposition()
    cert = build_certification_ladder()
    write_readout(cal, mech, cert)
    OUT_SUMMARY.write_text(
        json.dumps(
            {
                "outputs": {
                    "16tag_external_calibration": str(OUT_16TAG),
                    "mechanism_decomposition": str(OUT_MECH),
                    "certification_conversion_ladder": str(OUT_CERT),
                    "readout": str(OUT_READOUT),
                },
                "n_rows": {
                    "16tag_external_calibration": int(len(cal)),
                    "mechanism_decomposition": int(len(mech)),
                    "certification_conversion_ladder": int(len(cert)),
                },
                "safe_limits": [
                    "JetBrains calibrates substitution premise at cluster level, not exact tag rank.",
                    "Mechanism decomposition separates channels but remains observational.",
                    "Certification ladder is a conversion readout, not mediation.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
