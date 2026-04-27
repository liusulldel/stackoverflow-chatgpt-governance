from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

DURABILITY = PROCESSED / "who_still_answers_durability_entrant_panel.parquet"
USER_TAG_MONTH = PROCESSED / "who_still_answers_user_tag_month_panel.parquet"
FOCAL_ANSWERS = PROCESSED / "stackexchange_20251231_focal_answers.parquet"
ANSWER_ROLE = PROCESSED / "who_still_answers_answer_role_question_panel.parquet"
QUESTION_PANEL = PROCESSED / "who_still_answers_question_closure_panel.parquet"

EMBED_PANEL_CSV = PROCESSED / "p1_established_user_embedding_panel.csv"
EMBED_RESULTS_CSV = PROCESSED / "p1_established_user_embedding_results.csv"
ROLE_PANEL_CSV = PROCESSED / "p1_established_user_role_embedding_panel.csv"
ROLE_RESULTS_CSV = PROCESSED / "p1_established_user_role_embedding_results.csv"
READOUT_MD = PAPER / "p1_established_user_embedding_readout_2026-04-05.md"


def month_to_int(s: pd.Series) -> pd.Series:
    period = pd.PeriodIndex(s.astype(str), freq="M")
    return period.year * 12 + period.month


def clustered_formula(df: pd.DataFrame, outcome: str) -> dict[str, float | int | str]:
    cols = [
        outcome,
        "exposure_post",
        "prior_answers",
        "observed_tenure_days",
        "primary_tag",
        "entry_month",
    ]
    reg = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if reg.empty or reg["primary_tag"].nunique() < 2:
        return {
            "outcome": outcome,
            "coef": np.nan,
            "p_value": np.nan,
            "nobs": 0,
            "clusters": 0,
        }
    reg["log_prior_answers"] = np.log1p(reg["prior_answers"])
    reg["log_observed_tenure_days"] = np.log1p(reg["observed_tenure_days"])
    model = smf.ols(
        f"{outcome} ~ exposure_post + log_prior_answers + log_observed_tenure_days + C(primary_tag) + C(entry_month)",
        data=reg,
    ).fit(cov_type="cluster", cov_kwds={"groups": reg["primary_tag"]})
    return {
        "outcome": outcome,
        "coef": float(model.params.get("exposure_post", np.nan)),
        "p_value": float(model.pvalues.get("exposure_post", np.nan)),
        "nobs": int(model.nobs),
        "clusters": int(reg["primary_tag"].nunique()),
    }


def build_returner_base() -> pd.DataFrame:
    cols = [
        "primary_tag",
        "answerer_user_id",
        "first_tag_answer_at",
        "entry_month",
        "entrant_type",
        "eligible_365d",
        "return_365d",
        "exposure_index",
        "high_tag",
        "post_chatgpt",
        "exposure_post",
        "prior_answers",
        "observed_tenure_days",
    ]
    entrants = pd.read_parquet(DURABILITY, columns=cols)
    ret = entrants[
        (entrants["entrant_type"] == "established_cross_tag")
        & (entrants["eligible_365d"] == 1)
        & (entrants["return_365d"] == 1)
    ].copy()
    ret["entry_month_int"] = month_to_int(ret["entry_month"])
    ret["window_end_at"] = ret["first_tag_answer_at"] + pd.Timedelta(days=365)
    return ret


def build_embedding_panel(returners: pd.DataFrame) -> pd.DataFrame:
    utm = pd.read_parquet(
        USER_TAG_MONTH,
        columns=[
            "primary_tag",
            "answerer_user_id",
            "month_id",
            "answer_count",
            "accepted_current_count",
        ],
    )
    utm = utm.rename(columns={"primary_tag": "activity_tag"})
    utm = utm[utm["answerer_user_id"].isin(returners["answerer_user_id"].unique())].copy()
    utm["month_int"] = month_to_int(utm["month_id"])
    merged = utm.merge(returners, on="answerer_user_id", how="inner")
    merged = merged[
        (merged["month_int"] > merged["entry_month_int"])
        & (merged["month_int"] <= merged["entry_month_int"] + 12)
    ].copy()
    if merged.empty:
        return returners.copy()
    merged["is_local"] = merged["activity_tag"] == merged["primary_tag"]
    grp_cols = ["primary_tag", "answerer_user_id", "first_tag_answer_at", "entry_month"]

    merged["local_answer_count_12m"] = np.where(merged["is_local"], merged["answer_count"], 0.0)
    merged["other_answer_count_12m"] = np.where(~merged["is_local"], merged["answer_count"], 0.0)
    merged["local_accepted_current_count_12m"] = np.where(merged["is_local"], merged["accepted_current_count"], 0.0)
    merged["other_accepted_current_count_12m"] = np.where(~merged["is_local"], merged["accepted_current_count"], 0.0)

    panel = merged.groupby(grp_cols, as_index=False).agg(
        local_answer_count_12m=("local_answer_count_12m", "sum"),
        other_answer_count_12m=("other_answer_count_12m", "sum"),
        local_accepted_current_count_12m=("local_accepted_current_count_12m", "sum"),
        other_accepted_current_count_12m=("other_accepted_current_count_12m", "sum"),
    )

    local_months = (
        merged.loc[merged["is_local"] & (merged["answer_count"] > 0), grp_cols + ["month_id"]]
        .drop_duplicates()
        .groupby(grp_cols, as_index=False)
        .agg(local_active_months_12m=("month_id", "nunique"))
    )
    other_months = (
        merged.loc[(~merged["is_local"]) & (merged["answer_count"] > 0), grp_cols + ["month_id"]]
        .drop_duplicates()
        .groupby(grp_cols, as_index=False)
        .agg(other_active_months_12m=("month_id", "nunique"))
    )
    distinct_tags = (
        merged.loc[merged["answer_count"] > 0, grp_cols + ["activity_tag"]]
        .drop_duplicates()
        .groupby(grp_cols, as_index=False)
        .agg(distinct_tags_12m=("activity_tag", "nunique"))
    )

    panel = panel.merge(local_months, on=grp_cols, how="left")
    panel = panel.merge(other_months, on=grp_cols, how="left")
    panel = panel.merge(distinct_tags, on=grp_cols, how="left")

    out = returners.merge(panel, on=grp_cols, how="left")
    fill_zero = [
        "local_answer_count_12m",
        "other_answer_count_12m",
        "local_accepted_current_count_12m",
        "other_accepted_current_count_12m",
        "local_active_months_12m",
        "other_active_months_12m",
        "distinct_tags_12m",
    ]
    out[fill_zero] = out[fill_zero].fillna(0)
    total_answers = out["local_answer_count_12m"] + out["other_answer_count_12m"]
    out["local_share_12m"] = np.where(total_answers > 0, out["local_answer_count_12m"] / total_answers, np.nan)
    out["other_share_12m"] = np.where(total_answers > 0, out["other_answer_count_12m"] / total_answers, np.nan)
    out["local_only_12m"] = ((out["local_answer_count_12m"] > 0) & (out["other_answer_count_12m"] == 0)).astype(float)
    out["multi_tag_12m"] = (out["distinct_tags_12m"] > 1).astype(float)
    out["local_bench_3x3_12m"] = (
        (out["local_answer_count_12m"] >= 3)
        & (out["local_active_months_12m"] >= 3)
        & (out["local_share_12m"].fillna(0) >= 0.5)
    ).astype(float)
    out["local_accepted_current_rate_12m"] = np.where(
        out["local_answer_count_12m"] > 0,
        out["local_accepted_current_count_12m"] / out["local_answer_count_12m"],
        np.nan,
    )
    return out


def build_role_panel(returners: pd.DataFrame) -> pd.DataFrame:
    answers = pd.read_parquet(
        FOCAL_ANSWERS,
        columns=[
            "answer_id",
            "question_id",
            "answer_created_at",
            "owner_user_id",
            "is_current_accepted_answer",
        ],
    ).rename(columns={"owner_user_id": "answerer_user_id"})
    answers = answers[answers["answerer_user_id"].isin(returners["answerer_user_id"].unique())].copy()

    qcols = [
        "question_id",
        "question_created_at",
        "accepted_30d",
        "first_answer_1d",
        "primary_tag",
    ]
    qpanel = pd.read_parquet(QUESTION_PANEL, columns=qcols).rename(columns={"primary_tag": "closure_question_tag"})
    answers = answers.merge(qpanel, on="question_id", how="left")
    answers["answer_lag_hours"] = (
        (answers["answer_created_at"] - answers["question_created_at"]).dt.total_seconds() / 3600.0
    )

    answers = answers.rename(columns={"closure_question_tag": "question_tag"})

    merge_cols = [
        "primary_tag",
        "answerer_user_id",
        "first_tag_answer_at",
        "entry_month",
        "window_end_at",
        "prior_answers",
        "observed_tenure_days",
        "exposure_post",
    ]
    merged = answers.merge(
        returners[merge_cols],
        left_on=["answerer_user_id", "question_tag"],
        right_on=["answerer_user_id", "primary_tag"],
        how="inner",
    )
    merged = merged[
        (merged["answer_created_at"] > merged["first_tag_answer_at"])
        & (merged["answer_created_at"] <= merged["window_end_at"])
    ].copy()

    role_rows = pd.read_parquet(
        ANSWER_ROLE,
        columns=["answer_id", "role"],
    )
    role_rows = role_rows[role_rows["answer_id"].isin(merged["answer_id"].unique())].copy()
    role_flags = (
        pd.crosstab(role_rows["answer_id"], role_rows["role"])
        .reindex(columns=["first_answer", "first_positive", "top_score", "accepted_current"], fill_value=0)
        .reset_index()
        .rename(
            columns={
                "first_answer": "first_answer_role",
                "first_positive": "first_positive_role",
                "top_score": "top_score_role",
                "accepted_current": "accepted_current_role",
            }
        )
    )
    merged = merged.merge(role_flags, on="answer_id", how="left")
    for col in ["first_answer_role", "first_positive_role", "top_score_role", "accepted_current_role"]:
        merged[col] = merged[col].fillna(0).astype(float)
    merged["later_or_deeper_role"] = 1.0 - merged["first_answer_role"]

    grp_cols = ["primary_tag", "answerer_user_id", "first_tag_answer_at", "entry_month"]
    panel = merged.groupby(grp_cols, as_index=False).agg(
        local_answer_rows=("answer_id", "size"),
        local_first_answer_rate=("first_answer_role", "mean"),
        local_later_or_deeper_rate=("later_or_deeper_role", "mean"),
        local_top_score_rate=("top_score_role", "mean"),
        local_first_positive_rate=("first_positive_role", "mean"),
        local_accepted_current_role_rate=("accepted_current_role", "mean"),
        local_mean_answer_lag_hours=("answer_lag_hours", "mean"),
        local_question_accepted30d_rate=("accepted_30d", "mean"),
        local_question_first_answer1d_rate=("first_answer_1d", "mean"),
    )
    out = returners.merge(panel, on=grp_cols, how="left")
    return out


def write_readout(embed_results: pd.DataFrame, role_results: pd.DataFrame, embed_panel: pd.DataFrame, role_panel: pd.DataFrame) -> None:
    def fmt(df: pd.DataFrame, name: str) -> str:
        row = df.loc[df["outcome"] == name].iloc[0]
        return f"`{name}`: coef `{row['coef']:.4f}`, p `{row['p_value']:.4g}`"

    returned_n = len(embed_panel)
    unique_users = int(embed_panel["answerer_user_id"].nunique())
    local_only_mean = float(embed_panel["local_only_12m"].mean())
    local_share_mean = float(embed_panel["local_share_12m"].mean(skipna=True))
    bench_mean = float(embed_panel["local_bench_3x3_12m"].mean())
    text = f"""# Established Cross-Tag User-Level Embedding / Assimilation Readout

Date: `2026-04-05`

Sample: returning `established_cross_tag` entrants with observable `365d` window.

- returning user-tag pairs: `{returned_n:,}`
- unique users: `{unique_users:,}`
- mean `local_share_12m`: `{local_share_mean:.3f}`
- mean `local_only_12m`: `{local_only_mean:.3f}`
- mean `local_bench_3x3_12m`: `{bench_mean:.3f}`

## Local Embedding Surface

- {fmt(embed_results, "local_share_12m")}
- {fmt(embed_results, "local_only_12m")}
- {fmt(embed_results, "local_bench_3x3_12m")}
- {fmt(embed_results, "distinct_tags_12m")}
- {fmt(embed_results, "local_active_months_12m")}
- {fmt(embed_results, "local_accepted_current_rate_12m")}

## Role / Depth / Certification Surface

- {fmt(role_results, "local_first_answer_rate")}
- {fmt(role_results, "local_later_or_deeper_rate")}
- {fmt(role_results, "local_accepted_current_role_rate")}
- {fmt(role_results, "local_mean_answer_lag_hours")}
- {fmt(role_results, "local_question_accepted30d_rate")}

## Read

These tests ask whether returning established cross-tag entrants actually embed into the focal tag as a sustainable local bench, or whether they remain a narrow imported fallback pool.

The local-embedding surface is the key one for assimilation:

- higher `local_share_12m`, `local_only_12m`, and `local_bench_3x3_12m` would look like stronger focal-tag embedding
- lower `distinct_tags_12m` would also support local embedding

The role/depth surface distinguishes two stories:

- if `local_first_answer_rate` is high and lag is short, the returners are still acting like front-end responders
- if `local_later_or_deeper_rate`, `local_accepted_current_role_rate`, and `local_mean_answer_lag_hours` rise, the returners look more like narrow deeper-stage fallback

This layer should still be treated as bounded observational evidence rather than causal assimilation.
"""
    READOUT_MD.write_text(text, encoding="utf-8")


def main() -> None:
    returners = build_returner_base()
    embed_panel = build_embedding_panel(returners)
    role_panel = build_role_panel(returners)

    embed_panel.to_csv(EMBED_PANEL_CSV, index=False)
    role_panel.to_csv(ROLE_PANEL_CSV, index=False)

    embed_outcomes = [
        "local_share_12m",
        "local_only_12m",
        "local_bench_3x3_12m",
        "distinct_tags_12m",
        "local_active_months_12m",
        "local_accepted_current_rate_12m",
    ]
    role_outcomes = [
        "local_first_answer_rate",
        "local_later_or_deeper_rate",
        "local_accepted_current_role_rate",
        "local_mean_answer_lag_hours",
        "local_question_accepted30d_rate",
    ]

    embed_results = pd.DataFrame([clustered_formula(embed_panel, o) for o in embed_outcomes])
    role_results = pd.DataFrame([clustered_formula(role_panel, o) for o in role_outcomes])

    embed_results.to_csv(EMBED_RESULTS_CSV, index=False)
    role_results.to_csv(ROLE_RESULTS_CSV, index=False)
    write_readout(embed_results, role_results, embed_panel, role_panel)


if __name__ == "__main__":
    main()
