from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

DURABILITY_ENTRANT_PANEL = PROCESSED / "who_still_answers_durability_entrant_panel.parquet"
USER_TAG_MONTH_PANEL = PROCESSED / "who_still_answers_user_tag_month_panel.parquet"
ROLE_QUESTION_PANEL = PROCESSED / "who_still_answers_answer_role_question_panel.parquet"

EMBEDDING_USER_PANEL_CSV = PROCESSED / "p1_established_user_embedding_panel.csv"
EMBEDDING_USER_RESULTS_CSV = PROCESSED / "p1_established_user_embedding_results.csv"
EMBEDDING_ROLE_PANEL_CSV = PROCESSED / "p1_established_user_role_embedding_panel.csv"
EMBEDDING_ROLE_RESULTS_CSV = PROCESSED / "p1_established_user_role_embedding_results.csv"
READOUT_MD = PAPER / "p1_established_user_embedding_readout_2026-04-05.md"

ROLE_ORDER = ["first_answer", "first_positive", "top_score", "accepted_current"]


def fit_wls(frame: pd.DataFrame, formula: str, weight_col: str, cluster_col: str = "primary_tag"):
    model_frame = frame.dropna(subset=[weight_col, cluster_col]).copy()
    y, X = dmatrices(formula, data=model_frame, return_type="dataframe", NA_action="drop")
    weights = model_frame.loc[X.index, weight_col].astype(float)
    groups = model_frame.loc[X.index, cluster_col]
    return sm.WLS(y, X, weights=weights).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True, "df_correction": True},
    )


def load_returning_established() -> pd.DataFrame:
    entrants = pd.read_parquet(DURABILITY_ENTRANT_PANEL)
    entrants["entry_month"] = entrants["entry_month"].astype(str)
    month_order = {month: idx + 1 for idx, month in enumerate(sorted(entrants["entry_month"].dropna().unique()))}
    entrants["time_index"] = entrants["entry_month"].map(month_order)
    entrants["high_post"] = entrants["high_tag"] * entrants["post_chatgpt"]
    entrants["exposure_post"] = entrants["exposure_index"] * entrants["post_chatgpt"]
    entrants["answerer_user_id"] = pd.to_numeric(entrants["answerer_user_id"], errors="coerce")
    sub = entrants.loc[
        (entrants["entrant_type"] == "established_cross_tag")
        & (entrants["eligible_365d"] == 1)
        & (entrants["return_365d"] == 1)
    ].copy()
    sub["entrant_row_id"] = np.arange(len(sub))
    return sub[
        [
            "entrant_row_id",
            "primary_tag",
            "answerer_user_id",
            "entry_month",
            "time_index",
            "post_chatgpt",
            "exposure_index",
            "high_tag",
            "high_post",
            "exposure_post",
            "first_tag_answer_at",
            "first_return_at",
        ]
    ].copy()


def build_user_embedding_panel(returners: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("returners", returners)
    entrant_panel = con.execute(
        f"""
        WITH utm AS (
            SELECT
                primary_tag,
                CAST(answerer_user_id AS BIGINT) AS answerer_user_id,
                CAST(month_id || '-01' AS DATE) AS month_date,
                answer_count,
                accepted_current_count
            FROM read_parquet('{USER_TAG_MONTH_PANEL.as_posix()}')
        )
        SELECT
            r.entrant_row_id,
            r.primary_tag,
            r.entry_month,
            r.time_index,
            r.post_chatgpt,
            r.exposure_index,
            r.high_tag,
            r.high_post,
            r.exposure_post,
            COALESCE(SUM(u.answer_count), 0.0) AS total_answers_post_return_365d,
            COALESCE(SUM(CASE WHEN u.primary_tag = r.primary_tag THEN u.answer_count ELSE 0 END), 0.0) AS focal_answers_post_return_365d,
            COALESCE(SUM(CASE WHEN u.primary_tag = r.primary_tag THEN u.accepted_current_count ELSE 0 END), 0.0) AS focal_accepted_current_post_return_365d,
            COALESCE(SUM(u.accepted_current_count), 0.0) AS total_accepted_current_post_return_365d,
            COALESCE(COUNT(DISTINCT CASE WHEN u.primary_tag = r.primary_tag AND u.answer_count > 0 THEN u.month_date END), 0.0) AS focal_active_months_post_return_365d,
            COALESCE(COUNT(DISTINCT CASE WHEN u.primary_tag <> r.primary_tag AND u.answer_count > 0 THEN u.primary_tag END), 0.0) AS other_tags_touched_post_return_365d
        FROM returners r
        LEFT JOIN utm u
          ON u.answerer_user_id = CAST(r.answerer_user_id AS BIGINT)
         AND u.month_date >= CAST(date_trunc('month', r.first_return_at) AS DATE)
         AND u.month_date <= CAST(date_trunc('month', r.first_tag_answer_at + INTERVAL 365 DAY) AS DATE)
        GROUP BY ALL
        """
    ).df()
    entrant_panel["focal_answer_share_post_return_365d"] = entrant_panel["focal_answers_post_return_365d"] / entrant_panel[
        "total_answers_post_return_365d"
    ].replace({0: np.nan})
    entrant_panel["any_focal_accepted_current_post_return_365d"] = (
        entrant_panel["focal_accepted_current_post_return_365d"] > 0
    ).astype(float)
    entrant_panel["focal_accepted_share_post_return_365d"] = entrant_panel[
        "focal_accepted_current_post_return_365d"
    ] / entrant_panel["total_accepted_current_post_return_365d"].replace({0: np.nan})
    entrant_panel.to_csv(EMBEDDING_USER_PANEL_CSV, index=False)
    return entrant_panel


def build_role_embedding_panel(returners: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("returners", returners)
    role_entrant_panel = con.execute(
        f"""
        WITH roles AS (
            SELECT
                CAST(owner_user_id AS BIGINT) AS answerer_user_id,
                primary_tag,
                CAST(answer_created_at AS TIMESTAMP) AS answer_created_at,
                role
            FROM read_parquet('{ROLE_QUESTION_PANEL.as_posix()}')
        )
        SELECT
            r.entrant_row_id,
            r.primary_tag,
            r.entry_month,
            r.time_index,
            r.post_chatgpt,
            r.exposure_index,
            r.high_tag,
            r.high_post,
            r.exposure_post,
            COUNT(ro.role) AS role_total_post_return_365d,
            SUM(CASE WHEN ro.role = 'first_answer' THEN 1 ELSE 0 END) AS first_answer_count_post_return_365d,
            SUM(CASE WHEN ro.role = 'first_positive' THEN 1 ELSE 0 END) AS first_positive_count_post_return_365d,
            SUM(CASE WHEN ro.role = 'top_score' THEN 1 ELSE 0 END) AS top_score_count_post_return_365d,
            SUM(CASE WHEN ro.role = 'accepted_current' THEN 1 ELSE 0 END) AS accepted_current_count_post_return_365d
        FROM returners r
        LEFT JOIN roles ro
          ON ro.answerer_user_id = CAST(r.answerer_user_id AS BIGINT)
         AND ro.primary_tag = r.primary_tag
         AND ro.answer_created_at >= CAST(r.first_return_at AS TIMESTAMP)
         AND ro.answer_created_at <= CAST(r.first_tag_answer_at + INTERVAL 365 DAY AS TIMESTAMP)
        GROUP BY ALL
        """
    ).df()
    role_total = role_entrant_panel["role_total_post_return_365d"].replace({0: np.nan})
    role_entrant_panel["accepted_current_role_share_post_return_365d"] = role_entrant_panel[
        "accepted_current_count_post_return_365d"
    ] / role_total
    role_entrant_panel["any_accepted_current_role_post_return_365d"] = (
        role_entrant_panel["accepted_current_count_post_return_365d"] > 0
    ).astype(float)
    role_entrant_panel["any_top_score_role_post_return_365d"] = (
        role_entrant_panel["top_score_count_post_return_365d"] > 0
    ).astype(float)
    role_entrant_panel.to_csv(EMBEDDING_ROLE_PANEL_CSV, index=False)
    return role_entrant_panel


def summarize_to_tag_month(frame: pd.DataFrame, value_cols: list[str], weight_col_name: str) -> pd.DataFrame:
    return (
        frame.groupby(
            ["primary_tag", "entry_month", "time_index", "post_chatgpt", "exposure_index", "high_tag", "high_post", "exposure_post"],
            as_index=False,
        )
        .agg({**{col: "mean" for col in value_cols}, "entrant_row_id": "size"})
        .rename(columns={"entrant_row_id": weight_col_name})
    )


def run_results(user_panel: pd.DataFrame, role_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_cols = [
        "focal_answer_share_post_return_365d",
        "focal_active_months_post_return_365d",
        "other_tags_touched_post_return_365d",
        "focal_accepted_current_post_return_365d",
        "any_focal_accepted_current_post_return_365d",
        "focal_accepted_share_post_return_365d",
    ]
    role_cols = [
        "role_total_post_return_365d",
        "first_answer_count_post_return_365d",
        "first_positive_count_post_return_365d",
        "top_score_count_post_return_365d",
        "accepted_current_count_post_return_365d",
        "accepted_current_role_share_post_return_365d",
        "any_accepted_current_role_post_return_365d",
        "any_top_score_role_post_return_365d",
    ]

    user_tag_month = summarize_to_tag_month(user_panel, user_cols, "n_returners")
    role_tag_month = summarize_to_tag_month(role_panel, role_cols, "n_role_returners")

    user_rows: list[dict[str, object]] = []
    for outcome in user_cols:
        for family, term in [("binary", "high_post"), ("continuous", "exposure_post")]:
            formula = f"{outcome} ~ {term} + C(primary_tag):time_index + C(primary_tag) + C(entry_month)"
            model = fit_wls(user_tag_month, formula, "n_returners")
            user_rows.append(
                {
                    "panel": "user_embedding",
                    "family": family,
                    "outcome": outcome,
                    "term": term,
                    "coef": float(model.params.get(term, np.nan)),
                    "se": float(model.bse.get(term, np.nan)),
                    "pval": float(model.pvalues.get(term, np.nan)),
                    "nobs": int(model.nobs),
                    "mean_outcome": float(user_tag_month[outcome].mean()),
                }
            )

    role_rows: list[dict[str, object]] = []
    for outcome in role_cols:
        for family, term in [("binary", "high_post"), ("continuous", "exposure_post")]:
            formula = f"{outcome} ~ {term} + C(primary_tag):time_index + C(primary_tag) + C(entry_month)"
            model = fit_wls(role_tag_month, formula, "n_role_returners")
            role_rows.append(
                {
                    "panel": "role_embedding",
                    "family": family,
                    "outcome": outcome,
                    "term": term,
                    "coef": float(model.params.get(term, np.nan)),
                    "se": float(model.bse.get(term, np.nan)),
                    "pval": float(model.pvalues.get(term, np.nan)),
                    "nobs": int(model.nobs),
                    "mean_outcome": float(role_tag_month[outcome].mean()),
                }
            )

    user_results = pd.DataFrame(user_rows)
    role_results = pd.DataFrame(role_rows)
    user_results.to_csv(EMBEDDING_USER_RESULTS_CSV, index=False)
    role_results.to_csv(EMBEDDING_ROLE_RESULTS_CSV, index=False)
    return user_results, role_results


def format_result(results: pd.DataFrame, outcome: str, family: str = "continuous") -> str:
    row = results.loc[(results["outcome"] == outcome) & (results["family"] == family)].iloc[0]
    return f"coef `{row['coef']:.4f}`, p `{row['pval']:.4g}`"


def write_readout(user_results: pd.DataFrame, role_results: pd.DataFrame) -> None:
    text = f"""# Established Cross-Tag User-Level Embedding Readout

Date: `2026-04-05`

## Design

This build moves from tag-month share splits to a user-level embedding question.

Sample:

- returning `established_cross_tag` entrants only
- `eligible_365d == 1`
- `return_365d == 1`

The user-level layer asks whether returners become more locally embedded in the focal tag after they return, or whether they remain a narrow imported fallback pool.

The main user-level outcomes are:

- `focal_answer_share_post_return_365d`
- `focal_active_months_post_return_365d`
- `other_tags_touched_post_return_365d`
- `focal_accepted_current_post_return_365d`
- `any_focal_accepted_current_post_return_365d`
- `focal_accepted_share_post_return_365d`

The role layer asks whether these returners perform more certification-adjacent work after return inside the focal tag:

- `accepted_current_count_post_return_365d`
- `accepted_current_role_share_post_return_365d`
- `any_accepted_current_role_post_return_365d`
- `top_score_count_post_return_365d`

## User-Level Embedding Surface

- `focal_answer_share_post_return_365d`: {format_result(user_results, 'focal_answer_share_post_return_365d')}
- `focal_active_months_post_return_365d`: {format_result(user_results, 'focal_active_months_post_return_365d')}
- `other_tags_touched_post_return_365d`: {format_result(user_results, 'other_tags_touched_post_return_365d')}
- `focal_accepted_current_post_return_365d`: {format_result(user_results, 'focal_accepted_current_post_return_365d')}
- `any_focal_accepted_current_post_return_365d`: {format_result(user_results, 'any_focal_accepted_current_post_return_365d')}
- `focal_accepted_share_post_return_365d`: {format_result(user_results, 'focal_accepted_share_post_return_365d')}

## Role / Certification-Adjacent Surface

- `accepted_current_count_post_return_365d`: {format_result(role_results, 'accepted_current_count_post_return_365d')}
- `accepted_current_role_share_post_return_365d`: {format_result(role_results, 'accepted_current_role_share_post_return_365d')}
- `any_accepted_current_role_post_return_365d`: {format_result(role_results, 'any_accepted_current_role_post_return_365d')}
- `top_score_count_post_return_365d`: {format_result(role_results, 'top_score_count_post_return_365d')}
- `any_top_score_role_post_return_365d`: {format_result(role_results, 'any_top_score_role_post_return_365d')}
- `first_answer_count_post_return_365d`: {format_result(role_results, 'first_answer_count_post_return_365d')}

## Safe Read

This is the first clean user-level attempt to answer the assimilation question rather than another share split.

Positive embedding evidence would mean:

- more focal-tag concentration after return
- more focal active months
- fewer other tags touched
- more focal accepted-current or certification-adjacent participation

Negative embedding evidence would mean:

- diffuse activity across tags
- little growth in focal accepted-current participation
- a returner pool that remains narrow and fallback-like rather than becoming a local bench

This layer should still be read as bounded observational triangulation rather than proof of local assimilation.
"""
    READOUT_MD.write_text(text, encoding="utf-8")


def main() -> None:
    returners = load_returning_established()
    user_panel = build_user_embedding_panel(returners)
    role_panel = build_role_embedding_panel(returners)
    user_results, role_results = run_results(user_panel, role_panel)
    write_readout(user_results, role_results)
    print(f"Built {READOUT_MD.name}")


if __name__ == "__main__":
    main()
