from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


ROOT = Path(r"D:\AI alignment\projects\stackoverflow_chatgpt_governance")
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

SOURCE = PROCESSED / "question_level_exposure_fullsample_scores.parquet"
OUT_RESULTS = PROCESSED / "who_still_answers_question_level_identification_results.csv"
OUT_SUMMARY = PROCESSED / "who_still_answers_question_level_identification_summary.json"
OUT_MEMO = PAPER / "who_still_answers_question_level_identification_readout_2026-04-04.md"


def twoway_cluster(model, used: pd.DataFrame, term: str) -> tuple[float, float, float]:
    cov = cov_cluster_2groups(
        model,
        used["primary_tag"].astype("category").cat.codes,
        used["month_id"].astype("category").cat.codes,
    )[0]
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    idx = model.model.exog_names.index(term)
    coef = float(model.params.iloc[idx])
    se_term = float(se[idx])
    pval = float(2 * stats.norm.sf(abs(coef / se_term))) if se_term > 0 else np.nan
    return coef, se_term, pval


def main() -> None:
    df = pd.read_parquet(
        SOURCE,
        columns=[
            "primary_tag",
            "month_id",
            "post_chatgpt",
            "high_tag",
            "predicted_exposure_score",
            "predicted_exposure_within_tag_z",
            "exposure_quintile",
            "first_answer_1d",
            "accepted_30d",
        ],
    )
    df = df[df["primary_tag"].notna()].copy()
    df["fast_and_accepted"] = ((df["first_answer_1d"] == 1) & (df["accepted_30d"] == 1)).astype(int)

    share_panel = (
        df.groupby(["primary_tag", "month_id", "post_chatgpt", "high_tag"], as_index=False)
        .agg(
            n=("exposure_quintile", "size"),
            top_share=("exposure_quintile", lambda s: (s == 5).mean()),
            pred_mean=("predicted_exposure_within_tag_z", "mean"),
        )
    )

    outcome_panel = (
        df.groupby(
            ["primary_tag", "month_id", "post_chatgpt", "high_tag", "exposure_quintile"],
            as_index=False,
        )
        .agg(
            n=("first_answer_1d", "size"),
            exp_mean=("predicted_exposure_within_tag_z", "mean"),
            first_answer_1d_rate=("first_answer_1d", "mean"),
            accepted_30d_rate=("accepted_30d", "mean"),
            fast_and_accepted_rate=("fast_and_accepted", "mean"),
        )
    )
    outcome_panel["response_cert_gap"] = (
        outcome_panel["first_answer_1d_rate"] - outcome_panel["accepted_30d_rate"]
    )
    outcome_panel["accepted_given_fast_rate"] = np.where(
        outcome_panel["first_answer_1d_rate"] > 0,
        outcome_panel["fast_and_accepted_rate"] / outcome_panel["first_answer_1d_rate"],
        np.nan,
    )
    outcome_panel["top_quintile"] = (outcome_panel["exposure_quintile"] == 5).astype(int)

    rows = []

    # Residualization of the queue in legacy high-exposure domains.
    for outcome, term in [("top_share", "high_tag:post_chatgpt"), ("pred_mean", "high_tag:post_chatgpt")]:
        model = smf.wls(
            f"{outcome} ~ high_tag*post_chatgpt + C(primary_tag) + C(month_id)",
            data=share_panel,
            weights=share_panel["n"],
        ).fit()
        used = share_panel.loc[model.model.data.row_labels]
        coef, se, pval = twoway_cluster(model, used, term)
        rows.append(
            {
                "family": "queue_residualization",
                "sample": "all_tags",
                "outcome": outcome,
                "term": term,
                "coef": coef,
                "cluster_se": se,
                "cluster_pval": pval,
                "nobs": len(used),
            }
        )

    # Within-tag identification layer.
    for outcome in [
        "first_answer_1d_rate",
        "accepted_30d_rate",
        "response_cert_gap",
        "accepted_given_fast_rate",
    ]:
        data = outcome_panel.dropna(subset=[outcome]).copy()
        model = smf.wls(
            f"{outcome} ~ post_chatgpt*top_quintile + C(primary_tag) + C(month_id)",
            data=data,
            weights=data["n"],
        ).fit()
        used = data.loc[model.model.data.row_labels]
        coef, se, pval = twoway_cluster(model, used, "post_chatgpt:top_quintile")
        rows.append(
            {
                "family": "within_tag_quintile",
                "sample": "all_tags",
                "outcome": outcome,
                "term": "post_chatgpt:top_quintile",
                "coef": coef,
                "cluster_se": se,
                "cluster_pval": pval,
                "nobs": len(used),
            }
        )

    # Legacy high-exposure domains only: does within-tag exposure shift the response-certification gap?
    high_only = outcome_panel[outcome_panel["high_tag"] == 1].dropna(
        subset=["first_answer_1d_rate", "accepted_30d_rate", "response_cert_gap", "accepted_given_fast_rate"]
    )
    for outcome in [
        "first_answer_1d_rate",
        "accepted_30d_rate",
        "response_cert_gap",
        "accepted_given_fast_rate",
    ]:
        data = high_only.dropna(subset=[outcome]).copy()
        model = smf.wls(
            f"{outcome} ~ post_chatgpt*exp_mean + C(primary_tag) + C(month_id)",
            data=data,
            weights=data["n"],
        ).fit()
        used = data.loc[model.model.data.row_labels]
        coef, se, pval = twoway_cluster(model, used, "post_chatgpt:exp_mean")
        rows.append(
            {
                "family": "within_tag_continuous_high_tags",
                "sample": "high_tags_only",
                "outcome": outcome,
                "term": "post_chatgpt:exp_mean",
                "coef": coef,
                "cluster_se": se,
                "cluster_pval": pval,
                "nobs": len(used),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(OUT_RESULTS, index=False)

    summary = {
        "n_questions": int(len(df)),
        "n_tags": int(df["primary_tag"].nunique()),
        "share_panel_rows": int(len(share_panel)),
        "outcome_panel_rows": int(len(outcome_panel)),
        "results": results.to_dict(orient="records"),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def pick(family: str, outcome: str) -> dict:
        row = results[(results["family"] == family) & (results["outcome"] == outcome)].iloc[0]
        return row.to_dict()

    top_share = pick("queue_residualization", "top_share")
    pred_mean = pick("queue_residualization", "pred_mean")
    first1 = pick("within_tag_quintile", "first_answer_1d_rate")
    accepted30 = pick("within_tag_quintile", "accepted_30d_rate")
    gap = pick("within_tag_quintile", "response_cert_gap")
    cond = pick("within_tag_quintile", "accepted_given_fast_rate")
    high_gap = pick("within_tag_continuous_high_tags", "response_cert_gap")

    lines = [
        "# Who Still Answers: Question-Level Identification Layer",
        "",
        "Date: April 4, 2026",
        "",
        "## Purpose",
        "",
        "This layer upgrades the question-level exposure score from a generic validation appendix into a main-setting identification aid. The goal is to move beyond a pure domain-level proxy and ask two questions:",
        "",
        "1. Do legacy high-exposure domains retain a less AI-substitutable residual public queue after the transition?",
        "2. Within tags, do more AI-substitutable questions behave differently in the conversion from rapid response to later certified resolution?",
        "",
        "## Queue residualization read",
        "",
        f"- `top_share` in legacy high-exposure domains: coef `{top_share['coef']:.4f}`, clustered `p={top_share['cluster_pval']:.4f}`",
        f"- `pred_mean` in legacy high-exposure domains: coef `{pred_mean['coef']:.4f}`, clustered `p={pred_mean['cluster_pval']:.4f}`",
        "",
        "These estimates indicate that the average question remaining in the old high-exposure queue becomes less exposure-heavy after the transition. That is the cleanest residualization read available from the main setting.",
        "",
        "## Within-tag post x exposure-quintile read",
        "",
        f"- `first_answer_1d_rate`: coef `{first1['coef']:.4f}`, clustered `p={first1['cluster_pval']:.4g}`",
        f"- `accepted_30d_rate`: coef `{accepted30['coef']:.4f}`, clustered `p={accepted30['cluster_pval']:.4g}`",
        f"- `response_cert_gap`: coef `{gap['coef']:.4f}`, clustered `p={gap['cluster_pval']}`",
        f"- `accepted_given_fast_rate`: coef `{cond['coef']:.4f}`, clustered `p={cond['cluster_pval']:.4g}`",
        "",
        "The safe interpretation is not that higher-exposure questions become harder after the transition. It is almost the opposite: conditional on still being asked publicly, the higher-exposure questions that remain continue to convert relatively well from fast response into later certified closure. This strengthens the residualization story because it implies the public queue is becoming more concentrated in lower-exposure, more context-heavy questions.",
        "",
        "## High-tag-only continuous read",
        "",
        f"- In legacy high-exposure tags, `post x exposure` on `response_cert_gap`: coef `{high_gap['coef']:.4f}`, clustered `p={high_gap['cluster_pval']:.4f}`",
        "",
        "This says that inside the legacy high-exposure domains, months with a more exposure-heavy residual queue exhibit a larger response-certification gap after the transition. That is consistent with the paper's main bridge logic, but it should still be written as a bounded supportive layer rather than a new headline result.",
        "",
        "## Read",
        "",
        "This is the strongest available answer to the remaining `proxy` criticism inside the main setting. The paper still does not observe direct AI use, but it now shows that the residual public queue becomes measurably less exposure-heavy and that within-tag exposure gradients map into later response-to-certification conversion. That is materially stronger than a frozen domain split alone.",
    ]
    OUT_MEMO.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
