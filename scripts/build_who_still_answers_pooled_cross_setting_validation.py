from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

OUT_PANEL = PROCESSED / "who_still_answers_pooled_cross_setting_panel.csv"
OUT_RESULTS = PROCESSED / "who_still_answers_pooled_cross_setting_results.csv"
OUT_SUMMARY = PROCESSED / "who_still_answers_pooled_cross_setting_summary.json"
OUT_MEMO = PAPER / "who_still_answers_pooled_cross_setting_readout_2026-04-04.md"


def two_way_cluster(model, used: pd.DataFrame, term: str) -> tuple[float, float, float]:
    cov = cov_cluster_2groups(
        model,
        used["cluster_id"].astype("category").cat.codes,
        used["month_id"].astype("category").cat.codes,
    )[0]
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    idx = model.model.exog_names.index(term)
    coef = float(model.params.iloc[idx])
    se_term = float(se[idx])
    pval = float(2 * stats.norm.sf(abs(coef / se_term))) if se_term > 0 else np.nan
    return coef, se_term, pval


def load_main() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "who_still_answers_infrastructure_bridge_panel.csv")
    out = pd.DataFrame(
        {
            "setting": "stackoverflow",
            "primary_tag": df["primary_tag"],
            "month_id": df["month_id"],
            "time_index": df["time_index"],
            "n_questions": df["accepted_cond_any_answer_30d_denom"].fillna(df["first_answer_1d_denom_closure"]),
            "exposure_index": df["exposure_index"],
            "predicted_exposure_mean": np.nan,
            "residual_queue_complexity_index_mean": df["residual_queue_complexity_index_mean"],
            "first_answer_1d_rate": df["first_answer_1d_rate_closure"],
            "accepted_30d_rate": df["accepted_cond_any_answer_30d_rate"],
            "response_cert_gap": df["first_answer_1d_rate_closure"] - df["accepted_cond_any_answer_30d_rate"],
            "post_chatgpt": ((pd.to_datetime(df["month_id"] + "-01") >= pd.Timestamp("2022-11-30"))).astype(int),
            "chatgpt_z": np.nan,
        }
    )
    pageviews = pd.read_csv(PROCESSED / "who_still_answers_external_ai_pageviews.csv")[["month_id", "chatgpt_z"]]
    out = out.drop(columns=["chatgpt_z"]).merge(pageviews, on="month_id", how="left")
    return out


def load_second(path: Path, setting: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "setting": setting,
            "primary_tag": df["primary_tag"],
            "month_id": df["month_id"],
            "time_index": df["time_index"],
            "n_questions": df["n_questions"],
            "exposure_index": df["exposure_index"],
            "predicted_exposure_mean": df["predicted_exposure_mean"],
            "residual_queue_complexity_index_mean": df["residual_queue_complexity_index_mean"],
            "first_answer_1d_rate": df["first_answer_1d_rate"],
            "accepted_30d_rate": df["accepted_30d_rate"],
            "response_cert_gap": df["response_cert_gap"],
            "post_chatgpt": df["post_chatgpt"],
            "chatgpt_z": df["chatgpt_z"],
        }
    )
    return out


def run_spec(data: pd.DataFrame, outcome: str, rhs_term: str) -> dict:
    formula = (
        f"{outcome} ~ {rhs_term} + residual_queue_complexity_index_mean + "
        "C(cluster_id) + C(month_id) + C(cluster_id):time_index"
    )
    model = smf.wls(formula, data=data, weights=data["n_questions"]).fit()
    used = data.loc[model.model.data.row_labels]
    coef, se, pval = two_way_cluster(model, used, rhs_term)
    return {
        "outcome": outcome,
        "term": rhs_term,
        "coef": coef,
        "cluster_se": se,
        "cluster_pval": pval,
        "nobs": len(used),
        "n_clusters": used["cluster_id"].nunique(),
        "n_settings": used["setting"].nunique(),
    }


def main() -> None:
    stack = load_main()
    superuser = load_second(PROCESSED / "superuser_second_setting_tag_month_panel.csv", "superuser")
    unix = load_second(PROCESSED / "unix_second_setting_tag_month_panel.csv", "unix")

    pooled = pd.concat([stack, superuser, unix], ignore_index=True)
    pooled["cluster_id"] = pooled["setting"] + "::" + pooled["primary_tag"]
    pooled["high_post"] = pooled["exposure_index"] * pooled["post_chatgpt"]
    pooled["exposure_chatgpt"] = pooled["exposure_index"] * pooled["chatgpt_z"]
    pooled.to_csv(OUT_PANEL, index=False)

    results = []
    for outcome in ["response_cert_gap", "accepted_30d_rate", "first_answer_1d_rate"]:
        results.append(run_spec(pooled.dropna(subset=[outcome, "exposure_index", "post_chatgpt"]), outcome, "exposure_index:post_chatgpt"))
        results.append(run_spec(pooled.dropna(subset=[outcome, "exposure_index", "chatgpt_z"]), outcome, "exposure_index:chatgpt_z"))

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_RESULTS, index=False)

    summary = {
        "n_rows": int(len(pooled)),
        "n_clusters": int(pooled["cluster_id"].nunique()),
        "n_settings": int(pooled["setting"].nunique()),
        "settings": sorted(pooled["setting"].unique().tolist()),
        "promoted_read": results_df.to_dict(orient="records"),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Who Still Answers: Pooled Cross-Setting Validation",
        "",
        "Date: April 4, 2026",
        "",
        "## Purpose",
        "",
        "This pooled layer combines Stack Overflow, Super User, and Unix tag-month panels to address the `few-cluster` criticism more directly. Rather than treating each second setting as an isolated appendix note, the pooled layer asks whether more exposed domains across multiple public technical platforms show weaker conversion from early response into later certified resolution.",
        "",
        "## Scope",
        "",
        f"- Pooled rows: `{summary['n_rows']}`",
        f"- Setting-tag clusters: `{summary['n_clusters']}`",
        f"- Settings: `{', '.join(summary['settings'])}`",
        "",
        "## Main pooled results",
        "",
    ]
    for row in results_df.to_dict(orient="records"):
        lines.append(
            f"- `{row['outcome']}` on `{row['term']}`: coef `{row['coef']:.4f}`, clustered `p={row['cluster_pval']:.4f}`, clusters `{row['n_clusters']}`"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "Use this layer as a cross-setting validation and cluster-relief device, not as a replacement for the main Stack Overflow mechanism stack. If the pooled response-certification gap remains exposure-graded across settings, that materially weakens the 'only 16 clusters on one site' objection even if each auxiliary setting is individually bounded.",
            "",
        ]
    )
    OUT_MEMO.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
