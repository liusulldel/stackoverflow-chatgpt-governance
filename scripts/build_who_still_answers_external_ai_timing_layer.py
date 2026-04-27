from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"

USER_AGENT = "CodexResearchBot/1.0 (liusully@gmail.com)"
PAGEVIEW_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia.org/all-access/all-agents/ChatGPT/monthly/20221101/20251231"
)
SEED = 42
N_PERMUTATIONS = 5000


def fetch_chatgpt_pageviews() -> pd.DataFrame:
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(PAGEVIEW_URL, headers=headers, timeout=30)
    response.raise_for_status()
    items = response.json()["items"]
    monthly = pd.DataFrame(
        {
            "month_id": [f"{x['timestamp'][:4]}-{x['timestamp'][4:6]}" for x in items],
            "chatgpt_views": [x["views"] for x in items],
        }
    )
    full_months = pd.DataFrame(
        {"month_id": pd.period_range("2020-01", "2025-12", freq="M").astype(str)}
    )
    out = full_months.merge(monthly, on="month_id", how="left").fillna({"chatgpt_views": 0})
    out["chatgpt_log_views"] = np.log1p(out["chatgpt_views"])
    out["chatgpt_z"] = (
        out["chatgpt_log_views"] - out["chatgpt_log_views"].mean()
    ) / out["chatgpt_log_views"].std(ddof=0)
    return out


def two_way_cluster_summary(
    formula: str, data: pd.DataFrame, weight_col: str, term: str
) -> tuple[float, float, float]:
    model = smf.wls(formula, data=data, weights=data[weight_col]).fit()
    used = data.loc[model.model.data.row_labels]
    tag_codes = used["primary_tag"].astype("category").cat.codes
    month_codes = used["month_id"].astype("category").cat.codes
    cov = cov_cluster_2groups(model, tag_codes, month_codes)[0]
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    idx = model.model.exog_names.index(term)
    coef = float(model.params.iloc[idx])
    se_term = float(se[idx])
    pval = float(2 * stats.norm.sf(abs(coef / se_term))) if se_term > 0 else np.nan
    return coef, se_term, pval


def randomization_p_value(
    data: pd.DataFrame, outcome: str, weight_col: str, term_name: str
) -> tuple[float, float]:
    # Frisch-Waugh-Lovell implementation so we can cheaply reshuffle tag-level exposure values.
    month_d = pd.get_dummies(data["month_id"], drop_first=True)
    tag_d = pd.get_dummies(data["primary_tag"], drop_first=True)
    trend_parts = []
    time_index = data["time_index"].to_numpy(dtype=float)
    for col in tag_d.columns:
        trend_parts.append(tag_d[col].to_numpy(dtype=float) * time_index)
    trend = np.column_stack(trend_parts) if trend_parts else np.empty((len(data), 0))
    controls = np.column_stack(
        [
            np.ones(len(data)),
            data["residual_queue_complexity_index_mean"].to_numpy(dtype=float),
            tag_d.to_numpy(dtype=float),
            month_d.to_numpy(dtype=float),
            trend,
        ]
    )

    weights = np.sqrt(data[weight_col].to_numpy(dtype=float))
    cw = controls * weights[:, None]

    y = data[outcome].to_numpy(dtype=float)
    yw = y * weights
    beta_y = np.linalg.lstsq(cw, yw, rcond=None)[0]
    ry = yw - cw.dot(beta_y)

    exposure_map = data.groupby("primary_tag")["exposure_index"].first()
    tags = exposure_map.index.to_list()
    exposure_values = exposure_map.to_numpy(dtype=float)
    chatgpt_z = data["chatgpt_z"].to_numpy(dtype=float)

    def partialled_coef(exposure_by_tag: dict[str, float]) -> float:
        x = data["primary_tag"].map(exposure_by_tag).to_numpy(dtype=float) * chatgpt_z
        xw = x * weights
        beta_x = np.linalg.lstsq(cw, xw, rcond=None)[0]
        rx = xw - cw.dot(beta_x)
        return float((rx @ ry) / (rx @ rx))

    actual_map = exposure_map.to_dict()
    actual_coef = partialled_coef(actual_map)

    rng = np.random.default_rng(SEED)
    extreme = 0
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(exposure_values)
        perm_map = dict(zip(tags, shuffled))
        perm_coef = partialled_coef(perm_map)
        if abs(perm_coef) >= abs(actual_coef):
            extreme += 1
    perm_p = (extreme + 1) / (N_PERMUTATIONS + 1)
    return actual_coef, perm_p


def build_figure(bridge: pd.DataFrame, pageviews: pd.DataFrame, out_path: Path) -> None:
    tag_map = (
        pd.read_csv(PROCESSED / "who_still_answers_answer_role_tag_month_panel.csv", usecols=["primary_tag", "high_tag"])
        .drop_duplicates()
    )
    plot_df = bridge.merge(tag_map, on="primary_tag", how="left")
    weighted_gap = (
        plot_df.groupby(["month_id", "high_tag"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "gap_mean": np.average(
                        g["recent_gap_first_vs_accepted"], weights=g["accepted_vote_30d_denom"]
                    )
                }
            )
        )
        .reset_index(drop=True)
    )
    weighted_gap["month_dt"] = pd.to_datetime(weighted_gap["month_id"] + "-01")
    pageviews = pageviews.copy()
    pageviews["month_dt"] = pd.to_datetime(pageviews["month_id"] + "-01")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(pageviews["month_dt"], pageviews["chatgpt_views"], color="#1f77b4", linewidth=2)
    axes[0].set_ylabel("ChatGPT wiki views")
    axes[0].set_title("External AI salience and answer-supply gap timing")
    axes[0].axvline(pd.Timestamp("2022-11-30"), color="black", linestyle="--", linewidth=1)
    axes[0].axvline(pd.Timestamp("2022-12-05"), color="gray", linestyle=":", linewidth=1)

    for high_tag, label, color in [(1, "High exposure tags", "#d62728"), (0, "Low exposure tags", "#2ca02c")]:
        sub = weighted_gap[weighted_gap["high_tag"] == high_tag]
        axes[1].plot(sub["month_dt"], sub["gap_mean"], label=label, linewidth=2, color=color)
    axes[1].axvline(pd.Timestamp("2022-11-30"), color="black", linestyle="--", linewidth=1)
    axes[1].axvline(pd.Timestamp("2022-12-05"), color="gray", linestyle=":", linewidth=1)
    axes[1].set_ylabel("Recent first vs accepted gap")
    axes[1].legend(frameon=False)
    axes[1].set_xlabel("Month")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    pageviews = fetch_chatgpt_pageviews()
    pageviews.to_csv(PROCESSED / "who_still_answers_external_ai_pageviews.csv", index=False)

    bridge = pd.read_csv(PROCESSED / "who_still_answers_infrastructure_bridge_panel.csv").merge(
        pageviews[["month_id", "chatgpt_z", "chatgpt_views"]], on="month_id", how="left"
    )

    specs = [
        {
            "family": "external_timing",
            "outcome": "recent_gap_first_vs_accepted",
            "weight_col": "accepted_vote_30d_denom",
            "promoted": True,
        },
        {
            "family": "external_timing",
            "outcome": "first_positive_answer_latency_mean",
            "weight_col": "first_positive_answer_latency_denom",
            "promoted": True,
        },
        {
            "family": "external_timing",
            "outcome": "accepted_cond_any_answer_30d_rate",
            "weight_col": "accepted_cond_any_answer_30d_denom",
            "promoted": True,
        },
        {
            "family": "external_timing",
            "outcome": "accepted_vote_30d_rate",
            "weight_col": "accepted_vote_30d_denom",
            "promoted": False,
        },
        {
            "family": "external_timing",
            "outcome": "any_answer_7d_rate",
            "weight_col": "any_answer_7d_denom",
            "promoted": False,
        },
        {
            "family": "external_timing",
            "outcome": "first_answer_1d_rate_closure",
            "weight_col": "first_answer_1d_denom_closure",
            "promoted": False,
        },
    ]

    results = []
    formula = (
        "{outcome} ~ exposure_index:chatgpt_z + residual_queue_complexity_index_mean + "
        "C(primary_tag) + C(month_id) + C(primary_tag):time_index"
    )
    for spec in specs:
        cols = [
            spec["outcome"],
            "exposure_index",
            "chatgpt_z",
            "residual_queue_complexity_index_mean",
            "primary_tag",
            "month_id",
            "time_index",
            spec["weight_col"],
        ]
        data = bridge[cols].dropna().copy()
        coef, cluster_se, cluster_p = two_way_cluster_summary(
            formula.format(outcome=spec["outcome"]),
            data,
            spec["weight_col"],
            "exposure_index:chatgpt_z",
        )
        perm_coef, perm_p = randomization_p_value(
            data,
            spec["outcome"],
            spec["weight_col"],
            "exposure_index:chatgpt_z",
        )
        results.append(
            {
                "family": spec["family"],
                "outcome": spec["outcome"],
                "promoted": int(spec["promoted"]),
                "coef": coef,
                "cluster_se": cluster_se,
                "cluster_pval": cluster_p,
                "permutation_coef": perm_coef,
                "permutation_pval": perm_p,
                "nobs": len(data),
                "n_tags": data["primary_tag"].nunique(),
                "n_months": data["month_id"].nunique(),
            }
        )

    results_df = pd.DataFrame(results).sort_values(["promoted", "cluster_pval"], ascending=[False, True])
    results_df.to_csv(PROCESSED / "who_still_answers_external_ai_timing_results.csv", index=False)

    build_figure(bridge, pageviews, FIGURES / "who_still_answers_external_ai_timing_layer.png")

    exposure_validation = json.loads((PROCESSED / "question_level_exposure_model_validation.json").read_text())
    summary = {
        "external_ai_timing_promoted": results_df.loc[results_df["promoted"] == 1].to_dict(orient="records"),
        "exposure_validation": {
            "api_calls": exposure_validation["api_calls"],
            "audited_cumulative_calls": exposure_validation["audited_cumulative_calls"],
            "n_labeled_questions": exposure_validation["n_labeled_questions"],
            "cv_r2": exposure_validation["cv_r2"],
            "cv_corr": exposure_validation["cv_corr"],
            "corr_with_legacy_exposure_index": exposure_validation["corr_with_legacy_exposure_index"],
            "top_quintile_validation": {
                key: exposure_validation["validation"][key]
                for key in ["first_answer_1d", "first_answer_7d", "accepted_30d", "predicted_exposure_score"]
            },
        },
    }
    (PROCESSED / "who_still_answers_external_ai_timing_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
