from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"

QUESTION_PARQUET = PROCESSED / "stackexchange_20251231_question_level_enriched.parquet"
BRIDGE_PANEL = PROCESSED / "who_still_answers_infrastructure_bridge_panel.csv"
EXTERNAL_PAGEVIEWS = PROCESSED / "who_still_answers_external_ai_pageviews.csv"

MONTHLY_TRACE_CSV = PROCESSED / "who_still_answers_internal_ai_title_trace_monthly.csv"
TAG_MONTH_TRACE_CSV = PROCESSED / "who_still_answers_internal_ai_title_trace_tag_month.csv"
TIMING_RESULTS_CSV = PROCESSED / "who_still_answers_internal_ai_timing_results.csv"
TIMING_SUMMARY_JSON = PROCESSED / "who_still_answers_internal_ai_timing_summary.json"

FIGURE_PATH = FIGURES / "who_still_answers_internal_ai_timing_trace.png"
READOUT_PATH = PAPER / "who_still_answers_internal_ai_timing_trace_readout_2026-04-04.md"

SEED = 42
N_PERMUTATIONS = 5000

AI_PATTERNS = {
    "chatgpt": r"\bchatgpt\b",
    "copilot": r"\b(?:github\s+)?copilot\b",
    "claude": r"\bclaude(?:\s+code|\s+ai)?\b|claude-code|claude code",
    "gemini": r"\b(?:google\s+)?gemini(?:\s+(?:ai|pro|flash|nano|1\.5|2(?:\.0)?))?\b",
    "deepseek": r"\bdeepseek\b",
    "openai": r"\bopenai\b",
    "anthropic": r"\banthropic\b",
    "bard": r"\bgoogle\s+bard\b|\bbard\s+ai\b",
    "llm": r"\blarge language model(?:s)?\b|\bllm(?:s)?\b",
    "gpt_family": r"\bgpt[- ]?(?:3|3\.5|4|4o|4\.1|4\.5|5|o1|o3|o4)\b|\bgenerative pre[- ]trained\b",
}


def load_question_trace() -> tuple[pd.DataFrame, pd.DataFrame]:
    question_df = pd.read_parquet(
        QUESTION_PARQUET,
        columns=["question_id", "month_id", "primary_tag", "high_tag", "exposure_index", "title"],
    ).copy()
    title_text = question_df["title"].fillna("")
    for label, pattern in AI_PATTERNS.items():
        question_df[f"hit_{label}"] = title_text.str.contains(pattern, case=False, regex=True)
    hit_cols = [f"hit_{label}" for label in AI_PATTERNS]
    question_df["internal_ai_title_hit"] = question_df[hit_cols].any(axis=1).astype(int)

    tag_month = (
        question_df.groupby(["primary_tag", "month_id"], as_index=False)
        .agg(
            questions=("question_id", "size"),
            ai_title_hits=("internal_ai_title_hit", "sum"),
            high_tag=("high_tag", "first"),
            exposure_index=("exposure_index", "first"),
            **{f"{label}_hits": (f"hit_{label}", "sum") for label in AI_PATTERNS},
        )
    )
    tag_month["internal_ai_title_rate"] = tag_month["ai_title_hits"] / tag_month["questions"]

    monthly = (
        tag_month.groupby("month_id", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "questions": int(g["questions"].sum()),
                    "ai_title_hits": int(g["ai_title_hits"].sum()),
                    "internal_ai_title_rate": float(g["ai_title_hits"].sum() / g["questions"].sum()),
                    "high_tag_rate": float(
                        g.loc[g["high_tag"] == 1, "ai_title_hits"].sum()
                        / g.loc[g["high_tag"] == 1, "questions"].sum()
                    ),
                    "low_tag_rate": float(
                        g.loc[g["high_tag"] == 0, "ai_title_hits"].sum()
                        / g.loc[g["high_tag"] == 0, "questions"].sum()
                    ),
                }
            )
        )
        .reset_index(drop=True)
        .sort_values("month_id")
        .reset_index(drop=True)
    )
    monthly["internal_ai_title_log_hits"] = np.log1p(monthly["ai_title_hits"])
    monthly["internal_ai_title_z"] = (
        monthly["internal_ai_title_rate"] - monthly["internal_ai_title_rate"].mean()
    ) / monthly["internal_ai_title_rate"].std(ddof=0)
    return monthly, tag_month


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


def randomization_p_value(data: pd.DataFrame, outcome: str, weight_col: str) -> tuple[float, float]:
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
    internal_z = data["internal_ai_title_z"].to_numpy(dtype=float)

    def partialled_coef(exposure_by_tag: dict[str, float]) -> float:
        x = data["primary_tag"].map(exposure_by_tag).to_numpy(dtype=float) * internal_z
        xw = x * weights
        beta_x = np.linalg.lstsq(cw, xw, rcond=None)[0]
        rx = xw - cw.dot(beta_x)
        return float((rx @ ry) / (rx @ rx))

    actual_coef = partialled_coef(exposure_map.to_dict())
    rng = np.random.default_rng(SEED)
    extreme = 0
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(exposure_values)
        perm_coef = partialled_coef(dict(zip(tags, shuffled)))
        if abs(perm_coef) >= abs(actual_coef):
            extreme += 1
    perm_p = (extreme + 1) / (N_PERMUTATIONS + 1)
    return actual_coef, perm_p


def build_figure(monthly: pd.DataFrame, bridge: pd.DataFrame, tag_month: pd.DataFrame) -> None:
    tag_flags = tag_month[["primary_tag", "high_tag"]].drop_duplicates()
    bridge = bridge.merge(tag_flags, on="primary_tag", how="left")
    gap_plot = (
        bridge.groupby(["month_id", "high_tag"], as_index=False)
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
    monthly = monthly.copy()
    monthly["month_dt"] = pd.to_datetime(monthly["month_id"] + "-01")
    gap_plot["month_dt"] = pd.to_datetime(gap_plot["month_id"] + "-01")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(monthly["month_dt"], monthly["internal_ai_title_rate"], color="#1f77b4", linewidth=2, label="All")
    axes[0].plot(monthly["month_dt"], monthly["high_tag_rate"], color="#d62728", linewidth=1.5, alpha=0.85, label="High tags")
    axes[0].plot(monthly["month_dt"], monthly["low_tag_rate"], color="#2ca02c", linewidth=1.5, alpha=0.85, label="Low tags")
    axes[0].axvline(pd.Timestamp("2022-11-30"), color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("AI-title mention rate")
    axes[0].set_title("Internal AI-title timing trace")
    axes[0].legend(frameon=False, ncol=3)

    for high_tag, label, color in [(1, "High exposure tags", "#d62728"), (0, "Low exposure tags", "#2ca02c")]:
        sub = gap_plot[gap_plot["high_tag"] == high_tag]
        axes[1].plot(sub["month_dt"], sub["gap_mean"], label=label, linewidth=2, color=color)
    axes[1].axvline(pd.Timestamp("2022-11-30"), color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Recent first vs accepted gap")
    axes[1].set_xlabel("Month")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def write_readout(results: pd.DataFrame, monthly: pd.DataFrame, corr_external: float) -> None:
    promoted = results.loc[results["promoted"] == 1].copy()
    lines = [
        "# Who Still Answers: Internal AI-Mention Timing Trace",
        "",
        "Date: April 4, 2026",
        "",
        "## Design",
        "",
        "This timing layer uses explicit AI-tool mentions in focal Stack Overflow question titles as a same-setting internal salience trace.",
        "The trace is conservative because it relies on question titles only, which sharply reduces false positives relative to a generic `AI` keyword search.",
        "",
        "## Descriptive Read",
        "",
        f"- total focal questions: `{int(monthly['questions'].sum()):,}`",
        f"- total AI-title hits: `{int(monthly['ai_title_hits'].sum()):,}`",
        f"- mean title-hit rate: `{monthly['internal_ai_title_rate'].mean():.4f}`",
        f"- correlation with external ChatGPT pageview z-score: `{corr_external:.3f}`",
        "",
        "## Promoted Timing Results",
        "",
    ]
    for _, row in promoted.iterrows():
        lines.append(
            f"- `{row['outcome']}`: coef `{row['coef']:.4f}`, clustered `p = {row['cluster_pval']:.4f}`, permutation `p = {row['permutation_pval']:.4f}`"
        )
    lines += [
        "",
        "## Safe Interpretation",
        "",
        "This trace does not turn the paper into a clean discontinuity design.",
        "It does show that the promoted bridge outcomes line up with a same-setting internal AI-discourse series rather than only an external salience series.",
        "That makes the bounded timing story materially stronger.",
    ]
    READOUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    monthly, tag_month = load_question_trace()
    monthly.to_csv(MONTHLY_TRACE_CSV, index=False)
    tag_month.to_csv(TAG_MONTH_TRACE_CSV, index=False)

    bridge = pd.read_csv(BRIDGE_PANEL).merge(
        monthly[["month_id", "internal_ai_title_z", "internal_ai_title_rate", "ai_title_hits"]],
        on="month_id",
        how="left",
    )
    external = pd.read_csv(EXTERNAL_PAGEVIEWS)
    corr_external = float(
        monthly.merge(external[["month_id", "chatgpt_z"]], on="month_id", how="left")[
            ["internal_ai_title_z", "chatgpt_z"]
        ].corr().iloc[0, 1]
    )

    specs = [
        {"outcome": "recent_gap_first_vs_accepted", "weight_col": "accepted_vote_30d_denom", "promoted": 1},
        {"outcome": "first_positive_answer_latency_mean", "weight_col": "first_positive_answer_latency_denom", "promoted": 1},
        {"outcome": "accepted_cond_any_answer_30d_rate", "weight_col": "accepted_cond_any_answer_30d_denom", "promoted": 1},
        {"outcome": "accepted_vote_30d_rate", "weight_col": "accepted_vote_30d_denom", "promoted": 0},
    ]
    formula = (
        "{outcome} ~ exposure_index:internal_ai_title_z + residual_queue_complexity_index_mean + "
        "C(primary_tag) + C(month_id) + C(primary_tag):time_index"
    )
    results = []
    for spec in specs:
        cols = [
            spec["outcome"],
            "exposure_index",
            "internal_ai_title_z",
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
            "exposure_index:internal_ai_title_z",
        )
        perm_coef, perm_p = randomization_p_value(data, spec["outcome"], spec["weight_col"])
        results.append(
            {
                "family": "internal_title_timing",
                "outcome": spec["outcome"],
                "promoted": spec["promoted"],
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

    results_df = pd.DataFrame(results)
    results_df.to_csv(TIMING_RESULTS_CSV, index=False)
    TIMING_SUMMARY_JSON.write_text(
        json.dumps(
            {
                "series_source": "question_title_mentions_only",
                "ai_patterns": list(AI_PATTERNS),
                "total_questions": int(monthly["questions"].sum()),
                "total_hits": int(monthly["ai_title_hits"].sum()),
                "overall_rate": float(monthly["internal_ai_title_rate"].mean()),
                "corr_with_external_chatgpt_z": corr_external,
                "peak_month": monthly.sort_values("internal_ai_title_rate", ascending=False).iloc[0]["month_id"],
                "peak_rate": float(monthly["internal_ai_title_rate"].max()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    build_figure(monthly, bridge, tag_month)
    write_readout(results_df, monthly, corr_external)


if __name__ == "__main__":
    main()
