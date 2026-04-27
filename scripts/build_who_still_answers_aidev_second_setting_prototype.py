from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(r"D:\AI alignment\projects\stackoverflow_chatgpt_governance")
PROCESSED = ROOT / "processed"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"

PR_PANEL_PARQUET = PROCESSED / "who_still_answers_aidev_pr_panel.parquet"
AGENT_SUMMARY_CSV = PROCESSED / "who_still_answers_aidev_agent_summary.csv"
FE_RESULTS_CSV = PROCESSED / "who_still_answers_aidev_fe_results.csv"
STRICT_FE_RESULTS_CSV = PROCESSED / "who_still_answers_aidev_strict_overlap_fe_results.csv"
SUMMARY_JSON = PROCESSED / "who_still_answers_aidev_summary.json"

FIGURE_PATH = FIGURES / "who_still_answers_aidev_review_certification.png"
READOUT_PATH = PAPER / "who_still_answers_aidev_second_setting_prototype_readout_2026-04-04.md"


def load_remote_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    ai = pd.read_parquet(
        "hf://datasets/hao-li/AIDev/all_pull_request.parquet",
        columns=["id", "agent", "created_at", "merged_at", "state", "repo_url"],
    ).copy()
    human = pd.read_parquet(
        "hf://datasets/hao-li/AIDev/human_pull_request.parquet",
        columns=["id", "created_at", "merged_at", "state", "repo_url"],
    ).copy()
    human["agent"] = "Human"
    reviews = pd.read_parquet(
        "hf://datasets/hao-li/AIDev/pr_reviews.parquet",
        columns=["pr_id", "state", "submitted_at"],
    ).copy()
    return pd.concat([ai, human], ignore_index=True), reviews


def prepare_panel(prs: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    prs = prs.rename(columns={"id": "pr_id"}).copy()
    prs["created_at"] = pd.to_datetime(prs["created_at"], errors="coerce", utc=True)
    prs["merged_at"] = pd.to_datetime(prs["merged_at"], errors="coerce", utc=True)
    prs["month_id"] = prs["created_at"].dt.to_period("M").astype(str)
    prs["is_ai"] = (prs["agent"] != "Human").astype(int)

    overlap_repos = set(prs.loc[prs["is_ai"] == 1, "repo_url"].dropna()) & set(
        prs.loc[prs["is_ai"] == 0, "repo_url"].dropna()
    )
    prs = prs.loc[prs["repo_url"].isin(overlap_repos)].copy()

    reviews["submitted_at"] = pd.to_datetime(reviews["submitted_at"], errors="coerce", utc=True)

    approved = (
        reviews.loc[reviews["state"] == "APPROVED"]
        .groupby("pr_id", as_index=False)["submitted_at"]
        .min()
        .rename(columns={"submitted_at": "first_approved_at"})
    )
    changes_requested = (
        reviews.loc[reviews["state"] == "CHANGES_REQUESTED"]
        .groupby("pr_id", as_index=False)["submitted_at"]
        .min()
        .rename(columns={"submitted_at": "first_changes_requested_at"})
    )
    review_panel = (
        reviews.groupby("pr_id", as_index=False)
        .agg(first_review_at=("submitted_at", "min"), review_count=("submitted_at", "size"))
        .merge(approved, on="pr_id", how="left")
        .merge(changes_requested, on="pr_id", how="left")
    )

    panel = prs.merge(review_panel, on="pr_id", how="left")
    panel["overlap_repo_month"] = (
        panel.groupby(["repo_url", "month_id"])["is_ai"].transform("nunique").eq(2).astype(int)
    )

    def hours_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
        return (later - earlier).dt.total_seconds() / 3600.0

    panel["merged"] = panel["merged_at"].notna().astype(int)
    panel["first_review_hours"] = hours_between(panel["first_review_at"], panel["created_at"])
    panel["approved_hours"] = hours_between(panel["first_approved_at"], panel["created_at"])
    panel["changes_requested_hours"] = hours_between(panel["first_changes_requested_at"], panel["created_at"])
    panel["merged_hours"] = hours_between(panel["merged_at"], panel["created_at"])

    panel["first_review_7d"] = panel["first_review_hours"].le(24 * 7).fillna(False).astype(int)
    panel["approved_30d"] = panel["approved_hours"].le(24 * 30).fillna(False).astype(int)
    panel["changes_requested_30d"] = panel["changes_requested_hours"].le(24 * 30).fillna(False).astype(int)
    panel["merged_30d"] = panel["merged_hours"].le(24 * 30).fillna(False).astype(int)
    panel["log_first_review_hours"] = np.log1p(panel["first_review_hours"])
    return panel


def fit_fe_results(panel: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("first_review_7d", None),
        ("approved_30d", None),
        ("changes_requested_30d", None),
        ("merged_30d", None),
        ("log_first_review_hours", panel["first_review_hours"].notna()),
    ]
    rows = []
    for sample_name, sample_filter in [
        ("overlap_repos", None),
        ("strict_overlap_repo_month", panel["overlap_repo_month"] == 1),
    ]:
        sample = panel.copy()
        if sample_filter is not None:
            sample = sample.loc[sample_filter].copy()
        for outcome, extra_filter in specs:
            frame = sample.copy()
            if extra_filter is not None:
                frame = frame.loc[extra_filter.loc[frame.index]].copy()
            if frame.empty:
                continue
            model = smf.ols(f"{outcome} ~ is_ai + C(repo_url) + C(month_id)", data=frame).fit(
                cov_type="cluster", cov_kwds={"groups": frame["repo_url"]}
            )
            coef = float(model.params["is_ai"])
            se = float(model.bse["is_ai"])
            pval = float(model.pvalues["is_ai"])
            if not np.isfinite(coef) or not np.isfinite(se) or not np.isfinite(pval) or abs(coef) > 1e6:
                continue
            rows.append(
                {
                    "sample": sample_name,
                    "outcome": outcome,
                    "coef_is_ai": coef,
                    "se_is_ai": se,
                    "pval_is_ai": pval,
                    "nobs": int(model.nobs),
                    "n_repos": int(frame["repo_url"].nunique()),
                    "n_months": int(frame["month_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def build_agent_summary(panel: pd.DataFrame) -> pd.DataFrame:
    summary = (
        panel.groupby("agent", as_index=False)
        .agg(
            prs=("pr_id", "size"),
            repos=("repo_url", "nunique"),
            first_review_7d=("first_review_7d", "mean"),
            approved_30d=("approved_30d", "mean"),
            changes_requested_30d=("changes_requested_30d", "mean"),
            merged_30d=("merged_30d", "mean"),
            median_first_review_hours=("first_review_hours", "median"),
        )
        .sort_values("prs", ascending=False)
        .reset_index(drop=True)
    )
    summary["review_cert_gap"] = summary["first_review_7d"] - summary["approved_30d"]
    return summary


def build_figure(agent_summary: pd.DataFrame) -> None:
    plot_df = agent_summary.loc[
        agent_summary["agent"].isin(["Human", "OpenAI_Codex", "Copilot", "Cursor", "Devin", "Claude_Code"])
    ].copy()
    plot_df = plot_df.sort_values("review_cert_gap", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(plot_df))
    ax.bar(x - 0.18, plot_df["first_review_7d"], width=0.36, label="First review within 7d", color="#1f77b4")
    ax.bar(x + 0.18, plot_df["approved_30d"], width=0.36, label="Approved within 30d", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["agent"], rotation=20, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("AIDev prototype: review arrival vs certification")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def write_readout(agent_summary: pd.DataFrame, fe_results: pd.DataFrame) -> None:
    main = fe_results.loc[fe_results["sample"] == "overlap_repos"].copy()
    strict = fe_results.loc[fe_results["sample"] == "strict_overlap_repo_month"].copy()
    lines = [
        "# Who Still Answers: AIDev Second-Setting Prototype",
        "",
        "Date: April 4, 2026",
        "",
        "## What This Prototype Is",
        "",
        "This prototype uses the public `AIDev` dataset as a direct-AI-use second setting.",
        "Unlike the main Stack Overflow setting, the AIDev records explicitly identify AI-authored pull requests by tool family (`OpenAI_Codex`, `Copilot`, `Cursor`, `Devin`, `Claude_Code`).",
        "",
        "## Sample Construction",
        "",
        f"- PR-level overlap-repo sample: `{int(agent_summary['prs'].sum()):,}` PRs",
        f"- agents in summary table: `{len(agent_summary)}`",
        "",
        "## Fixed-Effect Prototype Results",
        "",
    ]
    for _, row in main.iterrows():
        lines.append(
            f"- overlap-repo `{row['outcome']}` on `is_ai`: coef `{row['coef_is_ai']:.4f}`, clustered `p = {row['pval_is_ai']:.4f}`"
        )
    lines += ["", "## Strict Overlap Repo-Month Sensitivity", ""]
    for _, row in strict.iterrows():
        lines.append(
            f"- strict overlap-repo-month `{row['outcome']}` on `is_ai`: coef `{row['coef_is_ai']:.4f}`, clustered `p = {row['pval_is_ai']:.4f}`"
        )
    lines += [
        "",
        "## Safe Interpretation",
        "",
        "This prototype does not replicate the Stack Overflow design one-for-one.",
        "What it does add is a public technical collaboration setting with direct AI-use observation and public review/certification outcomes.",
        "If the AI-vs-human differences in review arrival and certification are directional and stable enough, AIDev can become the strongest current answer to the `second-setting is still bounded` critique.",
    ]
    READOUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    prs, reviews = load_remote_inputs()
    panel = prepare_panel(prs, reviews)
    panel.to_parquet(PR_PANEL_PARQUET, index=False)

    agent_summary = build_agent_summary(panel)
    agent_summary.to_csv(AGENT_SUMMARY_CSV, index=False)

    fe_results = fit_fe_results(panel)
    fe_results.loc[fe_results["sample"] == "overlap_repos"].to_csv(FE_RESULTS_CSV, index=False)
    fe_results.loc[fe_results["sample"] == "strict_overlap_repo_month"].to_csv(STRICT_FE_RESULTS_CSV, index=False)

    build_figure(agent_summary)
    write_readout(agent_summary, fe_results)

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "panel_rows": int(len(panel)),
                "overlap_repos": int(panel["repo_url"].nunique()),
                "strict_overlap_repo_month_rows": int(panel["overlap_repo_month"].sum()),
                "ai_share": float(panel["is_ai"].mean()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
