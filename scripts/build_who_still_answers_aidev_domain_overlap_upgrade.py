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

PANEL_PARQUET = PROCESSED / "who_still_answers_aidev_domain_overlap_panel.parquet"
RESULTS_CSV = PROCESSED / "who_still_answers_aidev_domain_overlap_results.csv"
SAMPLE_COUNTS_CSV = PROCESSED / "who_still_answers_aidev_domain_overlap_sample_counts.csv"
SUMMARY_JSON = PROCESSED / "who_still_answers_aidev_domain_overlap_summary.json"
FIGURE_PATH = FIGURES / "who_still_answers_aidev_domain_overlap_upgrade.png"
READOUT_PATH = PAPER / "who_still_answers_aidev_domain_overlap_upgrade_2026-04-04.md"


def load_remote_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    comments = pd.read_parquet(
        "hf://datasets/hao-li/AIDev/pr_comments.parquet",
        columns=["pr_id", "created_at"],
    ).copy()
    repo = pd.read_parquet(
        "hf://datasets/hao-li/AIDev/all_repository.parquet",
        columns=["url", "full_name", "language", "stars", "forks"],
    ).copy()
    ai_task = pd.read_parquet(
        "hf://datasets/hao-li/AIDev/pr_task_type.parquet",
        columns=["id", "type"],
    ).copy()
    human_task = pd.read_parquet(
        "hf://datasets/hao-li/AIDev/human_pr_task_type.parquet",
        columns=["id", "type"],
    ).copy()
    task = pd.concat([ai_task, human_task], ignore_index=True).drop_duplicates(subset=["id"])
    return pd.concat([ai, human], ignore_index=True), reviews, comments, repo, task


def assign_domain_family(language: str | None, full_name: str | None) -> str | None:
    language = "" if pd.isna(language) else str(language).strip()
    full_name = "" if pd.isna(full_name) else str(full_name).lower()
    if language in {"Python", "Jupyter Notebook"}:
        return "python_data"
    if language in {"JavaScript", "TypeScript", "HTML", "CSS"}:
        return "javascript_web"
    if language in {"Java", "Kotlin"} or "android" in full_name:
        return "android_mobile"
    if language in {"Go", "Shell", "Dockerfile"} or any(
        token in full_name for token in ["docker", "kubernetes", "k8s", "linux", "bash", "infra"]
    ):
        return "infra_cloud"
    if "sql" in full_name or any(token in full_name for token in ["postgres", "mysql", "database"]):
        return "sql_data"
    return None


def prepare_panel(
    prs: pd.DataFrame, reviews: pd.DataFrame, comments: pd.DataFrame, repo: pd.DataFrame, task: pd.DataFrame
) -> pd.DataFrame:
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
    comments["created_at"] = pd.to_datetime(comments["created_at"], errors="coerce", utc=True)

    first_review = (
        reviews.groupby("pr_id", as_index=False)["submitted_at"]
        .min()
        .rename(columns={"submitted_at": "first_review_at"})
    )
    approved = (
        reviews.loc[reviews["state"] == "APPROVED"]
        .groupby("pr_id", as_index=False)["submitted_at"]
        .min()
        .rename(columns={"submitted_at": "first_approved_at"})
    )
    first_comment = (
        comments.groupby("pr_id", as_index=False)["created_at"]
        .min()
        .rename(columns={"created_at": "first_comment_at"})
    )

    panel = (
        prs.merge(first_review, on="pr_id", how="left")
        .merge(approved, on="pr_id", how="left")
        .merge(first_comment, on="pr_id", how="left")
        .merge(repo, left_on="repo_url", right_on="url", how="left")
        .merge(task.rename(columns={"id": "pr_id", "type": "task_type"}), on="pr_id", how="left")
    )

    panel["domain_family"] = [
        assign_domain_family(language, full_name)
        for language, full_name in zip(panel["language"], panel["full_name"])
    ]
    panel["domain_overlap"] = panel["domain_family"].notna().astype(int)
    panel["fix_like"] = panel["task_type"].isin(["fix", "perf", "refactor"]).astype(int)
    panel["overlap_repo_month"] = (
        panel.groupby(["repo_url", "month_id"])["is_ai"].transform("nunique").eq(2).astype(int)
    )

    panel["first_feedback_at"] = panel[["first_review_at", "first_comment_at"]].min(axis=1)
    panel["first_feedback_hours"] = (panel["first_feedback_at"] - panel["created_at"]).dt.total_seconds() / 3600.0
    panel["approved_hours"] = (panel["first_approved_at"] - panel["created_at"]).dt.total_seconds() / 3600.0
    panel["merged_hours"] = (panel["merged_at"] - panel["created_at"]).dt.total_seconds() / 3600.0

    panel["first_feedback_7d"] = panel["first_feedback_hours"].le(24 * 7).fillna(False).astype(int)
    panel["approved_30d"] = panel["approved_hours"].le(24 * 30).fillna(False).astype(int)
    panel["merged_30d"] = panel["merged_hours"].le(24 * 30).fillna(False).astype(int)
    panel["review_merge_gap"] = panel["first_feedback_7d"] - panel["merged_30d"]
    return panel


def fit_results(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sample_defs = [
        ("overlap_full", panel),
        ("domain_overlap", panel.loc[panel["domain_overlap"] == 1].copy()),
        ("domain_overlap_fixlike", panel.loc[(panel["domain_overlap"] == 1) & (panel["fix_like"] == 1)].copy()),
        (
            "strict_domain_overlap_fixlike",
            panel.loc[(panel["domain_overlap"] == 1) & (panel["fix_like"] == 1) & (panel["overlap_repo_month"] == 1)].copy(),
        ),
    ]
    for sample_name, data in sample_defs:
        if data.empty or data["repo_url"].nunique() < 20:
            continue
        for outcome in ["first_feedback_7d", "approved_30d", "merged_30d", "review_merge_gap"]:
            frame = data.copy()
            model = smf.ols(f"{outcome} ~ is_ai + C(repo_url) + C(month_id)", data=frame).fit(
                cov_type="cluster", cov_kwds={"groups": frame["repo_url"]}
            )
            rows.append(
                {
                    "sample": sample_name,
                    "outcome": outcome,
                    "coef_is_ai": float(model.params["is_ai"]),
                    "se_is_ai": float(model.bse["is_ai"]),
                    "pval_is_ai": float(model.pvalues["is_ai"]),
                    "nobs": int(model.nobs),
                    "n_repos": int(frame["repo_url"].nunique()),
                    "n_months": int(frame["month_id"].nunique()),
                    "n_ai": int(frame["is_ai"].sum()),
                    "n_human": int((1 - frame["is_ai"]).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_sample_counts(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_name, data in [
        ("overlap_full", panel),
        ("domain_overlap", panel.loc[panel["domain_overlap"] == 1].copy()),
        ("domain_overlap_fixlike", panel.loc[(panel["domain_overlap"] == 1) & (panel["fix_like"] == 1)].copy()),
        (
            "strict_domain_overlap_fixlike",
            panel.loc[(panel["domain_overlap"] == 1) & (panel["fix_like"] == 1) & (panel["overlap_repo_month"] == 1)].copy(),
        ),
    ]:
        if data.empty:
            continue
        rows.append(
            {
                "sample": sample_name,
                "prs": int(len(data)),
                "repos": int(data["repo_url"].nunique()),
                "months": int(data["month_id"].nunique()),
                "ai_share": float(data["is_ai"].mean()),
                "fix_share": float(data["fix_like"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_figure(results: pd.DataFrame) -> None:
    plot_df = results.loc[
        results["sample"].isin(["domain_overlap", "domain_overlap_fixlike", "strict_domain_overlap_fixlike"])
        & results["outcome"].isin(["first_feedback_7d", "approved_30d", "merged_30d"])
    ].copy()
    if plot_df.empty:
        return
    pivot = plot_df.pivot(index="sample", columns="outcome", values="coef_is_ai").reset_index()
    order = ["domain_overlap", "domain_overlap_fixlike", "strict_domain_overlap_fixlike"]
    pivot["sample"] = pd.Categorical(pivot["sample"], categories=order, ordered=True)
    pivot = pivot.sort_values("sample")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pivot))
    ax.bar(x - 0.22, pivot["first_feedback_7d"], width=0.22, label="First feedback within 7d", color="#1f77b4")
    ax.bar(x, pivot["approved_30d"], width=0.22, label="Approved within 30d", color="#ff7f0e")
    ax.bar(x + 0.22, pivot["merged_30d"], width=0.22, label="Merged within 30d", color="#d62728")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Domain overlap", "Domain overlap + fix-like", "Strict overlap + fix-like"],
        rotation=15,
        ha="right",
    )
    ax.set_ylabel("AI - human coefficient")
    ax.set_title("AIDev upgrade: early feedback vs later certification")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def write_readout(sample_counts: pd.DataFrame, results: pd.DataFrame) -> None:
    lines = [
        "# Who Still Answers: AIDev Domain-Overlap Upgrade",
        "",
        "Date: April 4, 2026",
        "",
        "## Why This Upgrade Exists",
        "",
        "The baseline AIDev prototype established direct AI-use observation, but it still looked like a generic public-code-review analogue.",
        "This upgrade pushes the second setting toward a more question-like and domain-aligned pillar by adding repository-domain overlap and fix-like task restrictions.",
        "",
        "## Sample ladder",
        "",
    ]
    for _, row in sample_counts.iterrows():
        lines.append(
            f"- `{row['sample']}`: `{int(row['prs']):,}` PRs, `{int(row['repos']):,}` repos, AI share `{row['ai_share']:.3f}`, fix-like share `{row['fix_share']:.3f}`"
        )
    lines += ["", "## Fixed-effect results", ""]
    for _, row in results.iterrows():
        lines.append(
            f"- `{row['sample']}`, `{row['outcome']}` on `is_ai`: coef `{row['coef_is_ai']:.4f}`, clustered `p = {row['pval_is_ai']:.4f}`"
        )
    lines += [
        "",
        "## Safe read",
        "",
        "This still does not create a one-for-one Stack Overflow replication.",
        "What it does add is a direct-AI-use public technical collaboration setting that is more domain-aligned and more fix-task-oriented than the earlier generic overlap-repo build.",
        "If early feedback remains faster while later merge certification weakens inside the domain-overlap fix-like subset, the second pillar becomes much harder to dismiss as off-object.",
    ]
    READOUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    prs, reviews, comments, repo, task = load_remote_inputs()
    panel = prepare_panel(prs, reviews, comments, repo, task)
    panel.to_parquet(PANEL_PARQUET, index=False)

    results = fit_results(panel)
    sample_counts = build_sample_counts(panel)
    results.to_csv(RESULTS_CSV, index=False)
    sample_counts.to_csv(SAMPLE_COUNTS_CSV, index=False)
    build_figure(results)
    write_readout(sample_counts, results)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "rows": int(len(panel)),
                "domain_overlap_rows": int((panel["domain_overlap"] == 1).sum()),
                "fix_like_rows": int((panel["fix_like"] == 1).sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
