from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
FIGURES_DIR = BASE_DIR / "figures"
PAPER_DIR = BASE_DIR / "paper"

QUESTION_PANEL = PROCESSED_DIR / "who_still_answers_question_closure_panel.parquet"
FOCAL_ANSWERS = PROCESSED_DIR / "stackexchange_20251231_focal_answers.parquet"
ENTRANT_PROFILES = PROCESSED_DIR / "who_still_answers_entrant_profiles.csv"
SELECTION_PANEL = PROCESSED_DIR / "selection_composition_primary_panel.csv"

ROLE_QUESTION_PANEL = PROCESSED_DIR / "who_still_answers_answer_role_question_panel.parquet"
ROLE_TAG_MONTH_PANEL = PROCESSED_DIR / "who_still_answers_answer_role_tag_month_panel.csv"
ROLE_RESULTS_CSV = PROCESSED_DIR / "who_still_answers_answer_role_results.csv"
ROLE_SUMMARY_JSON = PROCESSED_DIR / "who_still_answers_answer_role_summary.json"

ROLE_FIGURE = FIGURES_DIR / "who_still_answers_answer_role_mechanism.png"
READOUT_MD = PAPER_DIR / "who_still_answers_answer_role_mechanism_readout_2026-04-04.md"

ROLE_ORDER = ["first_answer", "first_positive", "top_score", "accepted_current"]
PRIMARY_OUTCOME = "recent_entrant_90d_share"
SUPPORTING_OUTCOMES = [
    "recent_entrant_365d_share",
    "brand_new_platform_share",
    "mean_log_tenure_days",
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    questions = pd.read_parquet(
        QUESTION_PANEL,
        columns=["question_id", "primary_tag", "month_id", "post_chatgpt", "exposure_index", "high_tag"],
    ).drop_duplicates("question_id")
    answers = pd.read_parquet(FOCAL_ANSWERS)
    entrants = pd.read_csv(ENTRANT_PROFILES)
    selection = pd.read_csv(SELECTION_PANEL).rename(columns={"tag": "primary_tag"})
    selection = selection[["primary_tag", "month_id", "residual_queue_complexity_index_mean"]].drop_duplicates()

    answers["answer_created_at"] = pd.to_datetime(answers["answer_created_at"], utc=True, format="mixed")
    answers["owner_user_id"] = pd.to_numeric(answers["owner_user_id"], errors="coerce")
    answers = answers.dropna(subset=["owner_user_id"]).copy()
    answers["owner_user_id"] = answers["owner_user_id"].astype("int64")

    entrants["owner_user_id"] = pd.to_numeric(entrants["answerer_user_id"], errors="coerce")
    entrants = entrants.dropna(subset=["owner_user_id"]).copy()
    entrants["owner_user_id"] = entrants["owner_user_id"].astype("int64")
    entrants["first_tag_answer_at"] = pd.to_datetime(entrants["first_tag_answer_at"], utc=True, format="mixed")

    return questions, answers, entrants, selection


def build_role_question_panel(
    questions: pd.DataFrame,
    answers: pd.DataFrame,
    entrants: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    merged = answers.merge(questions, on="question_id", how="inner")
    merged = merged.merge(
        entrants[["primary_tag", "owner_user_id", "first_tag_answer_at", "entrant_type"]],
        on=["primary_tag", "owner_user_id"],
        how="left",
    )
    merged = merged.merge(selection, on=["primary_tag", "month_id"], how="left")
    merged["answerer_profile_observed"] = merged["first_tag_answer_at"].notna().astype(int)
    merged["tenure_days_at_answer"] = (
        (merged["answer_created_at"] - merged["first_tag_answer_at"]).dt.total_seconds() / 86400.0
    )
    merged["recent_entrant_30d"] = (
        (merged["tenure_days_at_answer"] >= 0) & (merged["tenure_days_at_answer"] <= 30)
    ).astype(float)
    merged["recent_entrant_90d"] = (
        (merged["tenure_days_at_answer"] >= 0) & (merged["tenure_days_at_answer"] <= 90)
    ).astype(float)
    merged["recent_entrant_365d"] = (
        (merged["tenure_days_at_answer"] >= 0) & (merged["tenure_days_at_answer"] <= 365)
    ).astype(float)
    merged["incumbent_365d"] = (merged["tenure_days_at_answer"] > 365).astype(float)
    merged["brand_new_platform"] = (merged["entrant_type"] == "brand_new_platform").astype(float)
    merged["low_tenure_existing"] = (merged["entrant_type"] == "low_tenure_existing").astype(float)
    merged["established_cross_tag"] = (merged["entrant_type"] == "established_cross_tag").astype(float)
    merged["mean_log_tenure_days"] = np.log1p(merged["tenure_days_at_answer"].clip(lower=0))

    first_answer = (
        merged.sort_values(["question_id", "answer_created_at", "answer_id"])
        .drop_duplicates("question_id")
        .assign(role="first_answer")
    )
    first_positive = (
        merged.loc[merged["score"] > 0]
        .sort_values(["question_id", "answer_created_at", "answer_id"])
        .drop_duplicates("question_id")
        .assign(role="first_positive")
    )
    top_score = (
        merged.sort_values(["question_id", "score", "answer_created_at", "answer_id"], ascending=[True, False, True, True])
        .drop_duplicates("question_id")
        .assign(role="top_score")
    )
    accepted_current = (
        merged.loc[merged["is_current_accepted_answer"] == 1]
        .sort_values(["question_id", "answer_created_at", "answer_id"])
        .drop_duplicates("question_id")
        .assign(role="accepted_current")
    )

    role_questions = pd.concat([first_answer, first_positive, top_score, accepted_current], ignore_index=True)
    month_order = {month: idx + 1 for idx, month in enumerate(sorted(role_questions["month_id"].dropna().unique()))}
    role_questions["time_index"] = role_questions["month_id"].map(month_order)
    role_questions["high_post"] = role_questions["high_tag"] * role_questions["post_chatgpt"]
    role_questions["answerer_profile_observed"] = role_questions["answerer_profile_observed"].fillna(0)
    role_questions.to_parquet(ROLE_QUESTION_PANEL, index=False)
    return role_questions


def aggregate_role_tag_month(role_questions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        role_questions.groupby(
            [
                "primary_tag",
                "month_id",
                "time_index",
                "role",
                "post_chatgpt",
                "exposure_index",
                "high_tag",
                "high_post",
                "residual_queue_complexity_index_mean",
            ],
            as_index=False,
        )
        .agg(
            n_role_questions=("question_id", "nunique"),
            profile_coverage=("answerer_profile_observed", "mean"),
            recent_entrant_30d_share=("recent_entrant_30d", "mean"),
            recent_entrant_90d_share=("recent_entrant_90d", "mean"),
            recent_entrant_365d_share=("recent_entrant_365d", "mean"),
            incumbent_365d_share=("incumbent_365d", "mean"),
            brand_new_platform_share=("brand_new_platform", "mean"),
            low_tenure_existing_share=("low_tenure_existing", "mean"),
            established_cross_tag_share=("established_cross_tag", "mean"),
            mean_log_tenure_days=("mean_log_tenure_days", "mean"),
        )
    )
    grouped.to_csv(ROLE_TAG_MONTH_PANEL, index=False)
    return grouped


def fit_role_models(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    outcomes = [PRIMARY_OUTCOME, *SUPPORTING_OUTCOMES]
    for outcome in outcomes:
        for role in ROLE_ORDER:
            sub = panel.loc[panel["role"] == role].copy()
            for model_name, formula in [
                (
                    "base",
                    f"{outcome} ~ exposure_index * post_chatgpt + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
                ),
                (
                    "with_complexity",
                    f"{outcome} ~ exposure_index * post_chatgpt + residual_queue_complexity_index_mean + "
                    "C(primary_tag):time_index + C(primary_tag) + C(month_id)",
                ),
            ]:
                model = smf.wls(formula, data=sub, weights=sub["n_role_questions"]).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": sub["primary_tag"], "use_correction": True, "df_correction": True},
                )
                rows.append(
                    {
                        "outcome": outcome,
                        "role": role,
                        "model": model_name,
                        "coef": float(model.params.get("exposure_index:post_chatgpt", np.nan)),
                        "se": float(model.bse.get("exposure_index:post_chatgpt", np.nan)),
                        "pval": float(model.pvalues.get("exposure_index:post_chatgpt", np.nan)),
                        "complexity_coef": float(model.params.get("residual_queue_complexity_index_mean", np.nan)),
                        "complexity_pval": float(model.pvalues.get("residual_queue_complexity_index_mean", np.nan)),
                        "n_cells": int(len(sub)),
                        "mean_outcome": float(sub[outcome].mean()),
                        "mean_weight": float(sub["n_role_questions"].mean()),
                    }
                )
    results = pd.DataFrame(rows)
    results.to_csv(ROLE_RESULTS_CSV, index=False)
    return results


def write_summary(role_questions: pd.DataFrame, panel: pd.DataFrame, results: pd.DataFrame) -> dict[str, object]:
    key = results.loc[(results["outcome"] == PRIMARY_OUTCOME) & (results["model"] == "with_complexity")].copy()
    key["role"] = pd.Categorical(key["role"], categories=ROLE_ORDER, ordered=True)
    key = key.sort_values("role")
    summary = {
        "n_role_rows": int(len(role_questions)),
        "n_tag_month_role_cells": int(len(panel)),
        "profile_coverage_share": float(role_questions["answerer_profile_observed"].mean()),
        "primary_outcome": PRIMARY_OUTCOME,
        "key_results_with_complexity": {
            row["role"]: {
                "coef": float(row["coef"]),
                "pval": float(row["pval"]),
                "mean_outcome": float(row["mean_outcome"]),
            }
            for _, row in key.iterrows()
        },
    }
    ROLE_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def make_figure(results: pd.DataFrame) -> None:
    plot_df = results.loc[
        (results["outcome"] == PRIMARY_OUTCOME) & (results["model"] == "with_complexity")
    ].copy()
    plot_df["role"] = pd.Categorical(plot_df["role"], categories=ROLE_ORDER, ordered=True)
    plot_df = plot_df.sort_values("role")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(plot_df))
    ax.errorbar(
        plot_df["coef"],
        y,
        xerr=1.96 * plot_df["se"],
        fmt="o",
        color="#1f4e79",
        ecolor="#8aa8c5",
        elinewidth=2,
        capsize=4,
    )
    ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(["First answer", "First positive", "Top score", "Accepted current"])
    ax.set_xlabel("Exposure x Post coefficient on recent-entrant (90d) share")
    ax.set_title("Answer-role reallocation is strongest before accepted-current certification")
    fig.tight_layout()
    fig.savefig(ROLE_FIGURE, dpi=200)
    plt.close(fig)


def write_readout(results: pd.DataFrame, summary: dict[str, object]) -> None:
    key = results.loc[(results["outcome"] == PRIMARY_OUTCOME) & (results["model"] == "with_complexity")].copy()
    key["role"] = pd.Categorical(key["role"], categories=ROLE_ORDER, ordered=True)
    key = key.sort_values("role")
    lines = [
        "# Answer-Role Mechanism Readout",
        "",
        "## Main Result",
        "",
        "The role-reallocation mechanism now has direct archival support, but it is narrower than a generic entrant-share story.",
        "In more exposed domains after the generative-AI transition, the share of very recent entrants rises most clearly in",
        "`first_answer`, `first_positive`, and `top_score` roles. The same shift is weaker in the `accepted_current` role",
        "once residual queue complexity is held constant.",
        "",
        "## Primary Role Ladder (`recent_entrant_90d_share`, complexity-controlled)",
        "",
    ]
    for _, row in key.iterrows():
        lines.append(f"- `{row['role']}`: coef `{row['coef']:.4f}`, p `{row['pval']:.4f}`")
    lines.extend(
        [
            "",
            "## Supporting Read",
            "",
            "- The role movement is not a generic brand-new-platform surge. `brand_new_platform_share` falls across roles,",
            "  which means the role story is better read as broader recent-entrant reallocation than as simple brand-new takeover.",
            "- `accepted_current` is not untouched, but it remains the weakest of the four role channels in the primary 90-day test.",
            "",
            "## Safe Interpretation",
            "",
            "The bounded claim is that exposed domains become more reliant on recent entrants in rapid-response and endorsed-answer roles,",
            "while accepted-current certification remains less responsive than those earlier answer slots. That makes the reallocation",
            "visible inside the answer pipeline rather than only in aggregate entrant counts.",
            "",
            f"Profile coverage across role rows: `{summary['profile_coverage_share']:.4f}`",
            "",
            f"Figure: `{ROLE_FIGURE.name}`",
        ]
    )
    READOUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    questions, answers, entrants, selection = load_inputs()
    role_questions = build_role_question_panel(questions, answers, entrants, selection)
    panel = aggregate_role_tag_month(role_questions)
    results = fit_role_models(panel)
    summary = write_summary(role_questions, panel, results)
    make_figure(results)
    write_readout(results, summary)


if __name__ == "__main__":
    main()
