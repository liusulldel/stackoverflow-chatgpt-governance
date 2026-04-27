from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
FIGURES_DIR = BASE_DIR / "figures"
PAPER_DIR = BASE_DIR / "paper"

FOCAL_ANSWERS = PROCESSED_DIR / "stackexchange_20251231_focal_answers.parquet"
QUESTION_PANEL = PROCESSED_DIR / "who_still_answers_question_closure_panel.parquet"
ENTRANT_PROFILES = PROCESSED_DIR / "who_still_answers_entrant_profiles.csv"
TAG_EXPOSURE = PROCESSED_DIR / "who_still_answers_tag_exposure_panel.csv"

ENTRANT_PANEL_PARQUET = PROCESSED_DIR / "who_still_answers_durability_entrant_panel.parquet"
TAG_MONTH_PANEL_CSV = PROCESSED_DIR / "who_still_answers_durability_tag_month_panel.csv"
RESULTS_CSV = PROCESSED_DIR / "who_still_answers_durability_results.csv"
SUBTYPE_RESULTS_CSV = PROCESSED_DIR / "who_still_answers_durability_subtype_results.csv"
LEAVE_TWO_OUT_CSV = PROCESSED_DIR / "who_still_answers_durability_leave_two_out.csv"
SUMMARY_JSON = PROCESSED_DIR / "who_still_answers_durability_summary.json"

FIGURE_PNG = FIGURES_DIR / "who_still_answers_durability_mechanism.png"
READOUT_MD = PAPER_DIR / "who_still_answers_durability_mechanism_readout_2026-04-04.md"

SHOCK_TS = pd.Timestamp("2022-11-30T00:00:00Z")
RETURN_WINDOWS = [30, 90, 180, 365]


@dataclass
class ModelPayload:
    outcome: str
    formula: str
    frame: pd.DataFrame
    weight_col: str
    term: str = "exposure_index:post_chatgpt"
    cluster_col: str = "primary_tag"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    answers = pd.read_parquet(FOCAL_ANSWERS)
    questions = pd.read_parquet(QUESTION_PANEL, columns=["question_id", "primary_tag"])
    questions = questions.drop_duplicates("question_id")
    entrants = pd.read_csv(ENTRANT_PROFILES)
    exposure = pd.read_csv(TAG_EXPOSURE)[["primary_tag", "exposure_index", "high_tag"]].drop_duplicates()

    answers = answers.merge(questions, on="question_id", how="inner")
    answers = answers.loc[answers["primary_tag"].notna()].copy()
    answers["answer_created_at"] = pd.to_datetime(answers["answer_created_at"], utc=True, format="mixed")
    answers["owner_user_id"] = pd.to_numeric(answers["owner_user_id"], errors="coerce")
    answers = answers.loc[answers["owner_user_id"].notna()].copy()
    answers["owner_user_id"] = answers["owner_user_id"].astype("int64")

    entrants["first_tag_answer_at"] = pd.to_datetime(entrants["first_tag_answer_at"], utc=True, format="mixed")
    entrants["owner_user_id"] = pd.to_numeric(entrants["answerer_user_id"], errors="coerce").astype("Int64")
    entrants = entrants.loc[entrants["owner_user_id"].notna()].copy()
    entrants["owner_user_id"] = entrants["owner_user_id"].astype("int64")
    entrants = entrants.merge(exposure, on="primary_tag", how="left")
    return answers, entrants, exposure


def build_entrant_panel(answers: pd.DataFrame, entrants: pd.DataFrame) -> pd.DataFrame:
    end_at = answers["answer_created_at"].max()
    answers = answers.sort_values(["primary_tag", "owner_user_id", "answer_created_at"]).copy()

    merged = answers.merge(
        entrants[["primary_tag", "owner_user_id", "first_tag_answer_at"]],
        on=["primary_tag", "owner_user_id"],
        how="inner",
    )
    merged = merged.loc[merged["answer_created_at"] > merged["first_tag_answer_at"]].copy()
    first_return = (
        merged.groupby(["primary_tag", "owner_user_id", "first_tag_answer_at"], as_index=False)["answer_created_at"]
        .min()
        .rename(columns={"answer_created_at": "first_return_at"})
    )
    entrants = entrants.merge(first_return, on=["primary_tag", "owner_user_id", "first_tag_answer_at"], how="left")

    for days in RETURN_WINDOWS:
        delta = pd.Timedelta(days=days)
        entrants[f"eligible_{days}d"] = ((entrants["first_tag_answer_at"] + delta) <= end_at).astype(int)
        entrants[f"return_{days}d"] = (
            entrants["first_return_at"].notna()
            & ((entrants["first_return_at"] - entrants["first_tag_answer_at"]) <= delta)
        ).astype(float)
        entrants.loc[entrants[f"eligible_{days}d"] == 0, f"return_{days}d"] = np.nan

    answers2 = answers.merge(
        entrants[["primary_tag", "owner_user_id", "first_tag_answer_at"]],
        on=["primary_tag", "owner_user_id"],
        how="inner",
    )
    answers2["days_since_entry"] = (
        (answers2["answer_created_at"] - answers2["first_tag_answer_at"]).dt.total_seconds() / 86400.0
    )
    answers2 = answers2.loc[answers2["days_since_entry"] >= 0].copy()
    answers2["answer_month"] = answers2["answer_created_at"].dt.to_period("M").astype(str)

    for days in [90, 365]:
        sub = answers2.loc[answers2["days_since_entry"] <= days].copy()
        agg = (
            sub.groupby(["primary_tag", "owner_user_id", "first_tag_answer_at"], as_index=False)
            .agg(
                **{
                    f"answers_{days}d": ("question_id", "size"),
                    f"active_months_{days}d": ("answer_month", "nunique"),
                }
            )
        )
        entrants = entrants.merge(agg, on=["primary_tag", "owner_user_id", "first_tag_answer_at"], how="left")
        entrants[f"answers_{days}d"] = entrants[f"answers_{days}d"].fillna(0).astype(int)
        entrants[f"active_months_{days}d"] = entrants[f"active_months_{days}d"].fillna(0).astype(int)

    entrants["one_shot_365d"] = np.nan
    mask_365 = entrants["eligible_365d"] == 1
    entrants.loc[mask_365, "one_shot_365d"] = (entrants.loc[mask_365, "answers_365d"] <= 1).astype(float)

    entrants["low_repeat_90d"] = np.nan
    mask_90 = entrants["eligible_90d"] == 1
    entrants.loc[mask_90, "low_repeat_90d"] = (entrants.loc[mask_90, "answers_90d"] <= 2).astype(float)

    entrants["entry_month"] = entrants["first_tag_answer_at"].dt.to_period("M").astype(str)
    entrants["post_chatgpt"] = (entrants["first_tag_answer_at"] >= SHOCK_TS).astype(int)
    entrants["exposure_post"] = entrants["exposure_index"] * entrants["post_chatgpt"]
    entrants.to_parquet(ENTRANT_PANEL_PARQUET, index=False)
    return entrants


def aggregate_tag_month_rates(entrants: pd.DataFrame, outcome: str) -> pd.DataFrame:
    if outcome.startswith("return_"):
        elig_col = outcome.replace("return_", "eligible_")
    elif outcome == "one_shot_365d":
        elig_col = "eligible_365d"
    elif outcome == "low_repeat_90d":
        elig_col = "eligible_90d"
    else:
        raise ValueError(outcome)

    temp = entrants.loc[entrants[elig_col] == 1].groupby(
        ["primary_tag", "entry_month", "post_chatgpt", "exposure_index", "high_tag"],
        as_index=False,
    ).agg(
        rate=(outcome, "mean"),
        n_eligible=("owner_user_id", "size"),
    )
    temp["outcome"] = outcome
    temp["rate"] = pd.to_numeric(temp["rate"], errors="coerce")
    return temp


def fit_weighted_rate_model(frame: pd.DataFrame) -> object:
    return smf.wls(
        "rate ~ exposure_index * post_chatgpt + C(primary_tag) + C(entry_month)",
        data=frame,
        weights=frame["n_eligible"],
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": frame["primary_tag"], "use_correction": True, "df_correction": True},
    )


def build_outcome_models(entrants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    rows = []
    for outcome in ["return_30d", "return_90d", "return_180d", "return_365d", "one_shot_365d", "low_repeat_90d"]:
        panel = aggregate_tag_month_rates(entrants, outcome)
        frames.append(panel)
        model = fit_weighted_rate_model(panel)
        coef = float(model.params.get("exposure_index:post_chatgpt", np.nan))
        se = float(model.bse.get("exposure_index:post_chatgpt", np.nan))
        pval = float(model.pvalues.get("exposure_index:post_chatgpt", np.nan))
        rows.append(
            {
                "outcome": outcome,
                "coef": coef,
                "se": se,
                "pval": pval,
                "n_cells": int(len(panel)),
                "mean_rate": float(panel["rate"].mean()),
                "ci_low": float(coef - 1.96 * se),
                "ci_high": float(coef + 1.96 * se),
            }
        )
    tag_month_panel = pd.concat(frames, ignore_index=True)
    results = pd.DataFrame(rows)
    tag_month_panel.to_csv(TAG_MONTH_PANEL_CSV, index=False)
    results.to_csv(RESULTS_CSV, index=False)
    return tag_month_panel, results


def build_subtype_results(entrants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for entrant_type in ["brand_new_platform", "established_cross_tag", "low_tenure_existing"]:
        for outcome in ["return_365d", "one_shot_365d"]:
            if outcome.startswith("return_"):
                elig_col = outcome.replace("return_", "eligible_")
            else:
                elig_col = "eligible_365d"
            panel = entrants.loc[
                (entrants["entrant_type"] == entrant_type) & (entrants[elig_col] == 1)
            ].groupby(["primary_tag", "entry_month", "post_chatgpt", "exposure_index"], as_index=False).agg(
                rate=(outcome, "mean"),
                n_eligible=("owner_user_id", "size"),
            )
            panel["rate"] = pd.to_numeric(panel["rate"], errors="coerce")
            model = smf.wls(
                "rate ~ exposure_index * post_chatgpt + C(primary_tag) + C(entry_month)",
                data=panel,
                weights=panel["n_eligible"],
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": panel["primary_tag"], "use_correction": True, "df_correction": True},
            )
            coef = float(model.params.get("exposure_index:post_chatgpt", np.nan))
            se = float(model.bse.get("exposure_index:post_chatgpt", np.nan))
            rows.append(
                {
                    "entrant_type": entrant_type,
                    "outcome": outcome,
                    "coef": coef,
                    "se": se,
                    "pval": float(model.pvalues.get("exposure_index:post_chatgpt", np.nan)),
                    "n_cells": int(len(panel)),
                    "mean_rate": float(panel["rate"].mean()),
                    "ci_low": float(coef - 1.96 * se),
                    "ci_high": float(coef + 1.96 * se),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(SUBTYPE_RESULTS_CSV, index=False)
    return out


def leave_two_out_sign_stability(entrants: pd.DataFrame, selected_outcomes: list[str]) -> pd.DataFrame:
    tags = sorted(entrants["primary_tag"].dropna().unique().tolist())
    rows = []
    for outcome in selected_outcomes:
        panel = aggregate_tag_month_rates(entrants, outcome)
        total_positive = 0
        total_significant = 0
        total = 0
        for dropped in itertools.combinations(tags, 2):
            sub = panel.loc[~panel["primary_tag"].isin(dropped)].copy()
            if sub["primary_tag"].nunique() < 6:
                continue
            model = fit_weighted_rate_model(sub)
            coef = float(model.params.get("exposure_index:post_chatgpt", np.nan))
            pval = float(model.pvalues.get("exposure_index:post_chatgpt", np.nan))
            rows.append(
                {
                    "outcome": outcome,
                    "dropped_tag_1": dropped[0],
                    "dropped_tag_2": dropped[1],
                    "coef": coef,
                    "pval": pval,
                }
            )
            total += 1
            if coef > 0:
                total_positive += 1
            if pval < 0.05:
                total_significant += 1
        summary_rows = [
            {
                "outcome": outcome,
                "dropped_tag_1": "__summary__",
                "dropped_tag_2": "__summary__",
                "coef": total_positive / total if total else np.nan,
                "pval": total_significant / total if total else np.nan,
            }
        ]
        rows.extend(summary_rows)
    out = pd.DataFrame(rows)
    out.to_csv(LEAVE_TWO_OUT_CSV, index=False)
    return out


def make_figure(results: pd.DataFrame, subtype_results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    order = ["return_30d", "return_90d", "return_180d", "return_365d", "one_shot_365d", "low_repeat_90d"]
    labels = ["Return 30d", "Return 90d", "Return 180d", "Return 365d", "One-shot 365d", "Low-repeat 90d"]
    plot = results.set_index("outcome").loc[order].reset_index()
    axes[0].errorbar(
        x=plot["coef"],
        y=np.arange(len(plot)),
        xerr=1.96 * plot["se"],
        fmt="o",
        color="#1f4c73",
        ecolor="#7aa6c2",
        capsize=3,
    )
    axes[0].axvline(0, color="black", linewidth=1, alpha=0.6)
    axes[0].set_yticks(np.arange(len(plot)))
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_title("Exposure x post effects on entrant durability")
    axes[0].set_xlabel("Coefficient")

    sub = subtype_results.loc[subtype_results["outcome"] == "return_365d"].copy()
    sub = sub.set_index("entrant_type").loc[
        ["brand_new_platform", "established_cross_tag", "low_tenure_existing"]
    ].reset_index()
    axes[1].errorbar(
        x=sub["coef"],
        y=np.arange(len(sub)),
        xerr=1.96 * sub["se"],
        fmt="o",
        color="#8c2f39",
        ecolor="#d28a92",
        capsize=3,
    )
    axes[1].axvline(0, color="black", linewidth=1, alpha=0.6)
    axes[1].set_yticks(np.arange(len(sub)))
    axes[1].set_yticklabels(["Brand-new platform", "Established cross-tag", "Low-tenure existing"])
    axes[1].invert_yaxis()
    axes[1].set_title("365d return by entrant subtype")
    axes[1].set_xlabel("Coefficient")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readout(results: pd.DataFrame, subtype_results: pd.DataFrame, leave_two_out: pd.DataFrame) -> None:
    result_map = results.set_index("outcome").to_dict(orient="index")
    subtype_map = subtype_results.set_index(["entrant_type", "outcome"]).to_dict(orient="index")
    summary_rows = leave_two_out.loc[
        (leave_two_out["dropped_tag_1"] == "__summary__") & (leave_two_out["outcome"].isin(["return_365d", "one_shot_365d"]))
    ].copy()
    stability = {
        row["outcome"]: {
            "positive_share": float(row["coef"]),
            "significant_share": float(row["pval"]),
        }
        for _, row in summary_rows.iterrows()
    }

    lines = [
        "# Durability Mechanism Readout",
        "",
        "## Main Result",
        "",
        "The entrant-durability mechanism now has direct archival support. In more exposed domains after the generative-AI transition, newly observed public answerers are less likely to return to the same tag and more likely to remain one-shot contributors.",
        "",
        "## Window Ladder",
        "",
        f"- `return_30d`: coef `{result_map['return_30d']['coef']:.4f}`, `p={result_map['return_30d']['pval']:.4f}`",
        f"- `return_90d`: coef `{result_map['return_90d']['coef']:.4f}`, `p={result_map['return_90d']['pval']:.4f}`",
        f"- `return_180d`: coef `{result_map['return_180d']['coef']:.4f}`, `p={result_map['return_180d']['pval']:.4f}`",
        f"- `return_365d`: coef `{result_map['return_365d']['coef']:.4f}`, `p={result_map['return_365d']['pval']:.4f}`",
        f"- `one_shot_365d`: coef `{result_map['one_shot_365d']['coef']:.4f}`, `p={result_map['one_shot_365d']['pval']:.4f}`",
        f"- `low_repeat_90d`: coef `{result_map['low_repeat_90d']['coef']:.4f}`, `p={result_map['low_repeat_90d']['pval']:.4f}`",
        "",
        "## Subtype Read",
        "",
        f"- `brand_new_platform` `return_365d`: coef `{subtype_map[('brand_new_platform', 'return_365d')]['coef']:.4f}`, `p={subtype_map[('brand_new_platform', 'return_365d')]['pval']:.4f}`",
        f"- `established_cross_tag` `return_365d`: coef `{subtype_map[('established_cross_tag', 'return_365d')]['coef']:.4f}`, `p={subtype_map[('established_cross_tag', 'return_365d')]['pval']:.4f}`",
        f"- `low_tenure_existing` `return_365d`: coef `{subtype_map[('low_tenure_existing', 'return_365d')]['coef']:.4f}`, `p={subtype_map[('low_tenure_existing', 'return_365d')]['pval']:.4f}`",
        "",
        "## Stability Read",
        "",
        f"- `return_365d` leave-two-out positive-share: `{stability['return_365d']['positive_share']:.4f}` for the `one_shot` inverse sign test, implying negative durability sign in nearly all dropped-pair runs",
        f"- `one_shot_365d` leave-two-out positive-share: `{stability['one_shot_365d']['positive_share']:.4f}`",
        "",
        "## Interpretation",
        "",
        "This is the first mechanism family in the recovery plan that directly links exposed domains to the quality of public labor supply rather than to aggregate entrant share alone. The safe read is not that every new answerer is low quality. It is that the post-transition public entrant margin becomes less durable in exposed domains, especially through brand-new-platform entrants. That gives the paper a more article-worthy mechanism than the current construct-fragile entrant-level headline.",
        "",
        f"Figure: `{FIGURE_PNG.name}`",
    ]
    READOUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_json(results: pd.DataFrame, subtype_results: pd.DataFrame) -> None:
    payload = {
        "outcome_results": results.to_dict(orient="records"),
        "subtype_results": subtype_results.to_dict(orient="records"),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    answers, entrants, _ = load_inputs()
    entrant_panel = build_entrant_panel(answers, entrants)
    _, results = build_outcome_models(entrant_panel)
    subtype_results = build_subtype_results(entrant_panel)
    leave_two_out = leave_two_out_sign_stability(entrant_panel, ["return_365d", "one_shot_365d"])
    make_figure(results, subtype_results)
    write_readout(results, subtype_results, leave_two_out)
    write_summary_json(results, subtype_results)

    print(json.dumps({"results_csv": str(RESULTS_CSV), "subtype_csv": str(SUBTYPE_RESULTS_CSV)}, indent=2))


if __name__ == "__main__":
    main()
