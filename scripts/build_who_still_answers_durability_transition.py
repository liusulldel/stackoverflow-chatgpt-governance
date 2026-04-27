from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
PAPER_DIR = BASE_DIR / "paper"

ROLE_QUESTION_PANEL = PROCESSED_DIR / "who_still_answers_answer_role_question_panel.parquet"
ENTRANT_PANEL = PROCESSED_DIR / "who_still_answers_durability_entrant_panel.parquet"

TAG_MONTH_OUT = PROCESSED_DIR / "who_still_answers_durability_transition_tag_month_panel.csv"
RESULTS_OUT = PROCESSED_DIR / "who_still_answers_durability_transition_results.csv"
SUMMARY_OUT = PROCESSED_DIR / "who_still_answers_durability_transition_summary.json"
READOUT_MD = PAPER_DIR / "who_still_answers_durability_transition_readout_2026-04-13.md"

WINDOW_DAYS = 365
TARGET_ROLES = ["first_positive", "top_score", "accepted_current"]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    entrants = pd.read_parquet(ENTRANT_PANEL)
    # Read only what we need to build role-occupancy transition timing.
    role = pd.read_parquet(
        ROLE_QUESTION_PANEL,
        columns=[
            "primary_tag",
            "owner_user_id",
            "role",
            "answer_created_at",
            "first_tag_answer_at",
            "answerer_profile_observed",
        ],
    )
    return entrants, role


def build_first_role_days(role: pd.DataFrame) -> pd.DataFrame:
    role = role.loc[role["answerer_profile_observed"] == 1].copy()
    role = role.loc[role["role"].isin(TARGET_ROLES)].copy()
    role["days_since_entry"] = (
        (role["answer_created_at"] - role["first_tag_answer_at"]).dt.total_seconds() / 86400.0
    )
    role = role.loc[role["days_since_entry"] >= 0].copy()

    first_days = (
        role.groupby(["primary_tag", "owner_user_id", "role"], as_index=False)["days_since_entry"]
        .min()
        .pivot(index=["primary_tag", "owner_user_id"], columns="role", values="days_since_entry")
        .reset_index()
    )
    # Ensure stable columns even if one role is absent in a sample.
    for r in TARGET_ROLES:
        if r not in first_days.columns:
            first_days[r] = np.nan
    return first_days


def attach_transition_outcomes(entrants: pd.DataFrame, first_days: pd.DataFrame) -> pd.DataFrame:
    frame = entrants.merge(first_days, on=["primary_tag", "owner_user_id"], how="left")

    elig_col = f"eligible_{WINDOW_DAYS}d"
    if elig_col not in frame.columns:
        raise ValueError(f"Missing eligibility column {elig_col} in entrant panel.")

    for r in TARGET_ROLES:
        out = f"transition_{r}_{WINDOW_DAYS}d"
        frame[out] = ((frame[r].notna()) & (frame[r] <= WINDOW_DAYS)).astype(float)
        frame.loc[frame[elig_col] != 1, out] = np.nan

    frame[f"transition_any_cert_role_{WINDOW_DAYS}d"] = (
        frame[[f"transition_{r}_{WINDOW_DAYS}d" for r in TARGET_ROLES]].max(axis=1)
    )
    return frame


def aggregate_tag_month_rates(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    elig_col = f"eligible_{WINDOW_DAYS}d"
    temp = frame.loc[frame[elig_col] == 1].groupby(
        ["primary_tag", "entry_month", "post_chatgpt", "exposure_index", "high_tag"],
        as_index=False,
    ).agg(
        rate=(outcome, "mean"),
        n_eligible=("owner_user_id", "size"),
    )
    temp["rate"] = pd.to_numeric(temp["rate"], errors="coerce")
    temp["outcome"] = outcome
    return temp


def fit_weighted_rate_model(panel: pd.DataFrame) -> object:
    return smf.wls(
        "rate ~ exposure_index * post_chatgpt + C(primary_tag) + C(entry_month)",
        data=panel,
        weights=panel["n_eligible"],
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel["primary_tag"], "use_correction": True, "df_correction": True},
    )


def build_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = [f"transition_{r}_{WINDOW_DAYS}d" for r in TARGET_ROLES] + [
        f"transition_any_cert_role_{WINDOW_DAYS}d"
    ]
    panels = []
    rows = []
    for outcome in outcomes:
        panel = aggregate_tag_month_rates(frame, outcome)
        panels.append(panel)
        model = fit_weighted_rate_model(panel)
        coef = float(model.params.get("exposure_index:post_chatgpt", np.nan))
        se = float(model.bse.get("exposure_index:post_chatgpt", np.nan))
        rows.append(
            {
                "outcome": outcome,
                "coef": coef,
                "se": se,
                "pval": float(model.pvalues.get("exposure_index:post_chatgpt", np.nan)),
                "ci_low": float(coef - 1.96 * se),
                "ci_high": float(coef + 1.96 * se),
                "n_cells": int(len(panel)),
                "mean_rate": float(panel["rate"].mean()),
                "mean_n_eligible": float(panel["n_eligible"].mean()),
            }
        )
    tag_month = pd.concat(panels, ignore_index=True)
    results = pd.DataFrame(rows).sort_values("outcome")
    return tag_month, results


def write_readout(summary: dict[str, object]) -> None:
    lines = [
        "# Durability Transition Readout (Prototype)",
        "",
        "Goal: test whether the post-transition public entrant margin is less likely to mature into later, certification-adjacent roles.",
        "",
        "This prototype defines a transition as the first time a newly observed focal-tag answerer appears as a role occupant",
        f"in `{', '.join(TARGET_ROLES)}` within {WINDOW_DAYS} days of their first focal-tag answer.",
        "",
        "## Key Counts",
        "",
        f"- Entrants in panel: `{summary['n_entrants']}`",
        f"- Eligible {WINDOW_DAYS}d entrants: `{summary['n_eligible_365d']}`",
        f"- Role-panel rows used (target roles only): `{summary['n_role_rows_target']}`",
        "",
        "## Mean Transition Rates (Eligible 365d)",
        "",
    ]
    for k, v in summary["mean_rates_eligible_365d"].items():
        lines.append(f"- `{k}`: `{v:.4f}`")
    lines += [
        "",
        "## Exposure x Post Results (Tag FE + Month FE; weighted by eligible entrants)",
        "",
        "This is a durability-upgrade attempt. Promote only if it is clearly stronger than generic return windows and stable under conservative inference.",
        "",
    ]
    for row in summary["results"]:
        lines.append(
            f"- `{row['outcome']}`: coef `{row['coef']:.4f}`, p `{row['pval']:.4g}`, mean `{row['mean_rate']:.4f}`"
        )
    READOUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    entrants, role = load_inputs()

    first_days = build_first_role_days(role)
    frame = attach_transition_outcomes(entrants, first_days)

    tag_month, results = build_models(frame)

    TAG_MONTH_OUT.parent.mkdir(parents=True, exist_ok=True)
    tag_month.to_csv(TAG_MONTH_OUT, index=False)
    results.to_csv(RESULTS_OUT, index=False)

    eligible = frame.loc[frame[f"eligible_{WINDOW_DAYS}d"] == 1].copy()
    mean_rates = {c: float(eligible[c].mean()) for c in results["outcome"].tolist()}

    summary = {
        "window_days": WINDOW_DAYS,
        "target_roles": TARGET_ROLES,
        "n_entrants": int(len(frame)),
        "n_eligible_365d": int(eligible.shape[0]),
        "n_role_rows_target": int(role.loc[role["role"].isin(TARGET_ROLES)].shape[0]),
        "mean_rates_eligible_365d": mean_rates,
        "results": results.to_dict(orient="records"),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readout(summary)


if __name__ == "__main__":
    main()

