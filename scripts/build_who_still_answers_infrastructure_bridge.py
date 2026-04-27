from __future__ import annotations

import itertools
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

ROLE_TAG_MONTH_PANEL = PROCESSED_DIR / "who_still_answers_answer_role_tag_month_panel.csv"
SUBTYPE_CONSEQUENCE_PANEL = PROCESSED_DIR / "p1_jmis_subtype_consequence_panel.csv"
HARMONIZED_SUBTYPE_PANEL = PROCESSED_DIR / "p1_p2_harmonized_subtype_panel.csv"

BRIDGE_PANEL_CSV = PROCESSED_DIR / "who_still_answers_infrastructure_bridge_panel.csv"
BRIDGE_RESULTS_CSV = PROCESSED_DIR / "who_still_answers_infrastructure_bridge_results.csv"
BRIDGE_LEAVE_TWO_OUT_CSV = PROCESSED_DIR / "who_still_answers_infrastructure_bridge_leave_two_out.csv"
BRIDGE_SUMMARY_JSON = PROCESSED_DIR / "who_still_answers_infrastructure_bridge_summary.json"

BRIDGE_FIGURE = FIGURES_DIR / "who_still_answers_infrastructure_bridge.png"
READOUT_MD = PAPER_DIR / "who_still_answers_infrastructure_bridge_readout_2026-04-04.md"

PRIMARY_BRIDGE_VAR = "recent_gap_first_vs_accepted"


def load_bridge_panel() -> pd.DataFrame:
    role_panel = pd.read_csv(ROLE_TAG_MONTH_PANEL)
    wide = role_panel.pivot_table(
        index=["primary_tag", "month_id", "time_index", "exposure_index", "residual_queue_complexity_index_mean"],
        columns="role",
        values=[
            "recent_entrant_90d_share",
            "recent_entrant_365d_share",
            "incumbent_365d_share",
            "brand_new_platform_share",
        ],
    )
    wide.columns = [f"{metric}_{role}" for metric, role in wide.columns]
    wide = wide.reset_index()

    subtype_panel = pd.read_csv(SUBTYPE_CONSEQUENCE_PANEL)
    exposure_map = pd.read_csv(HARMONIZED_SUBTYPE_PANEL, usecols=["primary_tag", "month_id", "exposure_index"]).drop_duplicates()
    panel = subtype_panel.merge(exposure_map, on=["primary_tag", "month_id"], how="left")
    panel = panel.merge(wide, on=["primary_tag", "month_id", "time_index", "exposure_index"], how="left")

    panel["first_recent_share"] = panel["recent_entrant_90d_share_first_answer"]
    panel["accepted_recent_share"] = panel["recent_entrant_90d_share_accepted_current"]
    panel["recent_gap_first_vs_accepted"] = panel["first_recent_share"] - panel["accepted_recent_share"]
    panel["accepted_incumbent_share"] = panel["incumbent_365d_share_accepted_current"]
    panel["high_complexity_tm"] = (
        panel["residual_queue_complexity_index_mean"] >= panel["residual_queue_complexity_index_mean"].median()
    ).astype(int)
    panel.to_csv(BRIDGE_PANEL_CSV, index=False)
    return panel


def fit_bridge_models(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    outcomes = [
        ("any_answer_7d_rate", "any_answer_7d_denom"),
        ("first_positive_answer_latency_mean", "first_positive_answer_latency_denom"),
        ("accepted_cond_any_answer_30d_rate", "accepted_cond_any_answer_30d_denom"),
        ("accepted_vote_30d_rate", "accepted_vote_30d_denom"),
    ]
    for outcome, weight_col in outcomes:
        base_sub = panel.dropna(subset=[outcome, "exposure_post", weight_col]).copy()
        baseline = smf.wls(
            f"{outcome} ~ exposure_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            data=base_sub,
            weights=base_sub[weight_col],
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": base_sub["primary_tag"], "use_correction": True, "df_correction": True},
        )
        rows.append(
            {
                "sample": "full",
                "model": "baseline",
                "outcome": outcome,
                "term": "exposure_post",
                "coef": float(baseline.params.get("exposure_post", np.nan)),
                "se": float(baseline.bse.get("exposure_post", np.nan)),
                "pval": float(baseline.pvalues.get("exposure_post", np.nan)),
                "weight_col": weight_col,
                "nobs": int(baseline.nobs),
            }
        )

        for candidate in ["first_recent_share", PRIMARY_BRIDGE_VAR, "accepted_incumbent_share"]:
            sub = panel.dropna(subset=[outcome, "exposure_post", candidate, weight_col]).copy()
            model = smf.wls(
                f"{outcome} ~ exposure_post + {candidate} + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
                data=sub,
                weights=sub[weight_col],
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": sub["primary_tag"], "use_correction": True, "df_correction": True},
            )
            rows.extend(
                [
                    {
                        "sample": "full",
                        "model": f"with_{candidate}",
                        "outcome": outcome,
                        "term": candidate,
                        "coef": float(model.params.get(candidate, np.nan)),
                        "se": float(model.bse.get(candidate, np.nan)),
                        "pval": float(model.pvalues.get(candidate, np.nan)),
                        "weight_col": weight_col,
                        "nobs": int(model.nobs),
                    },
                    {
                        "sample": "full",
                        "model": f"with_{candidate}",
                        "outcome": outcome,
                        "term": "exposure_post",
                        "coef": float(model.params.get("exposure_post", np.nan)),
                        "se": float(model.bse.get("exposure_post", np.nan)),
                        "pval": float(model.pvalues.get("exposure_post", np.nan)),
                        "weight_col": weight_col,
                        "nobs": int(model.nobs),
                    },
                ]
            )

        hard = panel.loc[panel["high_complexity_tm"] == 1].dropna(
            subset=[outcome, "exposure_post", PRIMARY_BRIDGE_VAR, weight_col]
        )
        if not hard.empty:
            model = smf.wls(
                f"{outcome} ~ exposure_post + {PRIMARY_BRIDGE_VAR} + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
                data=hard,
                weights=hard[weight_col],
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": hard["primary_tag"], "use_correction": True, "df_correction": True},
            )
            rows.extend(
                [
                    {
                        "sample": "high_complexity",
                        "model": f"with_{PRIMARY_BRIDGE_VAR}",
                        "outcome": outcome,
                        "term": PRIMARY_BRIDGE_VAR,
                        "coef": float(model.params.get(PRIMARY_BRIDGE_VAR, np.nan)),
                        "se": float(model.bse.get(PRIMARY_BRIDGE_VAR, np.nan)),
                        "pval": float(model.pvalues.get(PRIMARY_BRIDGE_VAR, np.nan)),
                        "weight_col": weight_col,
                        "nobs": int(model.nobs),
                    },
                    {
                        "sample": "high_complexity",
                        "model": f"with_{PRIMARY_BRIDGE_VAR}",
                        "outcome": outcome,
                        "term": "exposure_post",
                        "coef": float(model.params.get("exposure_post", np.nan)),
                        "se": float(model.bse.get("exposure_post", np.nan)),
                        "pval": float(model.pvalues.get("exposure_post", np.nan)),
                        "weight_col": weight_col,
                        "nobs": int(model.nobs),
                    },
                ]
            )
    results = pd.DataFrame(rows)
    results.to_csv(BRIDGE_RESULTS_CSV, index=False)
    return results


def leave_two_out_stability(panel: pd.DataFrame) -> pd.DataFrame:
    tags = sorted(panel["primary_tag"].dropna().unique().tolist())
    rows: list[dict[str, object]] = []
    for outcome, weight_col in [
        ("accepted_vote_30d_rate", "accepted_vote_30d_denom"),
        ("accepted_cond_any_answer_30d_rate", "accepted_cond_any_answer_30d_denom"),
    ]:
        base = panel.dropna(subset=[outcome, "exposure_post", PRIMARY_BRIDGE_VAR, weight_col]).copy()
        for dropped in itertools.combinations(tags, 2):
            sub = base.loc[~base["primary_tag"].isin(dropped)].copy()
            if sub["primary_tag"].nunique() < 6:
                continue
            model = smf.wls(
                f"{outcome} ~ exposure_post + {PRIMARY_BRIDGE_VAR} + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
                data=sub,
                weights=sub[weight_col],
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": sub["primary_tag"], "use_correction": True, "df_correction": True},
            )
            rows.append(
                {
                    "outcome": outcome,
                    "dropped_tag_1": dropped[0],
                    "dropped_tag_2": dropped[1],
                    "coef": float(model.params.get(PRIMARY_BRIDGE_VAR, np.nan)),
                    "pval": float(model.pvalues.get(PRIMARY_BRIDGE_VAR, np.nan)),
                    "negative_sign": int(float(model.params.get(PRIMARY_BRIDGE_VAR, np.nan)) < 0),
                    "significant_10pct": int(float(model.pvalues.get(PRIMARY_BRIDGE_VAR, np.nan)) < 0.10),
                }
            )
    out = pd.DataFrame(rows)
    summary_rows = []
    if not out.empty:
        for outcome, frame in out.groupby("outcome"):
            summary_rows.append(
                {
                    "outcome": outcome,
                    "dropped_tag_1": "__summary__",
                    "dropped_tag_2": "__summary__",
                    "coef": float(frame["coef"].mean()),
                    "pval": float(frame["pval"].mean()),
                    "negative_sign": float(frame["negative_sign"].mean()),
                    "significant_10pct": float(frame["significant_10pct"].mean()),
                }
            )
    out = pd.concat([out, pd.DataFrame(summary_rows)], ignore_index=True)
    out.to_csv(BRIDGE_LEAVE_TWO_OUT_CSV, index=False)
    return out


def write_summary(panel: pd.DataFrame, results: pd.DataFrame, leave_two_out: pd.DataFrame) -> dict[str, object]:
    summary = {"n_bridge_cells": int(len(panel)), "primary_bridge_variable": PRIMARY_BRIDGE_VAR, "primary_outcomes": {}, "leave_two_out": {}}
    for outcome in ["first_positive_answer_latency_mean", "accepted_cond_any_answer_30d_rate", "accepted_vote_30d_rate"]:
        row = results.loc[
            (results["sample"] == "full")
            & (results["model"] == f"with_{PRIMARY_BRIDGE_VAR}")
            & (results["outcome"] == outcome)
            & (results["term"] == PRIMARY_BRIDGE_VAR)
        ]
        if not row.empty:
            summary["primary_outcomes"][outcome] = {
                "coef": float(row.iloc[0]["coef"]),
                "pval": float(row.iloc[0]["pval"]),
            }
    for _, row in leave_two_out.loc[leave_two_out["dropped_tag_1"] == "__summary__"].iterrows():
        summary["leave_two_out"][row["outcome"]] = {
            "negative_share": float(row["negative_sign"]),
            "significant_share_10pct": float(row["significant_10pct"]),
        }
    BRIDGE_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def make_figure(results: pd.DataFrame) -> None:
    plot_rows = []
    labels = {
        "first_positive_answer_latency_mean": "First positive latency",
        "accepted_cond_any_answer_30d_rate": "Accepted | any answer (30d)",
        "accepted_vote_30d_rate": "Accepted vote (30d)",
    }
    for outcome, label in labels.items():
        row = results.loc[
            (results["sample"] == "full")
            & (results["model"] == f"with_{PRIMARY_BRIDGE_VAR}")
            & (results["outcome"] == outcome)
            & (results["term"] == PRIMARY_BRIDGE_VAR)
        ]
        if not row.empty:
            item = row.iloc[0].to_dict()
            item["label"] = label
            plot_rows.append(item)
    hard = results.loc[
        (results["sample"] == "high_complexity")
        & (results["model"] == f"with_{PRIMARY_BRIDGE_VAR}")
        & (results["outcome"] == "accepted_vote_30d_rate")
        & (results["term"] == PRIMARY_BRIDGE_VAR)
    ]
    if not hard.empty:
        item = hard.iloc[0].to_dict()
        item["label"] = "Accepted vote (30d), high complexity"
        plot_rows.append(item)
    plot_df = pd.DataFrame(plot_rows)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    y = np.arange(len(plot_df))
    ax.errorbar(
        plot_df["coef"],
        y,
        xerr=1.96 * plot_df["se"],
        fmt="o",
        color="#7a1f2b",
        ecolor="#d1a1aa",
        elinewidth=2,
        capsize=4,
    )
    ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Coefficient on recent-gap (first answer share - accepted share)")
    ax.set_title("Role-gap composition predicts slower or weaker downstream certification")
    fig.tight_layout()
    fig.savefig(BRIDGE_FIGURE, dpi=200)
    plt.close(fig)


def write_readout(results: pd.DataFrame, summary: dict[str, object]) -> None:
    lines = [
        "# Infrastructure Bridge Readout",
        "",
        "## Main Result",
        "",
        "The infrastructure bridge is now supported as an independent consequence layer rather than as a full mediation story.",
        "Tag-months in which recent entrants capture a larger share of first responses than accepted-current certification",
        "show slower endorsed resolution and weaker downstream stopping signals.",
        "",
        f"## Primary bridge variable: `{PRIMARY_BRIDGE_VAR}`",
        "",
    ]
    for outcome, values in summary["primary_outcomes"].items():
        lines.append(f"- `{outcome}`: coef `{values['coef']:.4f}`, p `{values['pval']:.4f}`")
    hard = results.loc[
        (results["sample"] == "high_complexity")
        & (results["model"] == f"with_{PRIMARY_BRIDGE_VAR}")
        & (results["outcome"] == "accepted_vote_30d_rate")
        & (results["term"] == PRIMARY_BRIDGE_VAR)
    ]
    if not hard.empty:
        row = hard.iloc[0]
        lines.extend(
            [
                "",
                "## Harder-queue read",
                "",
                f"- high-complexity `accepted_vote_30d_rate`: coef `{row['coef']:.4f}`, p `{row['pval']:.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Stability Read",
            "",
            f"- leave-two-out negative-share on `accepted_vote_30d_rate`: `{summary['leave_two_out'].get('accepted_vote_30d_rate', {}).get('negative_share', float('nan')):.4f}`",
            f"- leave-two-out negative-share on `accepted_cond_any_answer_30d_rate`: `{summary['leave_two_out'].get('accepted_cond_any_answer_30d_rate', {}).get('negative_share', float('nan')):.4f}`",
            "",
            "## Safe Interpretation",
            "",
            "The bridge evidence does not show that composition fully mediates exposure. The safer claim is narrower:",
            "when rapid-response slots become more entrant-heavy relative to accepted-current certification slots, the tag-month is more",
            "likely to display slower endorsed resolution and weaker visible certification outcomes. That is enough to make composition",
            "reviewer-visible at the platform-consequence level.",
            "",
            f"Figure: `{BRIDGE_FIGURE.name}`",
        ]
    )
    READOUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    panel = load_bridge_panel()
    results = fit_bridge_models(panel)
    leave_two_out = leave_two_out_stability(panel)
    summary = write_summary(panel, results, leave_two_out)
    make_figure(results)
    write_readout(results, summary)


if __name__ == "__main__":
    main()
