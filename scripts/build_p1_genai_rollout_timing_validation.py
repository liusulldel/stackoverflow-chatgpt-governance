from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

ROLL_OUT_INDEX = PROCESSED / "p1_genai_rollout_family_month_index.csv"
TAG_FAMILY_MAP = PROCESSED / "p1_tag_family_map.csv"
SUBTYPE_PANEL = PROCESSED / "p1_jmis_subtype_consequence_panel.csv"
RESIDUAL_PANEL = PROCESSED / "p1_jmis_residual_queue_panel.csv"

JOINED_PANEL = PROCESSED / "p1_genai_rollout_family_month_validation_panel.csv"
WINDOW_SUMMARY = PROCESSED / "p1_genai_rollout_family_month_validation_windows.csv"
MILESTONE_SUMMARY = PROCESSED / "p1_genai_rollout_family_month_validation_milestones.csv"
READOUT = PAPER / "p1_genai_rollout_timing_validation_readout_2026-04-05.md"


def wavg(values: pd.Series, weights: pd.Series) -> float:
    frame = pd.DataFrame({"value": values, "weight": weights}).dropna()
    if frame.empty:
        return np.nan
    weight_sum = frame["weight"].sum()
    if weight_sum <= 0:
        return np.nan
    return float(np.average(frame["value"], weights=frame["weight"]))


def build_family_month_panel() -> pd.DataFrame:
    tag_map = pd.read_csv(TAG_FAMILY_MAP)
    subtype = pd.read_csv(SUBTYPE_PANEL).merge(tag_map, on="primary_tag", how="left")
    residual = pd.read_csv(RESIDUAL_PANEL)[
        [
            "primary_tag",
            "month_id",
            "n_questions",
            "n_new_answerers_profiles",
            "residual_queue_complexity_index_mean",
            "body_word_count_mean",
            "tag_count_full_mean",
        ]
    ]
    panel = subtype.merge(residual, on=["primary_tag", "month_id"], how="left")

    rows: list[dict[str, object]] = []
    for (tag_family, month_id), frame in panel.groupby(["tag_family", "month_id"], sort=True):
        rows.append(
            {
                "tag_family": tag_family,
                "month_id": month_id,
                "queue_residual_complexity_mean": wavg(
                    frame["residual_queue_complexity_index_mean"], frame["n_questions"]
                ),
                "body_word_count_mean": wavg(frame["body_word_count_mean"], frame["n_questions"]),
                "tag_count_full_mean": wavg(frame["tag_count_full_mean"], frame["n_questions"]),
                "brand_new_platform_share": wavg(
                    frame["brand_new_platform_share"], frame["n_new_answerers_profiles"]
                ),
                "low_tenure_existing_share": wavg(
                    frame["low_tenure_existing_share"], frame["n_new_answerers_profiles"]
                ),
                "established_cross_tag_share": wavg(
                    frame["established_cross_tag_share"], frame["n_new_answerers_profiles"]
                ),
                "any_answer_7d_rate": wavg(frame["any_answer_7d_rate"], frame["any_answer_7d_denom"]),
                "first_positive_answer_latency_mean": wavg(
                    frame["first_positive_answer_latency_mean"], frame["first_positive_answer_latency_denom"]
                ),
                "accepted_cond_any_answer_30d_rate": wavg(
                    frame["accepted_cond_any_answer_30d_rate"], frame["accepted_cond_any_answer_30d_denom"]
                ),
                "first_answer_1d_rate_closure": wavg(
                    frame["first_answer_1d_rate_closure"], frame["first_answer_1d_denom_closure"]
                ),
                "accepted_vote_30d_rate": wavg(
                    frame["accepted_vote_30d_rate"], frame["accepted_vote_30d_denom"]
                ),
                "n_questions_sum": float(frame["n_questions"].sum()),
                "n_new_answerers_profiles_sum": float(frame["n_new_answerers_profiles"].sum()),
                "any_answer_7d_denom_sum": float(frame["any_answer_7d_denom"].sum()),
                "first_answer_1d_denom_closure_sum": float(frame["first_answer_1d_denom_closure"].sum()),
                "accepted_vote_30d_denom_sum": float(frame["accepted_vote_30d_denom"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_window_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    joined = joined.copy()
    joined["month_dt"] = pd.to_datetime(joined["month_id"] + "-01")
    for tag_family, frame in joined.groupby("tag_family"):
        family_specific_month = frame["first_family_specific_available_month"].dropna().unique()
        if len(family_specific_month) == 0 or family_specific_month[0] == "":
            continue
        event_month = str(family_specific_month[0])
        event_dt = pd.to_datetime(event_month + "-01")
        pre = frame[(frame["month_dt"] >= event_dt - pd.DateOffset(months=6)) & (frame["month_dt"] < event_dt)]
        post = frame[(frame["month_dt"] >= event_dt) & (frame["month_dt"] < event_dt + pd.DateOffset(months=6))]
        if pre.empty or post.empty:
            continue
        for outcome in [
            "queue_residual_complexity_mean",
            "brand_new_platform_share",
            "established_cross_tag_share",
            "any_answer_7d_rate",
            "first_answer_1d_rate_closure",
            "accepted_vote_30d_rate",
        ]:
            rows.append(
                {
                    "tag_family": tag_family,
                    "event_month": event_month,
                    "window": "pre6_vs_post6",
                    "outcome": outcome,
                    "pre_mean": float(pre[outcome].mean()),
                    "post_mean": float(post[outcome].mean()),
                    "delta_post_minus_pre": float(post[outcome].mean() - pre[outcome].mean()),
                    "pre_n_months": int(pre.shape[0]),
                    "post_n_months": int(post.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def build_milestone_summary(joined: pd.DataFrame) -> pd.DataFrame:
    milestone_months = ["2022-06", "2023-06", "2023-12", "2024-04", "2024-06"]
    cols = [
        "tag_family",
        "month_id",
        "availability_stage_sum",
        "active_tool_count_any",
        "active_family_specific_tool_count_any",
        "queue_residual_complexity_mean",
        "brand_new_platform_share",
        "any_answer_7d_rate",
        "first_answer_1d_rate_closure",
        "accepted_vote_30d_rate",
    ]
    out = joined.loc[joined["month_id"].isin(milestone_months), cols].copy()
    return out.sort_values(["tag_family", "month_id"])


def write_readout(joined: pd.DataFrame, windows: pd.DataFrame, milestones: pd.DataFrame) -> None:
    summary = joined.groupby("tag_family", as_index=False).agg(
        first_family_specific_available_month=("first_family_specific_available_month", "max"),
        max_stage_sum=("availability_stage_sum", "max"),
        max_active_tool_count_any=("active_tool_count_any", "max"),
    )

    lines = [
        "# P1 GenAI Rollout Timing Validation: First Joined Readout",
        "",
        "Date: `2026-04-05`  ",
        "Timezone: `America/New_York`",
        "",
        "## Purpose",
        "",
        "This readout joins the new source-backed `tag_family x month` rollout availability panel to the paper's family-level queue, re-sorting, and consequence surfaces.",
        "",
        "The goal is a bounded timing-validation layer. It is not a causal adoption design and it is not a natural-experiment claim.",
        "",
        "## Construction",
        "",
        "The joined panel merges:",
        "",
        "- [p1_genai_rollout_family_month_index.csv](D:/AI%20alignment/projects/stackoverflow_chatgpt_governance/processed/p1_genai_rollout_family_month_index.csv)",
        "- [p1_jmis_subtype_consequence_panel.csv](D:/AI%20alignment/projects/stackoverflow_chatgpt_governance/processed/p1_jmis_subtype_consequence_panel.csv)",
        "- [p1_jmis_residual_queue_panel.csv](D:/AI%20alignment/projects/stackoverflow_chatgpt_governance/processed/p1_jmis_residual_queue_panel.csv)",
        "- [p1_tag_family_map.csv](D:/AI%20alignment/projects/stackoverflow_chatgpt_governance/processed/p1_tag_family_map.csv)",
        "",
        "The resulting panel is:",
        "",
        f"- `{joined.shape[0]}` family-month observations",
        f"- `{joined['tag_family'].nunique()}` tag families",
        f"- `{joined['month_id'].nunique()}` months",
        "",
        "## Family-Specific Rollout State",
        "",
    ]
    for _, row in summary.sort_values("tag_family").iterrows():
        lines.append(
            f"- `{row['tag_family']}`: first family-specific availability `{row['first_family_specific_available_month']}`, "
            f"max stage sum `{row['max_stage_sum']:.1f}`, max active tools `{int(row['max_active_tool_count_any'])}`."
        )
    lines.extend(
        [
            "",
            "## Headline Bounded Read",
            "",
            "The strongest new improvement is coverage rather than identification. The joined panel now lets the project compare source-backed rollout thickening against family-level public-content outcomes on the same monthly backbone.",
            "",
        ]
    )

    if not windows.empty:
        lines.extend(["For the families with source-backed family-specific rollout events, a simple `pre6` versus `post6` window gives the following bounded read:", ""])
        for tag_family, frame in windows.groupby("tag_family"):
            event_month = frame["event_month"].iloc[0]
            lines.append(f"- `{tag_family}` around `{event_month}`:")
            for outcome in [
                "queue_residual_complexity_mean",
                "brand_new_platform_share",
                "any_answer_7d_rate",
                "first_answer_1d_rate_closure",
                "accepted_vote_30d_rate",
            ]:
                row = frame.loc[frame["outcome"] == outcome]
                if row.empty:
                    continue
                entry = row.iloc[0]
                lines.append(
                    f"  `{outcome}`: `{entry['pre_mean']:.4f} -> {entry['post_mean']:.4f}` "
                    f"(delta `{entry['delta_post_minus_pre']:.4f}`)."
                )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- This panel is now strong enough to support `timing-aligned` or `source-backed availability` language at the family level.",
            "- It is not strong enough to support adoption or exogenous-shock language.",
            "- The best use is as a validation surface: the manuscript can now show that family-specific AI-tool thickening appears on the same monthly backbone as the family-level queue, entrant, and settlement surfaces.",
            "",
            "## Current Limit",
            "",
            "- All three families now have family-specific official events, but each chronology is still sparse enough that the joined surface should be read as source-backed availability timing rather than adoption or access-shock identification.",
            "- OpenAI still lacks a clean local official snapshot under the current workflow, so the rollout panel remains stronger on developer-tool chronology than on consumer-chat chronology.",
            "- The current joined read is descriptive and bounded. A reviewer should read it as timing validation, not as causal effect estimation.",
            "",
            "## Related Files",
            "",
            "- [p1_genai_rollout_family_month_validation_panel.csv](D:/AI%20alignment/projects/stackoverflow_chatgpt_governance/processed/p1_genai_rollout_family_month_validation_panel.csv)",
            "- [p1_genai_rollout_family_month_validation_windows.csv](D:/AI%20alignment/projects/stackoverflow_chatgpt_governance/processed/p1_genai_rollout_family_month_validation_windows.csv)",
            "- [p1_genai_rollout_family_month_validation_milestones.csv](D:/AI%20alignment/projects/stackoverflow_chatgpt_governance/processed/p1_genai_rollout_family_month_validation_milestones.csv)",
            "",
        ]
    )
    READOUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    family_month = build_family_month_panel()
    rollout = pd.read_csv(ROLL_OUT_INDEX)
    joined = family_month.merge(rollout, on=["tag_family", "month_id"], how="left")
    joined["month_dt"] = pd.to_datetime(joined["month_id"] + "-01")
    joined = joined.sort_values(["tag_family", "month_id"]).reset_index(drop=True)

    windows = build_window_summary(joined)
    milestones = build_milestone_summary(joined)

    JOINED_PANEL.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(JOINED_PANEL, index=False)
    windows.to_csv(WINDOW_SUMMARY, index=False)
    milestones.to_csv(MILESTONE_SUMMARY, index=False)
    write_readout(joined, windows, milestones)

    print(f"Wrote {JOINED_PANEL}")
    print(f"Wrote {WINDOW_SUMMARY}")
    print(f"Wrote {MILESTONE_SUMMARY}")
    print(f"Wrote {READOUT}")


if __name__ == "__main__":
    main()
