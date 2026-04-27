from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\AI alignment\projects\stackoverflow_chatgpt_governance")
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

TAG_FAMILY_MAP = PROCESSED / "p1_tag_family_map.csv"
TAG_MONTH_PANEL = PROCESSED / "who_still_answers_tag_month_entry_panel.csv"
DISCLOSURE_EVENTS = PROCESSED / "p1_genai_author_linked_disclosure_events.parquet"

TAG_MONTH_SUMMARY = PROCESSED / "p1_genai_gemini_sidecar_tag_month_summary.csv"
EVENT_SUMMARY = PROCESSED / "p1_genai_gemini_sidecar_event_summary.csv"
SUMMARY_JSON = PROCESSED / "p1_genai_gemini_sidecar_summary.json"
READOUT = PAPER / "p1_genai_gemini_sidecar_readout_2026-04-07.md"


def summarize_tag_month(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.assign(
        period=lambda df: pd.to_datetime(df["month_id"], format="%Y-%m").dt.to_period("M")
    )
    panel["range"] = panel["period"].astype(str)
    mask = panel["period"] >= pd.Period("2024-04")
    panel["window"] = pd.Series(["post" if m else "pre" for m in mask])
    agg = (
        panel.groupby("window", as_index=False)
        .agg(
            months=("month_id", "nunique"),
            questions=("n_questions", "sum"),
            exposure_index=("exposure_index", "mean"),
            high_tag_share=("high_tag", "mean"),
            first_answer_1d_rate=("first_answer_1d_rate", "mean"),
            accepted_30d_rate=("accepted_30d_rate", "mean"),
            novice_entry_share=("novice_entry_share", "mean"),
            residual_exposure=("exposure_index_exp", "mean"),
            residual_high_tag=("high_tag_exp", "mean"),
        )
        .assign(window=lambda df: df["window"])
    )
    return agg


def describe_event_roles(events: pd.DataFrame) -> pd.DataFrame:
    events = events.sort_values(["user_id", "event_ts"]).copy()
    events["event_rank"] = events.groupby("user_id").cumcount()
    events["is_repeat"] = events["event_rank"] > 0
    events["category"] = events["is_repeat"].map({False: "first", True: "repeat"})
    events["high_exposure"] = events["exposure_index"] > events["exposure_index"].median(skipna=True)
    agg = (
        events.groupby("category", as_index=False)
        .agg(
            events=("event_id", "count"),
            high_tag_fraction=("high_tag", "mean"),
            exposure_index=("exposure_index", "mean"),
            repeated_high=("high_exposure", "mean"),
            answer_share=("role", lambda s: int((s == "answerer").sum()) / (len(s) or 1)),
        )
    )
    return agg


def table_block(df: pd.DataFrame) -> str:
    return "\n```\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n```"


def write_readout(tag_summary: pd.DataFrame, event_summary: pd.DataFrame, summary: dict[str, object]) -> None:
    lines = [
        "# P1 Gemini Narrow Quasi-Shock Sidecar",
        "Date: `2026-04-07`",
        "",
        "## Scope",
        "",
        "A bounded application_framework sidecar anchored on the 2024-04-08 Gemini in Android Studio milestone.",
        "",
        "## Tag-Month Read",
        "",
        table_block(tag_summary),
        "",
        "## Event-Time Read",
        "",
        table_block(event_summary),
        "",
        "## Safe Claim",
        "",
        "Gemini-linked availability makes application_framework tags more exposed and higher-traffic, but this sidecar only measures a bounded availability window and does not purport to identify adoption.",
        "",
        "## Files",
        "",
        f"- tag-month summary csv: `{TAG_MONTH_SUMMARY}`",
        f"- event summary csv: `{EVENT_SUMMARY}`",
        f"- summary json: `{SUMMARY_JSON}`",
    ]
    READOUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    tags = pd.read_csv(TAG_FAMILY_MAP)
    app_tags = tags.loc[tags["tag_family"] == "application_framework", "primary_tag"].tolist()
    panel = pd.read_csv(TAG_MONTH_PANEL)
    panel = panel.loc[panel["primary_tag"].isin(app_tags)].copy()
    tag_summary = summarize_tag_month(panel)
    tag_summary.to_csv(TAG_MONTH_SUMMARY, index=False)

    events = pd.read_parquet(DISCLOSURE_EVENTS)
    events = events.dropna(subset=["primary_tag"])
    events = events.loc[events["primary_tag"].isin(app_tags)].copy()
    event_summary = describe_event_roles(events)
    event_summary.to_csv(EVENT_SUMMARY, index=False)

    repeat_events = event_summary.loc[event_summary["category"] == "repeat", "events"]
    repeat_share = float(repeat_events.iloc[0] / len(events)) if not repeat_events.empty else 0.0
    summary = {
        "tag_month_rows": len(panel),
        "event_rows": len(events),
        "unique_users": int(events["user_id"].nunique()),
        "repeat_share": repeat_share,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readout(tag_summary, event_summary, summary)
    print("Gemini sidecar artifacts written.")


if __name__ == "__main__":
    main()
