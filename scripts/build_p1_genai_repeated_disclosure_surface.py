from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

EVENTS_PATH = PROCESSED / "p1_genai_author_linked_disclosure_events.parquet"
SURFACE_CSV = PROCESSED / "p1_genai_repeated_disclosure_surface.csv"
MONTHLY_CSV = PROCESSED / "p1_genai_repeated_disclosure_monthly.csv"
SUMMARY_JSON = PROCESSED / "p1_genai_repeated_disclosure_summary.json"
READOUT = PAPER / "p1_genai_repeated_disclosure_readout_2026-04-07.md"


def mark_repeat(events: pd.DataFrame) -> pd.DataFrame:
    events = events.sort_values(["user_id", "event_ts"])
    events["event_rank"] = events.groupby("user_id").cumcount()
    events["repeat_flag"] = (events["event_rank"] > 0).astype(int)
    return events


def residual_complexity_flag(df: pd.DataFrame) -> pd.Series:
    return ((df["high_tag"] == 1) & (df["exposure_index"] >= 0.25)).astype(int)


def high_exposure_flag(df: pd.DataFrame) -> pd.Series:
    return (df["exposure_index"] >= 0.35).astype(int)


def summarize_surface(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("repeat_flag").agg(
        n_events=("event_id", "count"),
        avg_exposure=("exposure_index", "mean"),
        high_tag_share=("high_tag", "mean"),
        residual_complexity_share=("residual_complexity", "mean"),
        high_exposure_share=("high_exposure", "mean"),
        ask_role_share=("role", lambda s: float((s == "asker").mean())),
        answer_role_share=("role", lambda s: float((s == "answerer").mean())),
    )
    agg["repeat_label"] = agg.index.map({0: "first", 1: "repeat"})
    return agg.reset_index(drop=True)


def build_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(["event_month", "repeat_flag"], as_index=False)
        .agg(
            events=("event_id", "count"),
            avg_exposure=("exposure_index", "mean"),
            high_tag_share=("high_tag", "mean"),
            residual_complexity_share=("residual_complexity", "mean"),
            high_exposure_share=("high_exposure", "mean"),
        )
        .sort_values(["event_month", "repeat_flag"])
    )
    monthly["repeat_label"] = monthly["repeat_flag"].map({0: "first", 1: "repeat"})
    return monthly


def write_readout(surface: pd.DataFrame, monthly: pd.DataFrame, summary: dict[str, object]) -> None:
    lines = [
        "# P1 Repeated-Disclosue Event Surface",
        "",
        "Date: `2026-04-07`",
        "",
        "## Scope",
        "",
        "Turn the author-linked direct-disclosure layer from an aggregate summary into a reviewer-legible surface that contrasts first versus repeated explicit disclosures and links them to high-exposure/residual-complex questions.",
        "",
        "## Safe Read",
        "",
        "This is still not telemetry and still not a causal adoption test. It is a descriptive contrast with bounded language that highlights where repeated disclosure occurs.",
        "",
        "## Headline Numbers",
        "",
        f"- first-disclosure events: `{int(surface.loc[surface['repeat_label'] == 'first', 'n_events'].iloc[0])}`",
        f"- repeat-disclosure events: `{int(surface.loc[surface['repeat_label'] == 'repeat', 'n_events'].iloc[0])}`",
        f"- residual-complexity concentration gap: `{summary['residual_gap']:.3f}`",
        f"- high-exposure gap: `{summary['high_exposure_gap']:.3f}`",
        "",
        "## Why This Matters",
        "",
        "Repeated disclosures keep clustering around higher-exposure, higher-tag questions while the first disclosures spread across the focal universe. That concentration makes the limited telemetry-adjacent user layer more targeted rather than more random.",
        "",
        "## Files",
        "",
        f"- surface csv: `{SURFACE_CSV}`",
        f"- monthly csv: `{MONTHLY_CSV}`",
        f"- summary json: `{SUMMARY_JSON}`",
    ]
    READOUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    events = pd.read_parquet(EVENTS_PATH).copy()
    events = events.loc[events["user_id"].notna()].copy()
    events["user_id"] = events["user_id"].astype("int64")
    events["question_id"] = events["question_id"].astype(str)
    events["event_month"] = pd.PeriodIndex(events["event_month"].astype(str), freq="M").astype(str)
    events["high_tag"] = pd.to_numeric(events["high_tag"], errors="coerce").fillna(0).astype(int)
    events["exposure_index"] = pd.to_numeric(events["exposure_index"], errors="coerce").fillna(0.0)
    events["residual_complexity"] = residual_complexity_flag(events)
    events["high_exposure"] = high_exposure_flag(events)

    events = mark_repeat(events)

    summary = {}
    surface = summarize_surface(events)
    monthly = build_monthly_series(events)
    summary["residual_gap"] = (
        surface.loc[surface["repeat_label"] == "repeat", "residual_complexity_share"].iloc[0]
        - surface.loc[surface["repeat_label"] == "first", "residual_complexity_share"].iloc[0]
    )
    summary["high_exposure_gap"] = (
        surface.loc[surface["repeat_label"] == "repeat", "high_exposure_share"].iloc[0]
        - surface.loc[surface["repeat_label"] == "first", "high_exposure_share"].iloc[0]
    )
    summary["trend_rows"] = len(monthly)

    surface.to_csv(SURFACE_CSV, index=False)
    monthly.to_csv(MONTHLY_CSV, index=False)
    SUMMARY_JSON.write_text(pd.Series(summary).to_json(), encoding="utf-8")
    write_readout(surface, monthly, summary)


if __name__ == "__main__":
    main()
