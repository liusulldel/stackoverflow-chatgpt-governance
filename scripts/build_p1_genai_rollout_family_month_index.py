from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw" / "external_validation" / "rollout_access"
PROCESSED_DIR = ROOT / "processed"

EVENT_LEDGER_PATH = RAW_DIR / "event_ledger.csv"
TOOL_MAP_PATH = RAW_DIR / "tool_to_tag_family_map_candidate_2026-04-04.csv"
TIMING_SERIES_PATH = PROCESSED_DIR / "p1_jmis_timing_series.csv"

DETAIL_PATH = PROCESSED_DIR / "p1_genai_rollout_family_month_tool_states.csv"
INDEX_PATH = PROCESSED_DIR / "p1_genai_rollout_family_month_index.csv"
SUMMARY_PATH = PROCESSED_DIR / "p1_genai_rollout_family_month_summary.csv"

STAGE_WEIGHTS = {
    "announce": 0.1,
    "preview": 0.3,
    "beta": 0.6,
    "ga": 1.0,
    "gating_change": 1.0,
    "deprecation": 0.0,
}

STAGE_ORDER = {
    "announce": 1,
    "preview": 2,
    "beta": 3,
    "ga": 4,
    "gating_change": 5,
    "deprecation": 6,
}


def split_families(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parts = [part.strip() for part in str(value).split(";")]
    return [part for part in parts if part]


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def expand_mapping_rows(frame: pd.DataFrame, family_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for record in frame.to_dict("records"):
        families = split_families(record.get(family_col))
        if not families:
            continue
        for family in families:
            expanded = dict(record)
            expanded["tag_family"] = family
            rows.append(expanded)
    return pd.DataFrame(rows)


def apply_event_weight(current_weight: float, access_direction: str, stage: str) -> float:
    stage_weight = STAGE_WEIGHTS.get(stage, 0.0)
    if access_direction == "decrease":
        if stage == "deprecation":
            return 0.0
        return min(current_weight, stage_weight)
    return max(current_weight, stage_weight)


def ordered_months(frame: pd.DataFrame) -> list[str]:
    months = sorted(frame["month_id"].dropna().astype(str).unique().tolist())
    return months


def month_to_ts(month_id: str) -> pd.Timestamp:
    return pd.Period(month_id, freq="M").to_timestamp()


def main() -> None:
    events = pd.read_csv(EVENT_LEDGER_PATH)
    tool_map = pd.read_csv(TOOL_MAP_PATH)
    timing = pd.read_csv(TIMING_SERIES_PATH)

    months = ordered_months(timing)
    if not months:
        raise ValueError("No month_id values found in p1_jmis_timing_series.csv")

    tool_map_expanded = expand_mapping_rows(tool_map, "mapped_tag_families")
    events = events.loc[events["event_month"].notna()].copy()
    events_expanded = expand_mapping_rows(events, "mapped_tag_families")

    if events_expanded.empty:
        raise ValueError("No source-backed rollout events with mapped_tag_families found.")

    tool_meta = (
        tool_map_expanded[
            [
                "tool_key",
                "tag_family",
                "tool_family",
                "vendor",
                "product_surface",
                "channel",
                "coverage_scope",
                "mapping_confidence",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    event_meta = (
        events_expanded[
            [
                "tool_key",
                "tag_family",
                "event_id",
                "event_type",
                "rollout_stage",
                "event_month",
                "access_direction",
                "mapping_confidence",
                "evidence_primary_source_id",
            ]
        ]
        .copy()
    )
    event_meta["stage_order"] = event_meta["rollout_stage"].map(STAGE_ORDER).fillna(99)
    event_meta = event_meta.sort_values(["tool_key", "tag_family", "event_month", "stage_order", "event_id"])

    candidate_counts = (
        tool_map_expanded.groupby("tag_family")["tool_key"]
        .nunique()
        .rename("candidate_tool_count_mapped")
        .reset_index()
    )
    source_backed_counts = (
        event_meta.groupby("tag_family")["tool_key"]
        .nunique()
        .rename("source_backed_tool_count_mapped")
        .reset_index()
    )

    tool_family_pairs = (
        tool_meta[["tool_key", "tag_family"]]
        .drop_duplicates()
        .merge(event_meta[["tool_key", "tag_family"]].drop_duplicates(), on=["tool_key", "tag_family"], how="inner")
    )

    detail_rows: list[dict] = []
    for pair in tool_family_pairs.to_dict("records"):
        tool_key = pair["tool_key"]
        tag_family = pair["tag_family"]
        meta_row = tool_meta.loc[
            (tool_meta["tool_key"] == tool_key) & (tool_meta["tag_family"] == tag_family)
        ].iloc[0]
        pair_events = event_meta.loc[
            (event_meta["tool_key"] == tool_key) & (event_meta["tag_family"] == tag_family)
        ].copy()

        current_weight = 0.0
        current_stage = ""
        current_event_id = ""
        current_event_month = ""
        current_source_id = ""

        for month_id in months:
            month_events = pair_events.loc[pair_events["event_month"] == month_id]
            if not month_events.empty:
                for event in month_events.to_dict("records"):
                    current_weight = apply_event_weight(
                        current_weight=current_weight,
                        access_direction=str(event.get("access_direction", "increase")),
                        stage=str(event.get("rollout_stage", "")),
                    )
                    current_stage = str(event.get("rollout_stage", ""))
                    current_event_id = str(event.get("event_id", ""))
                    current_event_month = str(event.get("event_month", ""))
                    current_source_id = str(event.get("evidence_primary_source_id", ""))

            detail_rows.append(
                {
                    "month_id": month_id,
                    "month_dt": month_to_ts(month_id),
                    "tool_key": tool_key,
                    "tag_family": tag_family,
                    "tool_family": meta_row["tool_family"],
                    "vendor": meta_row["vendor"],
                    "product_surface": meta_row["product_surface"],
                    "channel": meta_row["channel"],
                    "coverage_scope": meta_row["coverage_scope"],
                    "mapping_confidence": meta_row["mapping_confidence"],
                    "current_stage_weight": round(float(current_weight), 4),
                    "current_stage": current_stage,
                    "current_event_id": current_event_id,
                    "current_event_month": current_event_month,
                    "current_source_id": current_source_id,
                    "tool_active_any": int(current_weight > 0),
                    "tool_active_full": int(current_weight >= 1.0),
                }
            )

    detail = pd.DataFrame(detail_rows)

    def join_keys(values: Iterable[str]) -> str:
        keys = sorted({str(value) for value in values if str(value)})
        return ";".join(keys)

    family_month = (
        detail.groupby(["month_id", "month_dt", "tag_family"], as_index=False)
        .agg(
            availability_any_binary=("tool_active_any", "max"),
            availability_full_binary=("tool_active_full", "max"),
            availability_stage_max=("current_stage_weight", "max"),
            availability_stage_sum=("current_stage_weight", "sum"),
            active_tool_count_any=("tool_active_any", "sum"),
            active_tool_count_full=("tool_active_full", "sum"),
        )
        .copy()
    )

    active_keys = (
        detail.loc[detail["tool_active_any"] == 1]
        .groupby(["month_id", "tag_family"])["tool_key"]
        .agg(join_keys)
        .rename("active_tool_keys_any")
        .reset_index()
    )
    family_month = family_month.merge(active_keys, on=["month_id", "tag_family"], how="left")
    family_month["active_tool_keys_any"] = family_month["active_tool_keys_any"].fillna("")

    cross_cutting = (
        detail.loc[detail["coverage_scope"] == "cross_cutting"]
        .groupby(["month_id", "tag_family"], as_index=False)
        .agg(active_cross_cutting_tool_count_any=("tool_active_any", "sum"))
    )
    family_specific = (
        detail.loc[detail["coverage_scope"] == "family_specific"]
        .groupby(["month_id", "tag_family"], as_index=False)
        .agg(active_family_specific_tool_count_any=("tool_active_any", "sum"))
    )

    family_month = family_month.merge(cross_cutting, on=["month_id", "tag_family"], how="left")
    family_month = family_month.merge(family_specific, on=["month_id", "tag_family"], how="left")
    family_month["active_cross_cutting_tool_count_any"] = (
        family_month["active_cross_cutting_tool_count_any"].fillna(0).astype(int)
    )
    family_month["active_family_specific_tool_count_any"] = (
        family_month["active_family_specific_tool_count_any"].fillna(0).astype(int)
    )

    first_any = (
        family_month.loc[family_month["availability_any_binary"] == 1]
        .groupby("tag_family")["month_id"]
        .min()
        .rename("first_any_available_month")
        .reset_index()
    )
    first_full = (
        family_month.loc[family_month["availability_full_binary"] == 1]
        .groupby("tag_family")["month_id"]
        .min()
        .rename("first_full_available_month")
        .reset_index()
    )
    first_family_specific = (
        family_month.loc[family_month["active_family_specific_tool_count_any"] > 0]
        .groupby("tag_family")["month_id"]
        .min()
        .rename("first_family_specific_available_month")
        .reset_index()
    )

    family_month = family_month.merge(candidate_counts, on="tag_family", how="left")
    family_month = family_month.merge(source_backed_counts, on="tag_family", how="left")
    family_month = family_month.merge(first_any, on="tag_family", how="left")
    family_month = family_month.merge(first_full, on="tag_family", how="left")
    family_month = family_month.merge(first_family_specific, on="tag_family", how="left")

    family_month["availability_stage_sum"] = family_month["availability_stage_sum"].round(4)
    family_month["availability_stage_max"] = family_month["availability_stage_max"].round(4)
    family_month["candidate_tool_count_mapped"] = family_month["candidate_tool_count_mapped"].fillna(0).astype(int)
    family_month["source_backed_tool_count_mapped"] = (
        family_month["source_backed_tool_count_mapped"].fillna(0).astype(int)
    )

    summary = (
        family_month.groupby("tag_family", as_index=False)
        .agg(
            candidate_tool_count_mapped=("candidate_tool_count_mapped", "max"),
            source_backed_tool_count_mapped=("source_backed_tool_count_mapped", "max"),
            first_any_available_month=("first_any_available_month", "max"),
            first_full_available_month=("first_full_available_month", "max"),
            first_family_specific_available_month=("first_family_specific_available_month", "max"),
            max_stage_sum=("availability_stage_sum", "max"),
            max_stage_max=("availability_stage_max", "max"),
            max_active_tool_count_any=("active_tool_count_any", "max"),
            max_active_family_specific_tool_count_any=("active_family_specific_tool_count_any", "max"),
            end_month_active_tool_count_any=("active_tool_count_any", "last"),
            end_month_stage_sum=("availability_stage_sum", "last"),
            end_month_active_tool_keys_any=("active_tool_keys_any", "last"),
        )
        .sort_values("tag_family")
    )

    ensure_dir(DETAIL_PATH)
    detail.to_csv(DETAIL_PATH, index=False)
    family_month.sort_values(["tag_family", "month_id"]).to_csv(INDEX_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {INDEX_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
