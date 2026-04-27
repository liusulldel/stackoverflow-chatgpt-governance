from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

BASE_SCRIPT = ROOT / "scripts" / "build_p1_genai_author_linked_event_time.py"

PAIRS_CSV = PROCESSED / "p1_genai_author_linked_asker_conservative_pairs.csv"
BALANCE_CSV = PROCESSED / "p1_genai_author_linked_asker_conservative_balance.csv"
PREPOST_CSV = PROCESSED / "p1_genai_author_linked_asker_conservative_prepost.csv"
DID_CSV = PROCESSED / "p1_genai_author_linked_asker_conservative_did.csv"
SUMMARY_JSON = PROCESSED / "p1_genai_author_linked_asker_conservative_summary.json"
READOUT_PATH = PAPER / "p1_genai_author_linked_asker_conservative_readout_2026-04-06.md"


def load_base_module():
    spec = importlib.util.spec_from_file_location("p1_evt", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_prepost(panel: pd.DataFrame, disclosure: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    treated_first = disclosure.drop_duplicates("user_id", keep="first")[["user_id", "first_direct_ai_month_index"]]
    controls = pairs[["control_user_id", "first_direct_ai_month_index"]].rename(columns={"control_user_id": "user_id"})

    treated_ids = pairs["treated_user_id"].unique()
    treated_panel = panel.loc[panel["user_id"].isin(treated_ids)].merge(
        treated_first, on="user_id", how="left"
    )
    control_panel = panel.loc[panel["user_id"].isin(controls["user_id"])].merge(
        controls, on="user_id", how="left"
    )
    treated_panel["group"] = "treated"
    control_panel["group"] = "control"
    both = pd.concat([treated_panel, control_panel], ignore_index=True)
    both["months_since_event"] = both["month_index"] - both["first_direct_ai_month_index"]
    both = both.loc[both["months_since_event"].between(-6, 6)].copy()
    both["period"] = np.where(both["months_since_event"] < 0, "pre6", "post6")

    return (
        both.groupby(["group", "period"], as_index=False)
        .agg(
            ask_questions=("ask_questions", "mean"),
            ask_high_tag_share=("ask_high_tag_share", "mean"),
            ask_exposure_index_mean=("ask_exposure_index_mean", "mean"),
            answer_count_total=("answer_count_total", "mean"),
            accepted_current_total=("accepted_current_total", "mean"),
            answer_exposure_index_wavg=("answer_exposure_index_wavg", "mean"),
        )
        .sort_values(["group", "period"])
        .reset_index(drop=True)
    )


def build_balance(evt, panel: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "log_pre_ask_questions",
        "pre_ask_high_tag_share",
        "pre_ask_exposure_index_mean",
        "log_pre_answer_count_total",
        "pre_answer_exposure_index_wavg",
        "first_active_month_index",
    ]
    merged_frames = []
    for hit_idx in sorted(pairs["first_direct_ai_month_index"].unique().tolist()):
        features = evt.aggregate_window_features(panel, int(hit_idx) - 6, int(hit_idx) - 1, None)
        features = evt.add_matching_features(features)
        pair_sub = pairs.loc[pairs["first_direct_ai_month_index"] == hit_idx].copy()
        treated_sub = pair_sub.merge(features, left_on="treated_user_id", right_on="user_id", how="left")
        treated_sub = treated_sub.rename(columns={c: f"treated_{c}" for c in feature_cols}).drop(columns=["user_id"])
        merged = treated_sub.merge(features, left_on="control_user_id", right_on="user_id", how="left")
        merged = merged.rename(columns={c: f"control_{c}" for c in feature_cols}).drop(columns=["user_id"])
        merged_frames.append(merged)
    merged = pd.concat(merged_frames, ignore_index=True)

    rows = []
    for col in feature_cols:
        t = pd.to_numeric(merged[f"treated_{col}"], errors="coerce")
        c = pd.to_numeric(merged[f"control_{col}"], errors="coerce")
        scale = np.sqrt((t.var(ddof=0) + c.var(ddof=0)) / 2)
        smd = 0.0 if not np.isfinite(scale) or scale == 0 else float((t.mean() - c.mean()) / scale)
        rows.append(
            {
                "feature": col,
                "treated_mean": float(t.mean()),
                "control_mean": float(c.mean()),
                "smd": smd,
            }
        )
    return pd.DataFrame(rows)


def write_readout(summary: dict[str, object], prepost: pd.DataFrame, did: pd.DataFrame) -> None:
    def pick(group: str, period: str, col: str) -> float:
        row = prepost.loc[(prepost["group"] == group) & (prepost["period"] == period), col]
        return float(row.iloc[0]) if not row.empty else float("nan")

    lines = [
        "# P1 Author-Linked Asker-First Conservative Match",
        "",
        "Date: `2026-04-06`",
        "",
        "## Scope",
        "",
        "This sidecar keeps only users whose first direct-disclosure event is asker-side, then retains only the best-matched quarter of pair distances.",
        "",
        "## Safe Read",
        "",
        "This is a deliberately conservative comparative benchmark. It is not telemetry and not causal adoption identification.",
        "",
        "## Match Summary",
        "",
        f"- asker-first treated users: `{summary['n_asker_treated']:,}`",
        f"- conservative matched treated users: `{summary['n_pairs']:,}`",
        f"- conservative matched controls: `{summary['n_controls']:,}`",
        f"- max absolute standardized mean difference: `{summary['max_abs_smd']:.3f}`",
        "",
        "## Headline Read",
        "",
        f"- treated `ask_questions` pre/post: `{pick('treated', 'pre6', 'ask_questions'):.3f} -> {pick('treated', 'post6', 'ask_questions'):.3f}`",
        f"- control `ask_questions` pre/post: `{pick('control', 'pre6', 'ask_questions'):.3f} -> {pick('control', 'post6', 'ask_questions'):.3f}`",
        f"- treated `answer_count_total` pre/post: `{pick('treated', 'pre6', 'answer_count_total'):.3f} -> {pick('treated', 'post6', 'answer_count_total'):.3f}`",
        f"- control `answer_count_total` pre/post: `{pick('control', 'pre6', 'answer_count_total'):.3f} -> {pick('control', 'post6', 'answer_count_total'):.3f}`",
        "",
        "## Diff-in-Diff (Descriptive, Not Causal)",
        "",
    ]
    for _, row in did.iterrows():
        lines.append(
            f"- `{row['metric']}`: treated `{row['treated_change']:.3f}`, control `{row['control_change']:.3f}`, diff-in-diff `{row['diff_in_diff']:.3f}`"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This is the cleanest telemetry-adjacent comparison currently available in the archive. It supports ask-side reallocation after explicit disclosure without showing a corresponding answer-side expansion.",
        "",
        "## Files",
        "",
        f"- pairs csv: `{PAIRS_CSV}`",
        f"- balance csv: `{BALANCE_CSV}`",
        f"- pre/post csv: `{PREPOST_CSV}`",
        f"- diff-in-diff csv: `{DID_CSV}`",
        f"- summary json: `{SUMMARY_JSON}`",
    ]
    READOUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    evt = load_base_module()
    events = pd.read_parquet(evt.EVENTS_PATH).copy()
    events = events.loc[events["user_id"].notna()].copy()
    events["user_id"] = events["user_id"].astype("int64")
    first_role = (
        events.sort_values(["user_id", "event_ts"])
        .drop_duplicates("user_id", keep="first")[["user_id", "role"]]
        .rename(columns={"role": "first_role"})
    )

    panel = evt.prepare_behavior_panel()
    disclosure = evt.build_disclosure_user_panel(events).merge(first_role, on="user_id", how="left")
    asker = disclosure.loc[disclosure["first_role"] == "asker"].copy()

    pairs, _ = evt.match_controls(panel, asker)
    threshold = float(pairs["distance"].quantile(0.25))
    pairs = pairs.loc[pairs["distance"] <= threshold].copy()

    balance = build_balance(evt, panel, pairs)
    prepost = build_prepost(panel, asker, pairs)
    did = evt.build_matched_did(panel, asker, pairs)

    PAIRS_CSV.write_text(pairs.to_csv(index=False), encoding="utf-8")
    BALANCE_CSV.write_text(balance.to_csv(index=False), encoding="utf-8")
    PREPOST_CSV.write_text(prepost.to_csv(index=False), encoding="utf-8")
    DID_CSV.write_text(did.to_csv(index=False), encoding="utf-8")

    summary = {
        "n_asker_treated": int(asker["user_id"].nunique()),
        "n_pairs": int(len(pairs)),
        "n_controls": int(pairs["control_user_id"].nunique()),
        "distance_threshold_q25": threshold,
        "max_abs_smd": float(balance["smd"].abs().max()) if not balance.empty else None,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readout(summary, prepost, did)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
