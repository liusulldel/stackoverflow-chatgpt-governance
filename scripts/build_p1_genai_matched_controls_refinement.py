from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\AI alignment\projects\stackoverflow_chatgpt_governance")
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

BASE_SCRIPT = ROOT / "scripts" / "build_p1_genai_author_linked_event_time.py"

READOUT_PATH = PAPER / "p1_genai_matched_controls_refinement_readout_2026-04-07.md"


def load_base():
    spec = importlib.util.spec_from_file_location("base_evt", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def subset_by_role(disclosure: pd.DataFrame, role: str) -> pd.DataFrame:
    return disclosure.loc[disclosure["first_role"] == role].copy()


def filter_pairs_by_quantile(pairs: pd.DataFrame, q: float = 0.5) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    threshold = float(pairs["distance"].quantile(q))
    return pairs.loc[pairs["distance"] <= threshold].copy()


def build_balance(base, panel: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
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
        features = base.aggregate_window_features(panel, int(hit_idx) - 6, int(hit_idx) - 1, None)
        features = base.add_matching_features(features)
        pair_sub = pairs.loc[pairs["first_direct_ai_month_index"] == hit_idx].copy()
        treated_sub = pair_sub.merge(features, left_on="treated_user_id", right_on="user_id", how="left")
        treated_sub = treated_sub.rename(columns={c: f"treated_{c}" for c in feature_cols}).drop(columns=["user_id"])
        merged = treated_sub.merge(features, left_on="control_user_id", right_on="user_id", how="left")
        merged = merged.rename(columns={c: f"control_{c}" for c in feature_cols}).drop(columns=["user_id"])
        merged_frames.append(merged)
    if not merged_frames:
        return pd.DataFrame()
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


def export(role: str, pairs: pd.DataFrame, balance: pd.DataFrame, prepost: pd.DataFrame, did: pd.DataFrame) -> dict[str, object]:
    suffix = role.replace("/", "_")
    pair_path = PROCESSED / f"p1_genai_matched_controls_refinement_{suffix}_pairs.csv"
    balance_path = PROCESSED / f"p1_genai_matched_controls_refinement_{suffix}_balance.csv"
    prepost_path = PROCESSED / f"p1_genai_matched_controls_refinement_{suffix}_prepost.csv"
    did_path = PROCESSED / f"p1_genai_matched_controls_refinement_{suffix}_did.csv"

    pairs.to_csv(pair_path, index=False)
    balance.to_csv(balance_path, index=False)
    prepost.to_csv(prepost_path, index=False)
    did.to_csv(did_path, index=False)

    return {
        "role": role,
        "pairs": len(pairs),
        "controls": pairs["control_user_id"].nunique(),
        "smd": float(balance["smd"].abs().max()) if not balance.empty else np.nan,
        "event_time_rows": len(did),
        "pair_path": str(pair_path),
        "prepost_path": str(prepost_path),
        "did_path": str(did_path),
    }


def summarize(results: list[dict[str, object]]) -> None:
    lines = [
        "# P1 Matched Controls Refinement Readout",
        "",
        "Date: `2026-04-07`",
        "",
        "## Scope",
        "",
        "Each role-stratified comparison keeps a conservative subset of matched pairs so the user-layer stays reviewer-legible.",
        "",
        "## Role Results",
        "",
    ]

    for res in results:
        lines.extend(
            [
                f"- **{res['role']}**: {res['pairs']} pairs, {res['controls']} controls, max |SMD| = {res['smd']:.3f}",
                f"  - event-time/diff-in-diff rows: {res['event_time_rows']}",
                f"  - files: pairs={res['pair_path']}, prepost={res['prepost_path']}, did={res['did_path']}",
            ]
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Treat these role-stratified comparisons as conservative descriptive benchmarks. The best polished surface is still the asker-first conservative match, but these role-specific subsets add clarity about whether answer- or comment-side disclosure users show the same ask-focused reallocation pattern.",
        "",
        "## Files",
        "",
        f"- readout: `{READOUT_PATH}`",
    ]
    READOUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build():
    base = load_base()
    panel = base.prepare_behavior_panel()
    events = pd.read_parquet(base.EVENTS_PATH).copy()
    events = events.loc[events["user_id"].notna()].copy()
    events["user_id"] = events["user_id"].astype("int64")
    disclosure = base.build_disclosure_user_panel(events).reset_index()
    first_role = (
        events.sort_values(["user_id", "event_ts"])
        .drop_duplicates("user_id", keep="first")[["user_id", "role"]]
        .rename(columns={"role": "first_role"})
    )
    disclosure = disclosure.merge(first_role, on="user_id", how="left")

    role_defs = ["asker", "answerer", "question_commenter", "answer_commenter"]
    results = []
    for role in role_defs:
        subset = subset_by_role(disclosure, role)
        if subset.empty:
            continue
        pairs, _ = base.match_controls(panel, subset)
        if pairs.empty:
            continue
        pairs = filter_pairs_by_quantile(pairs, 0.5)
        balance = build_balance(base, panel, pairs)
        prepost = base.build_matched_did(panel, subset, pairs)
        did = prepost.copy()
        # Use built-in functions for metrics
        res = export(role, pairs, balance, prepost, did)
        results.append(res)

    summarize(results)


if __name__ == "__main__":
    build()
