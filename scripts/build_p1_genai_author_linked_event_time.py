from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\AI alignment\projects\stackoverflow_chatgpt_governance")
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

EVENTS_PATH = PROCESSED / "p1_genai_author_linked_disclosure_events.parquet"
QUESTION_PANEL = PROCESSED / "stackexchange_20251231_question_level_enriched.parquet"
ANSWER_PANEL = PROCESSED / "who_still_answers_user_tag_month_panel.parquet"

EVENT_TIME_CSV = PROCESSED / "p1_genai_author_linked_event_time.csv"
PREPOST_CSV = PROCESSED / "p1_genai_author_linked_prepost.csv"
MATCHED_PAIRS_CSV = PROCESSED / "p1_genai_author_linked_matched_pairs.csv"
MATCHED_BALANCE_CSV = PROCESSED / "p1_genai_author_linked_matched_balance.csv"
MATCHED_DID_CSV = PROCESSED / "p1_genai_author_linked_matched_did.csv"
SUMMARY_JSON = PROCESSED / "p1_genai_author_linked_event_time_summary.json"

READOUT_PATH = PAPER / "p1_genai_author_linked_event_time_readout_2026-04-06.md"


def month_to_index(month_id: pd.Series) -> pd.Series:
    return pd.PeriodIndex(month_id.astype(str), freq="M").asi8


def prepare_behavior_panel() -> pd.DataFrame:
    q = pd.read_parquet(
        QUESTION_PANEL,
        columns=[
            "owner_user_id",
            "month_id",
            "question_id",
            "high_tag",
            "exposure_index",
            "primary_tag",
            "keep_single_focal",
        ],
    ).copy()
    q = q.loc[q["keep_single_focal"] == 1].copy()
    q["owner_user_id"] = pd.to_numeric(q["owner_user_id"], errors="coerce").astype("Int64")
    q = q.loc[q["owner_user_id"].notna()].copy()
    q["user_id"] = q["owner_user_id"].astype("int64")
    q["month_id"] = q["month_id"].astype(str)
    q["month_index"] = month_to_index(q["month_id"])

    ask = (
        q.groupby(["user_id", "month_id", "month_index"], as_index=False)
        .agg(
            ask_questions=("question_id", "size"),
            ask_high_tag_questions=("high_tag", "sum"),
            ask_exposure_sum=("exposure_index", "sum"),
            ask_primary_tags_nunique=("primary_tag", "nunique"),
        )
        .sort_values(["user_id", "month_index"])
        .reset_index(drop=True)
    )
    ask["ask_high_tag_share"] = np.where(
        ask["ask_questions"] > 0,
        ask["ask_high_tag_questions"] / ask["ask_questions"],
        0.0,
    )
    ask["ask_exposure_index_mean"] = np.where(
        ask["ask_questions"] > 0,
        ask["ask_exposure_sum"] / ask["ask_questions"],
        np.nan,
    )

    a = pd.read_parquet(ANSWER_PANEL).copy()
    a["answerer_user_id"] = pd.to_numeric(a["answerer_user_id"], errors="coerce").astype("Int64")
    a = a.loc[a["answerer_user_id"].notna()].copy()
    a["user_id"] = a["answerer_user_id"].astype("int64")
    a["month_id"] = a["month_id"].astype(str)
    a["month_index"] = month_to_index(a["month_id"])
    a["answer_count"] = pd.to_numeric(a["answer_count"], errors="coerce").fillna(0.0)
    a["accepted_current_count"] = pd.to_numeric(a["accepted_current_count"], errors="coerce").fillna(0.0)
    a["exposure_index"] = pd.to_numeric(a["exposure_index"], errors="coerce")
    a["answer_exposure_num"] = a["answer_count"] * a["exposure_index"].fillna(0.0)

    answer = (
        a.groupby(["user_id", "month_id", "month_index"], as_index=False)
        .agg(
            answer_count_total=("answer_count", "sum"),
            accepted_current_total=("accepted_current_count", "sum"),
            answer_exposure_num=("answer_exposure_num", "sum"),
        )
        .sort_values(["user_id", "month_index"])
        .reset_index(drop=True)
    )
    answer["answer_exposure_index_wavg"] = np.where(
        answer["answer_count_total"] > 0,
        answer["answer_exposure_num"] / answer["answer_count_total"],
        np.nan,
    )

    panel = ask.merge(answer, on=["user_id", "month_id", "month_index"], how="outer")
    for col in [
        "ask_questions",
        "ask_high_tag_questions",
        "ask_exposure_sum",
        "ask_primary_tags_nunique",
        "ask_high_tag_share",
        "answer_count_total",
        "accepted_current_total",
        "answer_exposure_num",
    ]:
        panel[col] = panel[col].fillna(0.0)

    panel = panel.sort_values(["user_id", "month_index"]).reset_index(drop=True)
    panel["first_active_month_index"] = panel.groupby("user_id")["month_index"].transform("min")
    return panel


def build_disclosure_user_panel(events: pd.DataFrame) -> pd.DataFrame:
    events = events.loc[events["user_id"].notna()].copy()
    events["user_id"] = events["user_id"].astype("int64")
    disclosure = (
        events.groupby(["user_id", "event_month"], as_index=False)
        .agg(
            direct_ai_events=("event_id", "size"),
            answer_side_direct_ai_events=("role", lambda s: int((s == "answerer").sum())),
            ask_side_direct_ai_events=("role", lambda s: int((s == "asker").sum())),
            comment_side_direct_ai_events=("role", lambda s: int(s.isin(["question_commenter", "answer_commenter"]).sum())),
            distinct_tools=("tool_family_labels", lambda s: int(pd.Series(s).replace("", np.nan).dropna().nunique())),
        )
        .sort_values(["user_id", "event_month"])
        .reset_index(drop=True)
    )
    first = disclosure.drop_duplicates("user_id", keep="first")[["user_id", "event_month"]].rename(
        columns={"event_month": "first_direct_ai_month"}
    )
    disclosure = disclosure.merge(first, on="user_id", how="left")
    disclosure["event_month_index"] = month_to_index(disclosure["event_month"])
    disclosure["first_direct_ai_month_index"] = month_to_index(disclosure["first_direct_ai_month"])
    disclosure["months_since_first_direct_ai"] = (
        disclosure["event_month_index"] - disclosure["first_direct_ai_month_index"]
    )
    disclosure["cumulative_direct_ai_events"] = disclosure.groupby("user_id")["direct_ai_events"].cumsum()
    disclosure["repeat_disclosure_month"] = (
        disclosure["cumulative_direct_ai_events"] > disclosure["direct_ai_events"]
    ).astype(int)
    return disclosure


def build_event_time(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ask_questions",
        "ask_high_tag_share",
        "ask_exposure_index_mean",
        "answer_count_total",
        "accepted_current_total",
        "answer_exposure_index_wavg",
        "direct_ai_events",
    ]
    keep = panel.loc[panel["months_since_first_direct_ai"].between(-6, 12)].copy()
    agg = keep.groupby("months_since_first_direct_ai", as_index=False)[cols].mean()
    agg["n_user_months"] = keep.groupby("months_since_first_direct_ai").size().values
    return agg


def build_prepost(panel: pd.DataFrame) -> pd.DataFrame:
    temp = panel.loc[panel["months_since_first_direct_ai"].between(-6, 6)].copy()
    temp["period"] = np.where(temp["months_since_first_direct_ai"] < 0, "pre6", "post6")
    agg = (
        temp.groupby("period", as_index=False)
        .agg(
            ask_questions=("ask_questions", "mean"),
            ask_high_tag_share=("ask_high_tag_share", "mean"),
            ask_exposure_index_mean=("ask_exposure_index_mean", "mean"),
            answer_count_total=("answer_count_total", "mean"),
            accepted_current_total=("accepted_current_total", "mean"),
            answer_exposure_index_wavg=("answer_exposure_index_wavg", "mean"),
            direct_ai_events=("direct_ai_events", "mean"),
        )
    )
    return agg


def aggregate_window_features(panel: pd.DataFrame, start: int, end: int, user_ids: np.ndarray) -> pd.DataFrame:
    mask = panel["month_index"].between(start, end)
    if user_ids is not None:
        mask &= panel["user_id"].isin(user_ids)
    sub = panel.loc[mask].copy()
    features = (
        sub.groupby("user_id", as_index=False)
        .agg(
            pre_ask_questions=("ask_questions", "sum"),
            pre_ask_high_tag_questions=("ask_high_tag_questions", "sum"),
            pre_ask_exposure_sum=("ask_exposure_sum", "sum"),
            pre_ask_high_tag_share=("ask_high_tag_share", "mean"),
            pre_ask_exposure_index_mean=("ask_exposure_index_mean", "mean"),
            pre_answer_count_total=("answer_count_total", "sum"),
            pre_answer_exposure_num=("answer_exposure_num", "sum"),
            pre_answer_exposure_index_wavg=("answer_exposure_index_wavg", "mean"),
            first_active_month_index=("first_active_month_index", "min"),
        )
    )
    features["pre_ask_questions"] = features["pre_ask_questions"].fillna(0.0)
    features["pre_answer_count_total"] = features["pre_answer_count_total"].fillna(0.0)
    features["pre_ask_high_tag_questions"] = features["pre_ask_high_tag_questions"].fillna(0.0)
    features["pre_ask_exposure_sum"] = features["pre_ask_exposure_sum"].fillna(0.0)
    features["pre_answer_exposure_num"] = features["pre_answer_exposure_num"].fillna(0.0)
    features["pre_ask_high_tag_share"] = np.where(
        features["pre_ask_questions"] > 0,
        features["pre_ask_high_tag_questions"] / features["pre_ask_questions"],
        0.0,
    )
    features["pre_ask_exposure_index_mean"] = np.where(
        features["pre_ask_questions"] > 0,
        features["pre_ask_exposure_sum"] / features["pre_ask_questions"],
        0.0,
    )
    features["pre_answer_exposure_index_wavg"] = np.where(
        features["pre_answer_count_total"] > 0,
        features["pre_answer_exposure_num"] / features["pre_answer_count_total"],
        0.0,
    )
    if sub.empty:
        features["first_active_month_index"] = features["first_active_month_index"].fillna(0)
    else:
        features["first_active_month_index"] = features["first_active_month_index"].fillna(sub["month_index"].min())
    return features


def add_matching_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["log_pre_ask_questions"] = np.log1p(out["pre_ask_questions"].fillna(0.0))
    out["log_pre_answer_count_total"] = np.log1p(out["pre_answer_count_total"].fillna(0.0))
    out["first_active_bucket"] = pd.cut(
        out["first_active_month_index"],
        bins=[-np.inf, month_to_index(pd.Series(["2021-12"]))[0], month_to_index(pd.Series(["2023-12"]))[0], np.inf],
        labels=["early", "mid", "late"],
    ).astype(str)
    out["ask_bin"] = pd.cut(
        out["log_pre_ask_questions"],
        bins=[-0.01, 0.01, 0.7, 1.6, 2.6, np.inf],
        labels=["zero", "one", "low", "mid", "high"],
    ).astype(str)
    out["answer_bin"] = pd.cut(
        out["log_pre_answer_count_total"],
        bins=[-0.01, 0.01, 0.7, 1.6, 2.6, np.inf],
        labels=["zero", "one", "low", "mid", "high"],
    ).astype(str)
    out["high_tag_bin"] = pd.cut(
        out["pre_ask_high_tag_share"].fillna(0.0),
        bins=[-0.01, 0.05, 0.33, 0.66, 1.01],
        labels=["none", "low", "mid", "high"],
    ).astype(str)
    out["exposure_bin"] = pd.cut(
        out["pre_ask_exposure_index_mean"].fillna(0.0),
        bins=[-0.01, 0.05, 0.15, 0.30, np.inf],
        labels=["none", "low", "mid", "high"],
    ).astype(str)
    out["activity_class"] = "inactive"
    out.loc[
        (out["pre_ask_questions"] > 0) & (out["pre_answer_count_total"] == 0),
        "activity_class",
    ] = "ask_only"
    out.loc[
        (out["pre_ask_questions"] == 0) & (out["pre_answer_count_total"] > 0),
        "activity_class",
    ] = "answer_only"
    out.loc[
        (out["pre_ask_questions"] > 0)
        & (out["pre_answer_count_total"] > 0)
        & (out["pre_ask_questions"] >= 2 * out["pre_answer_count_total"]),
        "activity_class",
    ] = "ask_dominant"
    out.loc[
        (out["pre_ask_questions"] > 0)
        & (out["pre_answer_count_total"] > 0)
        & (out["pre_answer_count_total"] >= 2 * out["pre_ask_questions"]),
        "activity_class",
    ] = "answer_dominant"
    out.loc[
        (out["pre_ask_questions"] > 0)
        & (out["pre_answer_count_total"] > 0)
        & (out["activity_class"] == "inactive"),
        "activity_class",
    ] = "mixed"
    return out


def standardized_distance(pool: pd.DataFrame, target: pd.Series, feature_cols: list[str]) -> pd.Series:
    numeric_pool = pool[feature_cols].astype(float).copy()
    center = numeric_pool.mean(axis=0)
    scale = numeric_pool.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    target_vals = target[feature_cols].astype(float).copy().fillna(center)
    numeric_pool = numeric_pool.fillna(center)
    distances = ((numeric_pool - target_vals) / scale) ** 2
    return distances.sum(axis=1)


def match_controls(panel: pd.DataFrame, disclosure: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    treated = disclosure.drop_duplicates("user_id", keep="first")[["user_id", "first_direct_ai_month", "first_direct_ai_month_index"]].copy()
    all_treated_ids = set(treated["user_id"].astype(int).tolist())
    pairs = []
    balance_rows = []
    feature_cols = [
        "log_pre_ask_questions",
        "pre_ask_high_tag_share",
        "pre_ask_exposure_index_mean",
        "log_pre_answer_count_total",
        "pre_answer_exposure_index_wavg",
        "first_active_month_index",
    ]

    for hit_idx in sorted(treated["first_direct_ai_month_index"].unique().tolist()):
        treated_month = treated.loc[treated["first_direct_ai_month_index"] == hit_idx].copy()
        features = aggregate_window_features(panel, int(hit_idx) - 6, int(hit_idx) - 1, None)
        features = add_matching_features(features)
        treated_feat = treated_month.merge(features, on="user_id", how="left")
        candidate_feat = features.loc[~features["user_id"].isin(all_treated_ids)].copy()
        candidate_feat = candidate_feat.loc[
            (candidate_feat["pre_ask_questions"] > 0) | (candidate_feat["pre_answer_count_total"] > 0)
        ].copy()
        used_control_ids: set[int] = set()

        for treated_row in treated_feat.to_dict("records"):
            pool = candidate_feat.loc[~candidate_feat["user_id"].isin(used_control_ids)].copy()
            if pool.empty:
                continue

            exact_bucket = pool.loc[pool["first_active_bucket"] == treated_row["first_active_bucket"]].copy()
            if not exact_bucket.empty:
                pool = exact_bucket

            exact_activity = pool.loc[pool["activity_class"] == treated_row["activity_class"]].copy()
            if not exact_activity.empty:
                pool = exact_activity

            exact_ask_bin = pool.loc[pool["ask_bin"] == treated_row["ask_bin"]].copy()
            if not exact_ask_bin.empty:
                pool = exact_ask_bin

            exact_answer_bin = pool.loc[pool["answer_bin"] == treated_row["answer_bin"]].copy()
            if not exact_answer_bin.empty:
                pool = exact_answer_bin

            exact_high_tag_bin = pool.loc[pool["high_tag_bin"] == treated_row["high_tag_bin"]].copy()
            if not exact_high_tag_bin.empty:
                pool = exact_high_tag_bin

            exact_exposure_bin = pool.loc[pool["exposure_bin"] == treated_row["exposure_bin"]].copy()
            if not exact_exposure_bin.empty:
                pool = exact_exposure_bin

            pool = pool.reset_index(drop=True)
            distances = standardized_distance(pool, pd.Series(treated_row), feature_cols)
            best_idx = int(distances.idxmin())
            best = pool.iloc[best_idx]
            used_control_ids.add(int(best["user_id"]))
            pairs.append(
                {
                    "treated_user_id": int(treated_row["user_id"]),
                    "control_user_id": int(best["user_id"]),
                    "first_direct_ai_month_index": int(hit_idx),
                    "first_direct_ai_month": str(pd.Period(ordinal=int(hit_idx), freq="M")),
                    "distance": float(distances.iloc[best_idx]),
                }
            )

    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        return pairs_df, pd.DataFrame()

    merged_frames = []
    for hit_idx in sorted(pairs_df["first_direct_ai_month_index"].unique().tolist()):
        features = aggregate_window_features(panel, int(hit_idx) - 6, int(hit_idx) - 1, None)
        features = add_matching_features(features)
        pair_sub = pairs_df.loc[pairs_df["first_direct_ai_month_index"] == hit_idx].copy()
        treated_sub = pair_sub.merge(features, left_on="treated_user_id", right_on="user_id", how="left")
        treated_sub = treated_sub.rename(columns={c: f"treated_{c}" for c in feature_cols})
        treated_sub = treated_sub.drop(columns=["user_id"])
        merged = treated_sub.merge(features, left_on="control_user_id", right_on="user_id", how="left")
        merged = merged.rename(columns={c: f"control_{c}" for c in feature_cols})
        merged = merged.drop(columns=["user_id"])
        merged_frames.append(merged)
    merged = pd.concat(merged_frames, ignore_index=True)

    for col in feature_cols:
        t = pd.to_numeric(merged[f"treated_{col}"], errors="coerce")
        c = pd.to_numeric(merged[f"control_{col}"], errors="coerce")
        scale = np.sqrt((t.var(ddof=0) + c.var(ddof=0)) / 2) if len(merged) else np.nan
        smd = 0.0 if not np.isfinite(scale) or scale == 0 else float((t.mean() - c.mean()) / scale)
        balance_rows.append(
            {
                "feature": col,
                "treated_mean": float(t.mean()),
                "control_mean": float(c.mean()),
                "smd": smd,
            }
        )
    return pairs_df, pd.DataFrame(balance_rows)


def build_matched_did(panel: pd.DataFrame, disclosure: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()

    treated_first = disclosure.drop_duplicates("user_id", keep="first")[["user_id", "first_direct_ai_month_index"]]

    controls = pairs[["control_user_id", "first_direct_ai_month_index"]].rename(
        columns={"control_user_id": "user_id"}
    )

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

    agg = (
        both.groupby(["group", "period"], as_index=False)
        .agg(
            ask_questions=("ask_questions", "mean"),
            ask_high_tag_share=("ask_high_tag_share", "mean"),
            ask_exposure_index_mean=("ask_exposure_index_mean", "mean"),
            answer_count_total=("answer_count_total", "mean"),
            accepted_current_total=("accepted_current_total", "mean"),
            answer_exposure_index_wavg=("answer_exposure_index_wavg", "mean"),
        )
    )

    rows = []
    metrics = [
        "ask_questions",
        "ask_high_tag_share",
        "ask_exposure_index_mean",
        "answer_count_total",
        "accepted_current_total",
        "answer_exposure_index_wavg",
    ]
    for metric in metrics:
        t_pre = float(agg.loc[(agg["group"] == "treated") & (agg["period"] == "pre6"), metric].iloc[0])
        t_post = float(agg.loc[(agg["group"] == "treated") & (agg["period"] == "post6"), metric].iloc[0])
        c_pre = float(agg.loc[(agg["group"] == "control") & (agg["period"] == "pre6"), metric].iloc[0])
        c_post = float(agg.loc[(agg["group"] == "control") & (agg["period"] == "post6"), metric].iloc[0])
        rows.append(
            {
                "metric": metric,
                "treated_pre": t_pre,
                "treated_post": t_post,
                "control_pre": c_pre,
                "control_post": c_post,
                "treated_change": t_post - t_pre,
                "control_change": c_post - c_pre,
                "diff_in_diff": (t_post - t_pre) - (c_post - c_pre),
            }
        )
    return pd.DataFrame(rows)


def write_readout(summary: dict[str, object]) -> None:
    lines = [
        "# P1 Author-Linked Direct-Disclosure Event-Time Layer",
        "",
        "Date: `2026-04-06`",
        "",
        "## Scope",
        "",
        "This build places first direct-disclosure episodes, repeated disclosure months, and matched non-disclosure controls on top of the new author-linked direct-disclosure event panel.",
        "",
        "## Safe Read",
        "",
        "This is still not telemetry and not causal adoption identification. It is a stronger reviewer-legible user-linked disclosure-event layer.",
        "",
        "## Headline Counts",
        "",
        f"- treated users with at least one direct-disclosure event: `{summary['n_treated_users']:,}`",
        f"- matched treated users: `{summary['n_matched_treated']:,}`",
        f"- matched controls: `{summary['n_matched_controls']:,}`",
        "",
        "## Why This Matters",
        "",
        "The layer now distinguishes first explicit direct-disclosure episodes from later repeated disclosure months and adds a matched non-disclosure benchmark. That makes the user-linked GenAI layer more reviewer-legible than a pure proxy count.",
        "",
        "## Files",
        "",
        f"- event-time csv: `{EVENT_TIME_CSV}`",
        f"- pre/post csv: `{PREPOST_CSV}`",
        f"- matched pairs csv: `{MATCHED_PAIRS_CSV}`",
        f"- matched balance csv: `{MATCHED_BALANCE_CSV}`",
        f"- matched diff-in-diff csv: `{MATCHED_DID_CSV}`",
        f"- summary json: `{SUMMARY_JSON}`",
    ]
    READOUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    events = pd.read_parquet(EVENTS_PATH).copy()
    events = events.loc[events["user_id"].notna()].copy()
    events["user_id"] = events["user_id"].astype("int64")

    panel = prepare_behavior_panel()
    disclosure = build_disclosure_user_panel(events)

    treated = disclosure.drop_duplicates("user_id", keep="first")[["user_id", "first_direct_ai_month", "first_direct_ai_month_index"]].copy()
    treated_panel = panel.merge(treated, on="user_id", how="inner")
    treated_panel["months_since_first_direct_ai"] = (
        treated_panel["month_index"] - treated_panel["first_direct_ai_month_index"]
    )
    treated_panel = treated_panel.merge(
        disclosure[["user_id", "event_month", "direct_ai_events", "repeat_disclosure_month"]].rename(columns={"event_month": "month_id"}),
        on=["user_id", "month_id"],
        how="left",
    )
    treated_panel["direct_ai_events"] = treated_panel["direct_ai_events"].fillna(0.0)
    treated_panel["repeat_disclosure_month"] = treated_panel["repeat_disclosure_month"].fillna(0).astype(int)

    event_time = build_event_time(treated_panel)
    prepost = build_prepost(treated_panel)
    pairs, balance = match_controls(panel, disclosure)
    did = build_matched_did(panel, disclosure, pairs)

    event_time.to_csv(EVENT_TIME_CSV, index=False)
    prepost.to_csv(PREPOST_CSV, index=False)
    pairs.to_csv(MATCHED_PAIRS_CSV, index=False)
    balance.to_csv(MATCHED_BALANCE_CSV, index=False)
    did.to_csv(MATCHED_DID_CSV, index=False)

    summary = {
        "n_treated_users": int(treated["user_id"].nunique()),
        "n_matched_treated": int(pairs["treated_user_id"].nunique()) if not pairs.empty else 0,
        "n_matched_controls": int(pairs["control_user_id"].nunique()) if not pairs.empty else 0,
        "event_time_rows": int(len(event_time)),
        "matched_did_rows": int(len(did)),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readout(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
