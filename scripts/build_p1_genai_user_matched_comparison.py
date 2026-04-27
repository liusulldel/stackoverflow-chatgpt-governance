from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

ASK_PANEL_PATH = PROCESSED / "p1_genai_user_level_proxy_panel.parquet"
STRICT_HIT_USERS_PATH = PROCESSED / "p1_genai_user_level_proxy_ai_hit_users.csv"
ANSWER_PANEL_PATH = PROCESSED / "who_still_answers_user_tag_month_panel.parquet"

MATCHED_PAIRS_PATH = PROCESSED / "p1_genai_user_matched_pairs.csv"
MATCHED_BALANCE_PATH = PROCESSED / "p1_genai_user_matched_balance.csv"
MATCHED_EVENT_TIME_PATH = PROCESSED / "p1_genai_user_matched_event_time.csv"
MATCHED_PREPOST_PATH = PROCESSED / "p1_genai_user_matched_prepost.csv"
MATCHED_DID_PATH = PROCESSED / "p1_genai_user_matched_did.csv"
READOUT_PATH = PAPER / "p1_genai_user_matched_comparison_readout_2026-04-05.md"


def month_to_index(month_id: pd.Series) -> pd.Series:
    return pd.PeriodIndex(month_id.astype(str), freq="M").asi8


def index_to_month(idx: int) -> str:
    return str(pd.Period(ordinal=int(idx), freq="M"))


def weighted_mean_sum(value: pd.Series, weight: pd.Series) -> float:
    frame = pd.DataFrame({"value": value, "weight": weight}).dropna()
    if frame.empty:
        return 0.0
    weight_sum = frame["weight"].sum()
    if weight_sum <= 0:
        return 0.0
    return float(np.average(frame["value"], weights=frame["weight"]))


def prepare_ask_panel() -> pd.DataFrame:
    ask = pd.read_parquet(ASK_PANEL_PATH).copy()
    ask["owner_user_id"] = pd.to_numeric(ask["owner_user_id"], errors="coerce")
    ask = ask.loc[ask["owner_user_id"].notna()].copy()
    ask["user_id"] = ask["owner_user_id"].astype("int64")
    ask["month_id"] = ask["month_id"].astype(str)
    ask["month_index"] = month_to_index(ask["month_id"])
    ask["questions"] = pd.to_numeric(ask["questions"], errors="coerce").fillna(0.0)
    ask["high_tag_questions"] = pd.to_numeric(ask["high_tag_questions"], errors="coerce").fillna(0.0)
    ask["mean_exposure_index"] = pd.to_numeric(ask["mean_exposure_index"], errors="coerce")
    ask["any_genai_proxy_user_month"] = (
        pd.to_numeric(ask["any_genai_proxy_user_month"], errors="coerce").fillna(0).astype(int)
    )
    ask["ask_exposure_num"] = ask["mean_exposure_index"].fillna(0.0) * ask["questions"]
    return ask[
        [
            "user_id",
            "month_id",
            "month_index",
            "questions",
            "high_tag_questions",
            "mean_exposure_index",
            "ask_exposure_num",
            "any_genai_proxy_user_month",
        ]
    ].copy()


def prepare_answer_panel() -> pd.DataFrame:
    answer = pd.read_parquet(
        ANSWER_PANEL_PATH,
        columns=[
            "answerer_user_id",
            "month_id",
            "answer_count",
            "accepted_current_count",
            "exposure_index",
        ],
    ).copy()
    answer["answerer_user_id"] = pd.to_numeric(answer["answerer_user_id"], errors="coerce")
    answer = answer.loc[answer["answerer_user_id"].notna()].copy()
    answer["user_id"] = answer["answerer_user_id"].astype("int64")
    answer["month_id"] = answer["month_id"].astype(str)
    answer["month_index"] = month_to_index(answer["month_id"])
    answer["answer_count"] = pd.to_numeric(answer["answer_count"], errors="coerce").fillna(0.0)
    answer["accepted_current_count"] = pd.to_numeric(
        answer["accepted_current_count"], errors="coerce"
    ).fillna(0.0)
    answer["exposure_index"] = pd.to_numeric(answer["exposure_index"], errors="coerce")
    answer["answer_exposure_num"] = answer["answer_count"] * answer["exposure_index"].fillna(0.0)

    monthly = (
        answer.groupby(["user_id", "month_id", "month_index"], as_index=False)
        .agg(
            answer_count_total=("answer_count", "sum"),
            accepted_current_total=("accepted_current_count", "sum"),
            answer_exposure_num=("answer_exposure_num", "sum"),
        )
        .sort_values(["user_id", "month_index"])
        .reset_index(drop=True)
    )
    monthly["answer_exposure_index_wavg"] = np.where(
        monthly["answer_count_total"] > 0,
        monthly["answer_exposure_num"] / monthly["answer_count_total"],
        np.nan,
    )
    return monthly[
        [
            "user_id",
            "month_id",
            "month_index",
            "answer_count_total",
            "accepted_current_total",
            "answer_exposure_index_wavg",
        ]
    ].copy()


def aggregate_window_features(
    ask: pd.DataFrame,
    answer: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    user_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    ask_sub = ask.loc[(ask["month_index"] >= start_idx) & (ask["month_index"] <= end_idx)].copy()
    ans_sub = answer.loc[(answer["month_index"] >= start_idx) & (answer["month_index"] <= end_idx)].copy()

    if user_ids is not None:
        ask_sub = ask_sub.loc[ask_sub["user_id"].isin(user_ids)].copy()
        ans_sub = ans_sub.loc[ans_sub["user_id"].isin(user_ids)].copy()

    ask_agg = (
        ask_sub.groupby("user_id", as_index=False)
        .agg(
            pre_ask_questions=("questions", "sum"),
            pre_high_tag_questions=("high_tag_questions", "sum"),
            pre_ask_exposure_num=("ask_exposure_num", "sum"),
        )
        .copy()
    )
    ask_agg["pre_ask_exposure_mean"] = np.where(
        ask_agg["pre_ask_questions"] > 0,
        ask_agg["pre_ask_exposure_num"] / ask_agg["pre_ask_questions"],
        np.nan,
    )
    ask_agg["pre_high_tag_share"] = np.where(
        ask_agg["pre_ask_questions"] > 0,
        ask_agg["pre_high_tag_questions"] / ask_agg["pre_ask_questions"],
        0.0,
    )

    ans_agg = (
        ans_sub.groupby("user_id", as_index=False)
        .agg(
            pre_answer_count=("answer_count_total", "sum"),
            pre_accepted_current=("accepted_current_total", "sum"),
            pre_answer_exposure_num=("answer_count_total", lambda s: 0.0),
        )
        .copy()
    )
    if not ans_sub.empty:
        tmp = (
            ans_sub.assign(
                answer_exposure_num=ans_sub["answer_exposure_index_wavg"].fillna(0.0)
                * ans_sub["answer_count_total"]
            )
            .groupby("user_id", as_index=False)
            .agg(pre_answer_exposure_num=("answer_exposure_num", "sum"))
        )
        ans_agg = ans_agg.drop(columns=["pre_answer_exposure_num"]).merge(tmp, on="user_id", how="left")
    ans_agg["pre_answer_exposure_num"] = ans_agg["pre_answer_exposure_num"].fillna(0.0)
    ans_agg["pre_answer_exposure_mean"] = np.where(
        ans_agg["pre_answer_count"] > 0,
        ans_agg["pre_answer_exposure_num"] / ans_agg["pre_answer_count"],
        np.nan,
    )

    features = ask_agg.merge(ans_agg, on="user_id", how="outer")
    for col in [
        "pre_ask_questions",
        "pre_high_tag_questions",
        "pre_ask_exposure_num",
        "pre_answer_count",
        "pre_accepted_current",
        "pre_answer_exposure_num",
    ]:
        features[col] = features[col].fillna(0.0)
    features["pre_high_tag_share"] = features["pre_high_tag_share"].fillna(0.0)
    return features


def add_matching_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["log_pre_ask_questions"] = np.log1p(out["pre_ask_questions"].fillna(0.0))
    out["log_pre_answer_count"] = np.log1p(out["pre_answer_count"].fillna(0.0))
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
    return out


def standardized_distance(pool: pd.DataFrame, target: pd.Series, feature_cols: list[str]) -> pd.Series:
    numeric_pool = pool[feature_cols].astype(float).copy()
    center = numeric_pool.mean(axis=0)
    scale = numeric_pool.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    target_vals = target[feature_cols].astype(float).copy()
    target_vals = target_vals.fillna(center)
    numeric_pool = numeric_pool.fillna(center)
    distances = ((numeric_pool - target_vals) / scale) ** 2
    return distances.sum(axis=1)


def build_match_pairs(ask: pd.DataFrame, answer: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict = pd.read_csv(STRICT_HIT_USERS_PATH)
    strict["user_id"] = pd.to_numeric(strict["user_id"], errors="coerce").astype("Int64")
    strict = strict.loc[strict["user_id"].notna()].copy()
    strict["user_id"] = strict["user_id"].astype("int64")
    strict["first_ai_hit_month_id"] = strict["first_ai_hit_month_id"].astype(str)
    strict["hit_month_index"] = month_to_index(strict["first_ai_hit_month_id"])

    first_active = (
        ask.groupby("user_id", as_index=False)["month_index"]
        .min()
        .rename(columns={"month_index": "first_active_month_index"})
    )

    any_proxy = ask.groupby("user_id", as_index=False)["any_genai_proxy_user_month"].max()
    any_proxy = any_proxy.rename(columns={"any_genai_proxy_user_month": "any_proxy_ever"})
    non_hit_ids = set(
        any_proxy.loc[any_proxy["any_proxy_ever"] == 0, "user_id"].astype(int).tolist()
    ) - set(strict["user_id"].astype(int).tolist())

    pair_rows: list[dict[str, object]] = []
    matched_feature_rows: list[dict[str, object]] = []

    feature_cols = [
        "log_pre_ask_questions",
        "pre_high_tag_share",
        "pre_ask_exposure_mean",
        "log_pre_answer_count",
        "first_active_month_index",
    ]

    for hit_month_index in sorted(strict["hit_month_index"].unique().tolist()):
        treated_month = strict.loc[strict["hit_month_index"] == hit_month_index].copy()
        pre_start = int(hit_month_index) - 6
        pre_end = int(hit_month_index) - 1

        features = aggregate_window_features(ask, answer, pre_start, pre_end)
        features = first_active.merge(features, on="user_id", how="left")
        features["pre_ask_questions"] = features["pre_ask_questions"].fillna(0.0)
        features["pre_high_tag_questions"] = features["pre_high_tag_questions"].fillna(0.0)
        features["pre_high_tag_share"] = features["pre_high_tag_share"].fillna(0.0)
        features["pre_ask_exposure_mean"] = features["pre_ask_exposure_mean"].fillna(0.0)
        features["pre_answer_count"] = features["pre_answer_count"].fillna(0.0)
        features["pre_accepted_current"] = features["pre_accepted_current"].fillna(0.0)
        features["pre_answer_exposure_mean"] = features["pre_answer_exposure_mean"].fillna(0.0)
        features = add_matching_features(features)

        treated_feat = treated_month.merge(features, on="user_id", how="left")
        treated_feat = add_matching_features(treated_feat)
        treated_feat["hit_month_index"] = hit_month_index

        candidate_feat = features.loc[features["user_id"].isin(non_hit_ids)].copy()
        candidate_feat = candidate_feat.loc[candidate_feat["pre_ask_questions"] > 0].copy()
        candidate_feat["hit_month_index"] = hit_month_index
        print(
            f"matching month={index_to_month(hit_month_index)} treated={treated_month.shape[0]} candidates={candidate_feat.shape[0]}"
        )

        used_control_ids: set[int] = set()
        for treated_row in treated_feat.to_dict("records"):
            pool = candidate_feat.loc[~candidate_feat["user_id"].isin(used_control_ids)].copy()
            if pool.empty:
                continue

            exact_bucket = pool.loc[pool["first_active_bucket"] == treated_row["first_active_bucket"]].copy()
            if not exact_bucket.empty:
                pool = exact_bucket
                bucket_rule = "first_active_bucket"
            else:
                bucket_rule = "relaxed_first_active_bucket"

            exact_ask_bin = pool.loc[pool["ask_bin"] == treated_row["ask_bin"]].copy()
            if not exact_ask_bin.empty:
                pool = exact_ask_bin
                ask_rule = "ask_bin"
            else:
                ask_rule = "relaxed_ask_bin"

            pool = pool.reset_index(drop=True)
            distances = standardized_distance(pool, pd.Series(treated_row), feature_cols)
            best_idx = int(distances.idxmin())
            match = pool.iloc[best_idx]
            used_control_ids.add(int(match["user_id"]))

            pair_rows.append(
                {
                    "treated_user_id": int(treated_row["user_id"]),
                    "control_user_id": int(match["user_id"]),
                    "hit_month_id": index_to_month(hit_month_index),
                    "hit_month_index": int(hit_month_index),
                    "distance": float(distances.iloc[best_idx]),
                    "matching_rule": f"{bucket_rule}+{ask_rule}",
                    "treated_first_active_bucket": treated_row["first_active_bucket"],
                    "control_first_active_bucket": match["first_active_bucket"],
                    "treated_ask_bin": treated_row["ask_bin"],
                    "control_ask_bin": match["ask_bin"],
                }
            )
            matched_feature_rows.append(
                {
                    "treated_user_id": int(treated_row["user_id"]),
                    "control_user_id": int(match["user_id"]),
                    "hit_month_id": index_to_month(hit_month_index),
                    "pre_ask_questions_treated": float(treated_row["pre_ask_questions"]),
                    "pre_ask_questions_control": float(match["pre_ask_questions"]),
                    "pre_high_tag_share_treated": float(treated_row["pre_high_tag_share"]),
                    "pre_high_tag_share_control": float(match["pre_high_tag_share"]),
                    "pre_ask_exposure_mean_treated": float(treated_row["pre_ask_exposure_mean"]),
                    "pre_ask_exposure_mean_control": float(match["pre_ask_exposure_mean"]),
                    "pre_answer_count_treated": float(treated_row["pre_answer_count"]),
                    "pre_answer_count_control": float(match["pre_answer_count"]),
                    "first_active_month_index_treated": float(treated_row["first_active_month_index"]),
                    "first_active_month_index_control": float(match["first_active_month_index"]),
                }
            )

    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        raise ValueError("No matched pairs were created.")
    return pairs, pd.DataFrame(matched_feature_rows)


def build_monthly_maps(ask: pd.DataFrame, answer: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ask_slim = ask[
        ["user_id", "month_id", "month_index", "questions", "high_tag_questions", "mean_exposure_index"]
    ].copy()
    answer_slim = answer[
        [
            "user_id",
            "month_id",
            "month_index",
            "answer_count_total",
            "accepted_current_total",
            "answer_exposure_index_wavg",
        ]
    ].copy()
    return ask_slim, answer_slim


def build_event_windows(
    pairs: pd.DataFrame, ask_slim: pd.DataFrame, answer_slim: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ask_lookup = ask_slim.set_index(["user_id", "month_index"])
    answer_lookup = answer_slim.set_index(["user_id", "month_index"])

    rows: list[dict[str, object]] = []
    for pair_id, pair in pairs.reset_index(drop=True).iterrows():
        hit_idx = int(pair["hit_month_index"])
        for group, user_id in [
            ("treated", int(pair["treated_user_id"])),
            ("matched_control", int(pair["control_user_id"])),
        ]:
            for rel in range(-6, 7):
                month_index = hit_idx + rel
                ask_key = (user_id, month_index)
                ans_key = (user_id, month_index)

                ask_questions = 0.0
                high_tag_questions = 0.0
                ask_exposure_index_mean = np.nan
                if ask_key in ask_lookup.index:
                    ask_row = ask_lookup.loc[ask_key]
                    if isinstance(ask_row, pd.DataFrame):
                        ask_row = ask_row.iloc[0]
                    ask_questions = float(ask_row["questions"])
                    high_tag_questions = float(ask_row["high_tag_questions"])
                    ask_exposure_index_mean = float(ask_row["mean_exposure_index"])

                answer_count_total = 0.0
                accepted_current_total = 0.0
                answer_exposure_index_wavg = np.nan
                if ans_key in answer_lookup.index:
                    ans_row = answer_lookup.loc[ans_key]
                    if isinstance(ans_row, pd.DataFrame):
                        ans_row = ans_row.iloc[0]
                    answer_count_total = float(ans_row["answer_count_total"])
                    accepted_current_total = float(ans_row["accepted_current_total"])
                    answer_exposure_index_wavg = float(ans_row["answer_exposure_index_wavg"])

                rows.append(
                    {
                        "pair_id": pair_id + 1,
                        "group": group,
                        "user_id": user_id,
                        "hit_month_id": pair["hit_month_id"],
                        "rel_month": rel,
                        "month_index": month_index,
                        "month_id": index_to_month(month_index),
                        "ask_questions": ask_questions,
                        "ask_high_tag_share": (high_tag_questions / ask_questions) if ask_questions > 0 else 0.0,
                        "ask_exposure_index_mean": ask_exposure_index_mean,
                        "answer_count_total": answer_count_total,
                        "accepted_current_total": accepted_current_total,
                        "answer_exposure_index_wavg": answer_exposure_index_wavg,
                    }
                )

    event_panel = pd.DataFrame(rows)
    event_time = (
        event_panel.groupby(["group", "rel_month"], as_index=False)
        .agg(
            users=("user_id", "nunique"),
            ask_questions_mean=("ask_questions", "mean"),
            ask_high_tag_share_mean=("ask_high_tag_share", "mean"),
            ask_exposure_index_mean=("ask_exposure_index_mean", "mean"),
            answer_count_total_mean=("answer_count_total", "mean"),
            accepted_current_total_mean=("accepted_current_total", "mean"),
            answer_exposure_index_wavg_mean=("answer_exposure_index_wavg", "mean"),
        )
        .sort_values(["group", "rel_month"])
        .reset_index(drop=True)
    )

    event_panel["period"] = np.where(event_panel["rel_month"] >= 0, "post_pseudo_hit", "pre_pseudo_hit")
    # Explicit pre/post table for readability.
    prepost_rows: list[dict[str, object]] = []
    for group, frame in event_panel.groupby("group"):
        for period, sub in frame.groupby("period"):
            prepost_rows.append(
                {
                    "group": group,
                    "period": period,
                    "ask_questions_mean": float(sub["ask_questions"].mean()),
                    "ask_high_tag_share_mean": float(sub["ask_high_tag_share"].mean()),
                    "ask_exposure_index_mean": float(sub["ask_exposure_index_mean"].mean()),
                    "answer_count_total_mean": float(sub["answer_count_total"].mean()),
                    "accepted_current_total_mean": float(sub["accepted_current_total"].mean()),
                    "answer_exposure_index_wavg_mean": float(sub["answer_exposure_index_wavg"].mean()),
                }
            )
    prepost_table = pd.DataFrame(prepost_rows).sort_values(["group", "period"]).reset_index(drop=True)

    return event_time, prepost_table


def build_diff_in_diff(prepost: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        "ask_questions_mean",
        "ask_high_tag_share_mean",
        "ask_exposure_index_mean",
        "answer_count_total_mean",
        "accepted_current_total_mean",
        "answer_exposure_index_wavg_mean",
    ]
    rows: list[dict[str, object]] = []
    treated = prepost.loc[prepost["group"] == "treated"].set_index("period")
    control = prepost.loc[prepost["group"] == "matched_control"].set_index("period")
    for col in outcomes:
        treated_pre = float(treated.loc["pre_pseudo_hit", col])
        treated_post = float(treated.loc["post_pseudo_hit", col])
        control_pre = float(control.loc["pre_pseudo_hit", col])
        control_post = float(control.loc["post_pseudo_hit", col])
        rows.append(
            {
                "outcome": col,
                "treated_delta": treated_post - treated_pre,
                "control_delta": control_post - control_pre,
                "diff_in_diff": (treated_post - treated_pre) - (control_post - control_pre),
                "treated_pre": treated_pre,
                "treated_post": treated_post,
                "control_pre": control_pre,
                "control_post": control_post,
            }
        )
    return pd.DataFrame(rows)


def build_balance_table(match_features: pd.DataFrame) -> pd.DataFrame:
    if match_features.empty:
        return pd.DataFrame()

    balance_rows: list[dict[str, object]] = []
    feature_pairs = [
        ("pre_ask_questions_treated", "pre_ask_questions_control", "pre_ask_questions"),
        ("pre_high_tag_share_treated", "pre_high_tag_share_control", "pre_high_tag_share"),
        ("pre_ask_exposure_mean_treated", "pre_ask_exposure_mean_control", "pre_ask_exposure_mean"),
        ("pre_answer_count_treated", "pre_answer_count_control", "pre_answer_count"),
        ("first_active_month_index_treated", "first_active_month_index_control", "first_active_month_index"),
    ]
    for treated_col, control_col, label in feature_pairs:
        treated = match_features[treated_col].astype(float)
        control = match_features[control_col].astype(float)
        pooled_sd = np.sqrt((treated.var(ddof=0) + control.var(ddof=0)) / 2)
        smd = 0.0 if pooled_sd == 0 or np.isnan(pooled_sd) else float((treated.mean() - control.mean()) / pooled_sd)
        balance_rows.append(
            {
                "feature": label,
                "treated_mean": float(treated.mean()),
                "control_mean": float(control.mean()),
                "standardized_mean_diff": smd,
            }
        )
    return pd.DataFrame(balance_rows)


def write_readout(
    pairs: pd.DataFrame,
    balance: pd.DataFrame,
    event_time: pd.DataFrame,
    prepost: pd.DataFrame,
    did: pd.DataFrame,
) -> None:
    treated_n = int(pairs["treated_user_id"].nunique())
    control_n = int(pairs["control_user_id"].nunique())
    matched_months = sorted(pairs["hit_month_id"].astype(str).unique().tolist())
    balance_max = float(balance["standardized_mean_diff"].abs().max()) if not balance.empty else np.nan

    def prepost_value(group: str, period: str, column: str) -> float:
        row = prepost.loc[(prepost["group"] == group) & (prepost["period"] == period), column]
        return float(row.iloc[0]) if not row.empty else float("nan")

    lines = [
        "# P1 GenAI User-Level Matched AI-Hit vs Non-Hit Comparison",
        "",
        "Date: `2026-04-05`  ",
        "Timezone: `America/New_York`",
        "",
        "## Purpose",
        "",
        "This layer builds a bounded matched descriptive comparison between strict `AI-hit` askers and users who never show any GenAI proxy hit in the focal question panel.",
        "",
        "The goal is not adoption identification. The goal is to make the user-linked layer more reviewer-legible by adding a conservative non-hit comparison benchmark.",
        "",
        "## Design",
        "",
        "- Treated users: strict `AI-hit` users from explicit title/tag disclosures",
        "- Controls: users with zero GenAI proxy hits in the focal ask panel",
        "- Match timing: each control inherits the treated user's `first_ai_hit_month_id` as a pseudo-hit month",
        "- Matching surface: pre6 ask volume, high-tag share, mean exposure, answer volume, and first active month",
        "- Interpretation: matched descriptive comparison only",
        "",
        "## Match Summary",
        "",
        f"- matched treated users: `{treated_n}`",
        f"- matched control users: `{control_n}`",
        f"- matched hit-months covered: `{len(matched_months)}`",
        f"- max absolute standardized mean difference across match features: `{balance_max:.3f}`",
        "",
        "## Headline Read",
        "",
        "After matching on pre-hit activity and timing, the strict `AI-hit` users can now be compared to a conservative non-hit benchmark without calling the result causal.",
        "",
        f"- treated `ask_questions` pre/post: `{prepost_value('treated', 'pre_pseudo_hit', 'ask_questions_mean'):.3f} -> {prepost_value('treated', 'post_pseudo_hit', 'ask_questions_mean'):.3f}`",
        f"- control `ask_questions` pre/post: `{prepost_value('matched_control', 'pre_pseudo_hit', 'ask_questions_mean'):.3f} -> {prepost_value('matched_control', 'post_pseudo_hit', 'ask_questions_mean'):.3f}`",
        f"- treated `answer_count_total` pre/post: `{prepost_value('treated', 'pre_pseudo_hit', 'answer_count_total_mean'):.3f} -> {prepost_value('treated', 'post_pseudo_hit', 'answer_count_total_mean'):.3f}`",
        f"- control `answer_count_total` pre/post: `{prepost_value('matched_control', 'pre_pseudo_hit', 'answer_count_total_mean'):.3f} -> {prepost_value('matched_control', 'post_pseudo_hit', 'answer_count_total_mean'):.3f}`",
        f"- treated `ask_exposure_index_mean` pre/post: `{prepost_value('treated', 'pre_pseudo_hit', 'ask_exposure_index_mean'):.3f} -> {prepost_value('treated', 'post_pseudo_hit', 'ask_exposure_index_mean'):.3f}`",
        f"- control `ask_exposure_index_mean` pre/post: `{prepost_value('matched_control', 'pre_pseudo_hit', 'ask_exposure_index_mean'):.3f} -> {prepost_value('matched_control', 'post_pseudo_hit', 'ask_exposure_index_mean'):.3f}`",
        "",
        "## Diff-in-Diff (Descriptive, Not Causal)",
        "",
        "These descriptive deltas show how treated and matched non-hit users move around their pseudo-hit month. They are a benchmark, not an adoption effect.",
        "",
    ]
    for _, row in did.iterrows():
        lines.append(
            f"- `{row['outcome']}` 螖 treated `{row['treated_delta']:.3f}`, 螖 control `{row['control_delta']:.3f}`, diff-in-diff `{row['diff_in_diff']:.3f}`"
        )
    lines += [
        "",
        "## Safe Interpretation",
        "",
        "- This layer improves the user-linked credibility of the paper because it no longer shows only AI-hit users in isolation.",
        "- It still does not observe telemetry of actual AI use.",
        "- It still should be framed as a bounded comparison benchmark, not as an adoption effect.",
        "",
        "## Related Files",
        "",
        "- [p1_genai_user_matched_pairs.csv](D:/AI%20alignment/projects/stackoverflow_chatgpt_governance/processed/p1_genai_user_matched_pairs.csv)",
        "- [p1_genai_user_matched_balance.csv](D:/AI%20alignment/projects/stackoverflow_chatgpt_governance/processed/p1_genai_user_matched_balance.csv)",
        "- [p1_genai_user_matched_event_time.csv](D:/AI%20alignment/projects/stackoverflow_chatgpt_governance/processed/p1_genai_user_matched_event_time.csv)",
        "- [p1_genai_user_matched_prepost.csv](D:/AI%20alignment/projects/stackoverflow_chatgpt_governance/processed/p1_genai_user_matched_prepost.csv)",
        "",
    ]
    READOUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ask = prepare_ask_panel()
    answer = prepare_answer_panel()
    pairs, match_features = build_match_pairs(ask, answer)
    ask_slim, answer_slim = build_monthly_maps(ask, answer)
    balance = build_balance_table(match_features)
    event_time, prepost = build_event_windows(pairs, ask_slim, answer_slim)
    did = build_diff_in_diff(prepost)

    MATCHED_PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(MATCHED_PAIRS_PATH, index=False)
    balance.to_csv(MATCHED_BALANCE_PATH, index=False)
    event_time.to_csv(MATCHED_EVENT_TIME_PATH, index=False)
    prepost.to_csv(MATCHED_PREPOST_PATH, index=False)
    did.to_csv(MATCHED_DID_PATH, index=False)
    write_readout(pairs, balance, event_time, prepost, did)

    print(f"Wrote {MATCHED_PAIRS_PATH}")
    print(f"Wrote {MATCHED_BALANCE_PATH}")
    print(f"Wrote {MATCHED_EVENT_TIME_PATH}")
    print(f"Wrote {MATCHED_PREPOST_PATH}")
    print(f"Wrote {MATCHED_DID_PATH}")
    print(f"Wrote {READOUT_PATH}")


if __name__ == "__main__":
    main()
