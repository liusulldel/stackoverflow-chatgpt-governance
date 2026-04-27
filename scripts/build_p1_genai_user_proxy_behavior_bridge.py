from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

PANEL_CSV = PROCESSED / "p1_genai_user_level_proxy_ai_hit_user_month_panel.csv"

OUT_EVENT_TIME = PROCESSED / "p1_genai_user_proxy_behavior_event_time.csv"
OUT_PREPOST = PROCESSED / "p1_genai_user_proxy_behavior_prepost.csv"
OUT_SUMMARY = PROCESSED / "p1_genai_user_proxy_behavior_summary.json"
OUT_MEMO = PAPER / "p1_genai_user_proxy_behavior_bridge_2026-04-05.md"


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_CSV)
    return df


def build_event_time(df: pd.DataFrame) -> pd.DataFrame:
    win = df.loc[df["months_since_first_ai_hit"].between(-6, 6, inclusive="both")].copy()
    out = (
        win.groupby("months_since_first_ai_hit", as_index=False)
        .agg(
            user_months=("user_id", "size"),
            answerer_share=("is_answerer_month", "mean"),
            answer_count_mean=("answer_count_total", "mean"),
            accepted_current_mean=("accepted_current_total", "mean"),
            answer_exposure_mean=("answer_exposure_index_wavg", "mean"),
            answer_high_share_mean=("answer_high_share", "mean"),
            answer_expert_share_mean=("answer_expert_share", "mean"),
        )
        .sort_values("months_since_first_ai_hit")
        .reset_index(drop=True)
    )
    out.to_csv(OUT_EVENT_TIME, index=False)
    return out


def build_prepost(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.loc[df["is_answerer_month"] == 1].copy()
    sub["period"] = sub["months_since_first_ai_hit"].ge(0).map({True: "post_first_hit", False: "pre_first_hit"})
    metrics = [
        "answer_count_total",
        "accepted_current_total",
        "answer_exposure_index_wavg",
        "answer_high_share",
        "answer_expert_share",
    ]
    rows = []
    for metric in metrics:
        pre = sub.loc[sub["period"] == "pre_first_hit", metric].mean()
        post = sub.loc[sub["period"] == "post_first_hit", metric].mean()
        rows.append(
            {
                "metric": metric,
                "pre_first_hit_mean": pre,
                "post_first_hit_mean": post,
                "post_minus_pre": post - pre,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PREPOST, index=False)
    return out


def write_summary(df: pd.DataFrame, event_time: pd.DataFrame, prepost: pd.DataFrame) -> dict[str, object]:
    summary = {
        "n_ai_hit_users": int(df["user_id"].nunique()),
        "n_ai_hit_user_months": int(len(df)),
        "n_ever_answerer_users": int(df.loc[df["is_answerer_month"] == 1, "user_id"].nunique()),
        "n_answerer_months": int((df["is_answerer_month"] == 1).sum()),
        "event_time_window": [-6, 6],
        "headline_prepost": prepost.to_dict(orient="records"),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_memo(summary: dict[str, object], event_time: pd.DataFrame, prepost: pd.DataFrame) -> None:
    def metric_row(metric: str) -> dict:
        return prepost.loc[prepost["metric"].eq(metric)].iloc[0].to_dict()

    count_row = metric_row("answer_count_total")
    exposure_row = metric_row("answer_exposure_index_wavg")
    high_row = metric_row("answer_high_share")
    accepted_row = metric_row("accepted_current_total")

    lines = [
        "# P1 User-Level Proxy to Behavior Bridge",
        "",
        "Date: `2026-04-05`",
        "",
        "## What This Build Does",
        "",
        "This build uses the strict `AI-hit` user-month panel and summarizes what happens to observable answering behavior around each user's first strict AI-hit month.",
        "It is a bounded bridge from user-linked proxy evidence to behavior, not a causal adoption design.",
        "",
        "## Coverage",
        "",
        f"- AI-hit users: `{summary['n_ai_hit_users']:,}`",
        f"- AI-hit user-months: `{summary['n_ai_hit_user_months']:,}`",
        f"- users who ever appear as answerers in focal tags: `{summary['n_ever_answerer_users']:,}`",
        f"- answerer-months in the AI-hit panel: `{summary['n_answerer_months']:,}`",
        "",
        "## Pre/Post First-Hit Read (Answerer Months Only)",
        "",
        f"- `answer_count_total`: `{count_row['pre_first_hit_mean']:.3f}` -> `{count_row['post_first_hit_mean']:.3f}` (`{count_row['post_minus_pre']:.3f}`)",
        f"- `accepted_current_total`: `{accepted_row['pre_first_hit_mean']:.3f}` -> `{accepted_row['post_first_hit_mean']:.3f}` (`{accepted_row['post_minus_pre']:.3f}`)",
        f"- `answer_exposure_index_wavg`: `{exposure_row['pre_first_hit_mean']:.3f}` -> `{exposure_row['post_first_hit_mean']:.3f}` (`{exposure_row['post_minus_pre']:.3f}`)",
        f"- `answer_high_share`: `{high_row['pre_first_hit_mean']:.3f}` -> `{high_row['post_first_hit_mean']:.3f}` (`{high_row['post_minus_pre']:.3f}`)",
        "",
        "## Conservative Read",
        "",
        "The strict AI-hit users who also appear as answerers are a small minority, so this layer should be treated as descriptive bridge evidence.",
        "The most suggestive pattern is not a large increase in answer volume after first strict AI-hit. Instead, the panel shows slightly higher answer-exposure weighting alongside lower answer counts and lower accepted-current counts. That makes this layer useful for bounded behavioral context, but not for strong causal storytelling.",
        "",
        "## Files",
        "",
        f"- event-time csv: `{OUT_EVENT_TIME}`",
        f"- pre/post csv: `{OUT_PREPOST}`",
        f"- summary json: `{OUT_SUMMARY}`",
    ]
    OUT_MEMO.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    df = load_panel()
    event_time = build_event_time(df)
    prepost = build_prepost(df)
    summary = write_summary(df, event_time, prepost)
    write_memo(summary, event_time, prepost)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
