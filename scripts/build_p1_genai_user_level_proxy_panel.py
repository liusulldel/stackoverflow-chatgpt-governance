from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\AI alignment\projects\stackoverflow_chatgpt_governance")
PROCESSED = ROOT / "processed"

QUESTION_PARQUET = PROCESSED / "stackexchange_20251231_question_level_enriched.parquet"
ANSWER_USER_TAG_MONTH_PARQUET = PROCESSED / "who_still_answers_user_tag_month_panel.parquet"

# Outputs (bounded to P1 user-level proxy lane)
OUT_MONTHLY_COUNTS_CSV = PROCESSED / "p1_genai_user_level_proxy_monthly_counts.csv"
OUT_AI_HIT_USERS_CSV = PROCESSED / "p1_genai_user_level_proxy_ai_hit_users.csv"
OUT_AI_HIT_USER_MONTH_PANEL_CSV = (
    PROCESSED / "p1_genai_user_level_proxy_ai_hit_user_month_panel.csv"
)
OUT_SUMMARY_JSON = PROCESSED / "p1_genai_user_level_proxy_summary.json"
OUT_VARIABLES_JSON = PROCESSED / "p1_genai_user_level_proxy_variables.json"


# Title patterns: designed to be conservative "explicit mention" signals.
# Notes:
# - We intentionally exclude ambiguous tokens like plain "cursor" (database cursor, UI cursor).
# - "gemini" can be ambiguous; we keep it but report it separately for transparency.
TITLE_PATTERNS_STRICT: dict[str, str] = {
    "chatgpt": r"chatgpt",
    "openai": r"\bopenai\b",
    "copilot": r"copilot",
    "claude": r"\bclaude\b|claude-code|claude code",
    "anthropic": r"\banthropic\b",
    "deepseek": r"deepseek",
    "llm": r"\bllm(?:s)?\b",
    # Keep GPT family broad but anchored to the token boundary.
    "gpt_family": r"\bgpt[- ]?(?:2|3|3\.5|4|4o|4\.1|4\.5|5|o1|o3|o4)\b",
    "gemini": r"\bgemini\b",
}


# Tag whitelist: exact tag matches in `question_tags` (semicolon-separated).
# We avoid broad substring rules because they create many false positives (e.g., "diffie-hellman").
TAG_WHITELIST_STRICT: set[str] = {
    # ChatGPT/GPT ecosystem
    "chatgpt",
    "chatgpt-api",
    "chatgpt-plugin",
    "chatgpt-function-call",
    "chat-gpt-4",
    "gpt-2",
    "gpt-3",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt4all",
    "pygpt4all",
    "autogpt",
    "privategpt",
    "gpt-index",
    "h2ogpt",
    # OpenAI interfaces (GenAI-relevant)
    "openai-api",
    "azure-openai",
    "openai-whisper",
    "openai-assistants-api",
    "openaiembeddings",
    "openai-clip",
    # Copilot tooling (exclude aws-copilot on purpose)
    "github-copilot",
    "microsoft-copilot",
    "vscode-copilot",
    # Claude / Anthropic
    "claude",
    "claude-code",
    "anthropic",
    # Gemini (Google)
    "google-gemini",
    "google-gemini-file-api",
    # Other named models/tools
    "deepseek",
    # Cursor IDE (exclude plain "cursor" tag)
    "cursor-ide",
}


TAG_GROUPS: dict[str, set[str]] = {
    "tag_chatgpt_gpt": {
        "chatgpt",
        "chatgpt-api",
        "chatgpt-plugin",
        "chatgpt-function-call",
        "chat-gpt-4",
        "gpt-2",
        "gpt-3",
        "gpt-4",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5",
        "gpt4all",
        "pygpt4all",
        "autogpt",
        "privategpt",
        "gpt-index",
        "h2ogpt",
    },
    "tag_openai": {
        "openai-api",
        "azure-openai",
        "openai-whisper",
        "openai-assistants-api",
        "openaiembeddings",
        "openai-clip",
    },
    "tag_copilot": {"github-copilot", "microsoft-copilot", "vscode-copilot"},
    "tag_claude_anthropic": {"claude", "claude-code", "anthropic"},
    "tag_gemini": {"google-gemini", "google-gemini-file-api"},
    "tag_deepseek": {"deepseek"},
    "tag_cursor_ide": {"cursor-ide"},
}


def exact_tag_regex(tags: set[str]) -> str:
    escaped = [re.escape(t) for t in sorted(tags, key=len, reverse=True)]
    # Semicolon-separated tags, no surrounding whitespace in the current dataset.
    return r"(?:^|;)(?:" + "|".join(escaped) + r")(?:;|$)"


@dataclass(frozen=True)
class BuildOutputs:
    monthly_counts: Path
    ai_hit_users: Path
    ai_hit_user_month_panel: Path
    summary_json: Path
    variables_json: Path


def build_user_level_proxy_panel() -> BuildOutputs:
    q_cols = [
        "question_id",
        "owner_user_id",
        "month_id",
        "title",
        "question_tags",
        "primary_tag",
        "high_tag",
        "exposure_index",
        "post_chatgpt",
    ]
    q = pd.read_parquet(QUESTION_PARQUET, columns=q_cols).copy()
    q["owner_user_id"] = pd.to_numeric(q["owner_user_id"], errors="coerce").astype("Int64")
    q = q.loc[q["owner_user_id"].notna()].copy()
    q["user_id"] = q["owner_user_id"].astype("int64")

    q["high_tag"] = pd.to_numeric(q["high_tag"], errors="coerce").fillna(0).astype("int8")
    q["post_chatgpt"] = pd.to_numeric(q["post_chatgpt"], errors="coerce").fillna(0).astype("int8")

    title = q["title"].fillna("")
    for label, pattern in TITLE_PATTERNS_STRICT.items():
        q[f"title_hit_{label}"] = title.str.contains(pattern, case=False, regex=True).astype(
            "int8"
        )
    title_hit_cols = [f"title_hit_{k}" for k in TITLE_PATTERNS_STRICT]
    q["ai_title_hit_strict"] = q[title_hit_cols].any(axis=1).astype("int8")

    tag_pat = exact_tag_regex(TAG_WHITELIST_STRICT)
    q["ai_tag_hit_strict"] = (
        q["question_tags"].fillna("").str.contains(tag_pat, case=False, regex=True).astype("int8")
    )
    for group_label, tags in TAG_GROUPS.items():
        group_pat = exact_tag_regex(tags)
        q[group_label] = (
            q["question_tags"].fillna("").str.contains(group_pat, case=False, regex=True).astype("int8")
        )

    q["ai_any_hit_strict"] = ((q["ai_title_hit_strict"] == 1) | (q["ai_tag_hit_strict"] == 1)).astype(
        "int8"
    )

    # Month list for a balanced panel (within this focal-tag question universe).
    months = sorted(q["month_id"].dropna().astype(str).unique().tolist())

    # Monthly aggregate denominators/numerators (useful for paper-level calibration).
    monthly = (
        q.groupby("month_id", as_index=False)
        .agg(
            questions=("question_id", "size"),
            ai_title_hits_strict=("ai_title_hit_strict", "sum"),
            ai_tag_hits_strict=("ai_tag_hit_strict", "sum"),
            ai_any_hits_strict=("ai_any_hit_strict", "sum"),
        )
        .sort_values("month_id")
        .reset_index(drop=True)
    )
    monthly["ai_any_share_strict"] = monthly["ai_any_hits_strict"] / monthly["questions"]
    monthly.to_csv(OUT_MONTHLY_COUNTS_CSV, index=False)

    # Identify "AI-hit users" based on any strict hit anywhere in their asked questions.
    ai_hit_users = (
        q.loc[q["ai_any_hit_strict"] == 1, ["user_id", "month_id"]]
        .drop_duplicates()
        .sort_values(["user_id", "month_id"])
        .reset_index(drop=True)
    )
    user_first = ai_hit_users.drop_duplicates("user_id", keep="first").rename(
        columns={"month_id": "first_ai_hit_month_id"}
    )
    hit_user_ids = user_first["user_id"].to_numpy(dtype="int64")

    # Asker-month panel (restricted to AI-hit users; balanced over months).
    q_hit_users = q.loc[q["user_id"].isin(hit_user_ids)].copy()
    ask = (
        q_hit_users.groupby(["user_id", "month_id"], as_index=False)
        .agg(
            ask_questions=("question_id", "size"),
            ask_primary_tags_nunique=("primary_tag", "nunique"),
            ask_high_tag_questions=("high_tag", "sum"),
            ask_ai_title_hits_strict=("ai_title_hit_strict", "sum"),
            ask_ai_tag_hits_strict=("ai_tag_hit_strict", "sum"),
            ask_ai_any_hits_strict=("ai_any_hit_strict", "sum"),
            ask_exposure_index_mean=("exposure_index", "mean"),
            ask_post_chatgpt=("post_chatgpt", "max"),
            **{f"ask_title_{k}_hits": (f"title_hit_{k}", "sum") for k in TITLE_PATTERNS_STRICT},
            **{f"ask_{k}_tag_hits": (k, "sum") for k in TAG_GROUPS},
        )
        .sort_values(["user_id", "month_id"])
        .reset_index(drop=True)
    )
    ask["ask_ai_any_share_strict"] = ask["ask_ai_any_hits_strict"] / ask["ask_questions"]
    ask["ask_high_tag_share"] = ask["ask_high_tag_questions"] / ask["ask_questions"]

    # Answerer-month behavior context, restricted to the same AI-hit users (if present as answerers).
    a_cols = [
        "answerer_user_id",
        "month_id",
        "primary_tag",
        "answer_count",
        "accepted_current_count",
        "any_answer",
        "high_tag",
        "exposure_index",
        "is_expert",
        "is_incumbent_nonexpert",
        "preshock_answers",
    ]
    a = pd.read_parquet(ANSWER_USER_TAG_MONTH_PARQUET, columns=a_cols).copy()
    a["answerer_user_id"] = pd.to_numeric(a["answerer_user_id"], errors="coerce").astype("Int64")
    a = a.loc[a["answerer_user_id"].notna()].copy()
    a["user_id"] = a["answerer_user_id"].astype("int64")
    a = a.loc[a["user_id"].isin(hit_user_ids)].copy()

    a["answer_count"] = pd.to_numeric(a["answer_count"], errors="coerce").fillna(0.0)
    a["accepted_current_count"] = pd.to_numeric(a["accepted_current_count"], errors="coerce").fillna(
        0.0
    )
    a["any_answer"] = pd.to_numeric(a["any_answer"], errors="coerce").fillna(0).astype("int8")
    a["high_tag"] = pd.to_numeric(a["high_tag"], errors="coerce").fillna(0).astype("int8")
    a["is_expert"] = pd.to_numeric(a["is_expert"], errors="coerce").fillna(0).astype("int8")
    a["is_incumbent_nonexpert"] = pd.to_numeric(a["is_incumbent_nonexpert"], errors="coerce").fillna(
        0
    ).astype("int8")
    a["preshock_answers"] = pd.to_numeric(a["preshock_answers"], errors="coerce").fillna(0.0)

    # Weighted sums for easy aggregation.
    a["answer_exposure_wsum"] = a["answer_count"] * pd.to_numeric(a["exposure_index"], errors="coerce").fillna(
        0.0
    )
    a["answer_high_wsum"] = a["answer_count"] * a["high_tag"]
    a["answer_expert_wsum"] = a["answer_count"] * a["is_expert"]

    ans = (
        a.groupby(["user_id", "month_id"], as_index=False)
        .agg(
            answer_count_total=("answer_count", "sum"),
            accepted_current_total=("accepted_current_count", "sum"),
            answer_tags_active=("any_answer", "sum"),
            is_expert_any=("is_expert", "max"),
            is_incumbent_nonexpert_any=("is_incumbent_nonexpert", "max"),
            preshock_answers_focal_sum=("preshock_answers", "sum"),
            answer_exposure_wsum=("answer_exposure_wsum", "sum"),
            answer_high_wsum=("answer_high_wsum", "sum"),
            answer_expert_wsum=("answer_expert_wsum", "sum"),
        )
        .sort_values(["user_id", "month_id"])
        .reset_index(drop=True)
    )
    denom = ans["answer_count_total"].replace(0, np.nan)
    ans["answer_exposure_index_wavg"] = ans["answer_exposure_wsum"] / denom
    ans["answer_high_share"] = ans["answer_high_wsum"] / denom
    ans["answer_expert_share"] = ans["answer_expert_wsum"] / denom
    ans = ans.drop(columns=["answer_exposure_wsum", "answer_high_wsum", "answer_expert_wsum"])

    # Build balanced user-month panel for AI-hit users.
    full_index = pd.MultiIndex.from_product([hit_user_ids, months], names=["user_id", "month_id"])
    ask_bal = ask.set_index(["user_id", "month_id"]).reindex(full_index).reset_index()
    ans_bal = ans.set_index(["user_id", "month_id"]).reindex(full_index).reset_index()

    panel = ask_bal.merge(ans_bal, on=["user_id", "month_id"], how="left", validate="one_to_one")
    panel = panel.merge(user_first[["user_id", "first_ai_hit_month_id"]], on="user_id", how="left")

    # Fill numeric missing with 0 for *counts* (ask/answer). Keep means and shares as NaN if denominator is 0.
    # Ask-side counts: anything that is an "ask_" count, excluding mean/share fields.
    ask_count_cols = [
        c
        for c in panel.columns
        if c.startswith("ask_")
        and c
        not in {
            "ask_exposure_index_mean",
            "ask_ai_any_share_strict",
            "ask_high_tag_share",
        }
    ]
    # Answer-side counts/flags: totals, active counts, sums, and binary flags.
    answer_fill_cols = [
        c
        for c in panel.columns
        if c.endswith("_total")
        or c.endswith("_active")
        or c.endswith("_sum")
        or c in {"is_expert_any", "is_incumbent_nonexpert_any"}
    ]
    fill_zero_cols = sorted(set(ask_count_cols + answer_fill_cols))
    for c in fill_zero_cols:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0.0)
    # Make key binary fields integers.
    panel["ask_post_chatgpt"] = panel["ask_post_chatgpt"].fillna(0).astype("int8")
    panel["is_expert_any"] = panel["is_expert_any"].fillna(0).astype("int8")
    panel["is_incumbent_nonexpert_any"] = panel["is_incumbent_nonexpert_any"].fillna(0).astype(
        "int8"
    )

    # Recompute shares after reindex fill.
    aq = panel["ask_questions"].replace(0, np.nan)
    panel["ask_ai_any_share_strict"] = panel["ask_ai_any_hits_strict"] / aq
    panel["ask_high_tag_share"] = panel["ask_high_tag_questions"] / aq

    # User-month role flags (within this focal-tag universe).
    panel["is_asker_month"] = (panel["ask_questions"] > 0).astype("int8")
    panel["is_answerer_month"] = (panel["answer_count_total"] > 0).astype("int8")
    panel["ai_proxy_any_strict_month"] = (panel["ask_ai_any_hits_strict"] > 0).astype("int8")

    # Months-since-first-hit index (for event-time style plots later).
    month_dt = pd.to_datetime(panel["month_id"] + "-01", errors="coerce")
    panel["month_index"] = (month_dt.dt.year * 12 + month_dt.dt.month).astype("Int64")
    first_dt = pd.to_datetime(panel["first_ai_hit_month_id"] + "-01", errors="coerce")
    first_index = (first_dt.dt.year * 12 + first_dt.dt.month).astype("Int64")
    panel["months_since_first_ai_hit"] = (panel["month_index"] - first_index).astype("Int64")

    panel.to_csv(OUT_AI_HIT_USER_MONTH_PANEL_CSV, index=False)

    # AI-hit user summary table (one row per user).
    user_rollup = (
        ask.groupby("user_id", as_index=False)
        .agg(
            ask_questions_total=("ask_questions", "sum"),
            ask_ai_any_hits_total=("ask_ai_any_hits_strict", "sum"),
            ask_ai_title_hits_total=("ask_ai_title_hits_strict", "sum"),
            ask_ai_tag_hits_total=("ask_ai_tag_hits_strict", "sum"),
            ask_active_months=("month_id", "nunique"),
        )
        .merge(user_first, on="user_id", how="left", validate="one_to_one")
        .sort_values(["ask_ai_any_hits_total", "ask_questions_total"], ascending=False)
        .reset_index(drop=True)
    )
    user_rollup["ask_ai_any_share_total"] = user_rollup["ask_ai_any_hits_total"] / user_rollup[
        "ask_questions_total"
    ].replace(0, np.nan)
    # Add whether the user ever appears as an answerer in the focal behavior panel (restricted).
    ever_answerer = (
        ans.groupby("user_id", as_index=False)
        .agg(answer_months=("month_id", "nunique"), answers_total=("answer_count_total", "sum"))
        .assign(ever_answerer=1)
    )
    user_rollup = user_rollup.merge(
        ever_answerer[["user_id", "ever_answerer", "answer_months", "answers_total"]],
        on="user_id",
        how="left",
        validate="one_to_one",
    )
    user_rollup["ever_answerer"] = user_rollup["ever_answerer"].fillna(0).astype("int8")
    user_rollup["answer_months"] = user_rollup["answer_months"].fillna(0).astype("int64")
    user_rollup["answers_total"] = user_rollup["answers_total"].fillna(0.0)
    user_rollup.to_csv(OUT_AI_HIT_USERS_CSV, index=False)

    # Diagnostics / metadata for claim-control and reproducibility.
    summary = {
        "inputs": {
            "question_parquet": str(QUESTION_PARQUET),
            "answer_user_tag_month_parquet": str(ANSWER_USER_TAG_MONTH_PARQUET),
        },
        "outputs": {
            "monthly_counts_csv": str(OUT_MONTHLY_COUNTS_CSV),
            "ai_hit_users_csv": str(OUT_AI_HIT_USERS_CSV),
            "ai_hit_user_month_panel_csv": str(OUT_AI_HIT_USER_MONTH_PANEL_CSV),
        },
        "scope": {
            "focal_primary_tags_n": int(q["primary_tag"].nunique()),
            "focal_primary_tags": sorted(q["primary_tag"].dropna().unique().tolist()),
            "months_n": int(len(months)),
            "months_min": str(min(months)) if months else None,
            "months_max": str(max(months)) if months else None,
        },
        "question_level_counts": {
            "questions_total": int(len(q)),
            "ai_title_hits_strict_total": int(q["ai_title_hit_strict"].sum()),
            "ai_tag_hits_strict_total": int(q["ai_tag_hit_strict"].sum()),
            "ai_any_hits_strict_total": int(q["ai_any_hit_strict"].sum()),
            "ai_any_share_strict_total": float(q["ai_any_hit_strict"].mean()),
        },
        "user_level_counts": {
            "ai_hit_users_n": int(len(hit_user_ids)),
            "ai_hit_user_month_rows": int(len(panel)),
            "ai_hit_user_ever_answerer_n": int(user_rollup["ever_answerer"].sum()),
        },
        "patterns": {
            "title_patterns_strict": TITLE_PATTERNS_STRICT,
            "tag_whitelist_strict": sorted(TAG_WHITELIST_STRICT),
            "tag_groups": {k: sorted(v) for k, v in TAG_GROUPS.items()},
        },
        "notes": [
            "Strict AI-hit definition is based on explicit AI tool/model mentions in question titles and/or exact GenAI-relevant tags within the focal 16-tag universe.",
            "This is not telemetry of actual AI usage; it is user self-disclosure / self-labeling and should be framed as an observed proxy.",
            "Answer-behavior columns are contextual proxies (activity reallocation) and should not be labeled as direct adoption.",
        ],
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    variables = {
        "panel_unit": "user_id x month_id (balanced for AI-hit users only)",
        "adoption_proxy_vs_telemetry": {
            "telemetry_present": False,
            "directish_disclosure": [
                "ask_ai_title_hits_strict (explicit AI tool/model string in question title)",
                "ask_ai_tag_hits_strict (question carries an explicit AI tool/model tag from whitelist)",
                "ask_ai_any_hits_strict (union of the above)",
            ],
            "behavior_proxies_context": [
                "answer_count_total, accepted_current_total, answer_high_share, answer_expert_share, answer_exposure_index_wavg",
            ],
        },
        "columns": {
            "user_id": "Stack Overflow user id (asker/answerer).",
            "month_id": "YYYY-MM month identifier (focal sample).",
            "first_ai_hit_month_id": "First month (within focal sample) where user has any strict AI hit in an asked question.",
            "months_since_first_ai_hit": "Event-time month index relative to first_ai_hit_month_id.",
            "ask_questions": "Number of focal-tag questions asked by the user in that month.",
            "ask_ai_any_hits_strict": "Count of asked questions in that month with strict AI disclosure (title or tag).",
            "ask_ai_any_share_strict": "ask_ai_any_hits_strict / ask_questions (NaN if ask_questions==0).",
            "answer_count_total": "Total answers posted by the user in focal tags in that month (from who_still_answers user-tag-month panel).",
            "answer_exposure_index_wavg": "Answer-weighted mean exposure_index across focal tags in that month (NaN if answer_count_total==0).",
        },
    }
    OUT_VARIABLES_JSON.write_text(json.dumps(variables, indent=2), encoding="utf-8")

    return BuildOutputs(
        monthly_counts=OUT_MONTHLY_COUNTS_CSV,
        ai_hit_users=OUT_AI_HIT_USERS_CSV,
        ai_hit_user_month_panel=OUT_AI_HIT_USER_MONTH_PANEL_CSV,
        summary_json=OUT_SUMMARY_JSON,
        variables_json=OUT_VARIABLES_JSON,
    )


def main() -> None:
    outputs = build_user_level_proxy_panel()
    print("Wrote:")
    print(f"- {outputs.monthly_counts}")
    print(f"- {outputs.ai_hit_users}")
    print(f"- {outputs.ai_hit_user_month_panel}")
    print(f"- {outputs.summary_json}")
    print(f"- {outputs.variables_json}")


if __name__ == "__main__":
    main()
