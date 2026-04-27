from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\AI alignment\projects\stackoverflow_chatgpt_governance")
PROCESSED = ROOT / "processed"
RAW = ROOT / "raw"
PAPER = ROOT / "paper"

QUESTION_PANEL = PROCESSED / "stackexchange_20251231_question_level_enriched.parquet"
FOCAL_ANSWERS = PROCESSED / "stackexchange_20251231_focal_answers.parquet"
ANSWER_USER_TAG_MONTH_PANEL = PROCESSED / "who_still_answers_user_tag_month_panel.parquet"
POSTS_XML = RAW / "stackexchange_20251231" / "stackoverflow.com_extracted" / "Posts.xml"
COMMENTS_XML = RAW / "stackexchange_20251231" / "stackoverflow.com_extracted" / "Comments.xml"
POST_CANDIDATES = PROCESSED / "who_still_answers_disclosed_ai_post_candidates.txt"
COMMENT_CANDIDATES = PROCESSED / "who_still_answers_disclosed_ai_comment_candidates.txt"

OUT_EVENTS_PARQUET = PROCESSED / "p1_genai_author_linked_disclosure_events.parquet"
OUT_EVENTS_CSV = PROCESSED / "p1_genai_author_linked_disclosure_events.csv"
OUT_USER_FIRST_HITS_CSV = PROCESSED / "p1_genai_author_linked_disclosure_user_first_hits.csv"
OUT_USER_MONTH_PANEL_CSV = PROCESSED / "p1_genai_author_linked_disclosure_user_month_panel.csv"
OUT_EVENT_TIME_CSV = PROCESSED / "p1_genai_author_linked_disclosure_event_time.csv"
OUT_REPEAT_SUMMARY_CSV = PROCESSED / "p1_genai_author_linked_disclosure_repeat_summary.csv"
OUT_SUMMARY_JSON = PROCESSED / "p1_genai_author_linked_disclosure_summary.json"
READOUT_PATH = PAPER / "p1_genai_author_linked_disclosure_readout_2026-04-06.md"


STRICT_PATTERNS = {
    "used_tool": r"\b(?:i|we|my|our)\s+(?:used|use|using|tried|ask(?:ed)?|prompt(?:ed)?|fed|gave|ran)\s+(?:chatgpt|copilot|claude|gemini|deepseek|openai|gpt(?:[- ]?\d(?:\.\d)?)?|llm|ai)\b",
    "tool_generated": r"\b(?:generated|written|produced|created|suggested|drafted)\s+(?:by|with)\s+(?:chatgpt|copilot|claude|gemini|deepseek|openai|gpt(?:[- ]?\d(?:\.\d)?)?|ai)\b",
    "from_tool": r"\b(?:from|via|using)\s+(?:chatgpt|copilot|claude|gemini|deepseek|openai|gpt(?:[- ]?\d(?:\.\d)?)?|llm)\b",
    "tool_said": r"\b(?:chatgpt|copilot|claude|gemini|deepseek|gpt(?:[- ]?\d(?:\.\d)?)?)(?:\s+(?:says|said|suggested|gave|returns?|wrote|generated))\b",
    "ai_assisted": r"\b(?:ai[- ]assisted|llm[- ]assisted|copilot[- ]generated|gpt[- ]generated)\b",
}

TOOL_PATTERNS = {
    "chatgpt": r"\bchatgpt\b",
    "copilot": r"\b(?:github\s+)?copilot\b",
    "claude": r"\bclaude(?:\s+code|\s+ai)?\b|claude[- ]code",
    "gemini": r"\b(?:google\s+)?gemini(?:\s+(?:ai|pro|flash|nano|1\.5|2(?:\.0)?))?\b",
    "deepseek": r"\bdeepseek\b",
    "openai": r"\bopenai\b",
    "anthropic": r"\banthropic\b",
    "gpt_family": r"\bgpt[- ]?(?:3|3\.5|4|4o|4\.1|4\.5|5|o1|o3|o4)\b|\bgenerative pre[- ]trained\b",
    "llm": r"\blarge language model(?:s)?\b|\bllm(?:s)?\b",
    "ai_generic": r"\bai\b",
}

COARSE_HINT = re.compile(
    r"chatgpt|copilot|openai|anthropic|claude|deepseek|gemini|llm|gpt|language model|ai-generated|ai assisted",
    flags=re.IGNORECASE,
)
HTML_TAG = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = HTML_TAG.sub(" ", text)
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def compile_patterns(patterns: dict[str, str]) -> dict[str, re.Pattern[str]]:
    return {label: re.compile(pattern, flags=re.IGNORECASE) for label, pattern in patterns.items()}


def match_text(text: str, compiled: dict[str, re.Pattern[str]]) -> list[str]:
    if not text:
        return []
    return [label for label, pattern in compiled.items() if pattern.search(text)]


def month_to_index(month_id: str | pd.Series) -> int | pd.Series:
    if isinstance(month_id, pd.Series):
        return pd.PeriodIndex(month_id.astype(str), freq="M").asi8
    return int(pd.Period(str(month_id), freq="M").ordinal)


def load_question_metadata() -> tuple[dict[int, dict[str, object]], set[int]]:
    cols = [
        "question_id",
        "question_created_at",
        "owner_user_id",
        "title",
        "primary_tag",
        "month_id",
        "high_tag",
        "exposure_index",
        "post_chatgpt",
        "keep_single_focal",
    ]
    q = pd.read_parquet(QUESTION_PANEL, columns=cols).copy()
    q = q.loc[q["keep_single_focal"] == 1].copy()
    q["question_id"] = q["question_id"].astype(int)
    q["question_created_at"] = pd.to_datetime(q["question_created_at"], utc=True)
    q["owner_user_id"] = pd.to_numeric(q["owner_user_id"], errors="coerce")
    meta = q.set_index("question_id").to_dict("index")
    return meta, set(meta.keys())


def load_answer_metadata(question_ids: set[int]) -> dict[int, dict[str, object]]:
    a = pd.read_parquet(
        FOCAL_ANSWERS,
        columns=["answer_id", "question_id", "owner_user_id", "answer_created_at"],
    ).copy()
    a["answer_id"] = a["answer_id"].astype(int)
    a["question_id"] = a["question_id"].astype(int)
    a["owner_user_id"] = pd.to_numeric(a["owner_user_id"], errors="coerce")
    a = a.loc[a["question_id"].isin(question_ids)].copy()
    a["answer_created_at"] = pd.to_datetime(a["answer_created_at"], utc=True, errors="coerce")
    return a.set_index("answer_id").to_dict("index")


def make_event_row(
    *,
    user_id: float | int | None,
    role: str,
    source_surface: str,
    event_ts: pd.Timestamp,
    text: str,
    strict_compiled: dict[str, re.Pattern[str]],
    tool_compiled: dict[str, re.Pattern[str]],
    question_id: int,
    answer_id: int | None,
    question_meta: dict[str, object],
) -> dict[str, object] | None:
    strict_labels = match_text(text, strict_compiled)
    if not strict_labels:
        return None
    tool_labels = match_text(text, tool_compiled)
    if not tool_labels:
        tool_labels = ["unspecified_ai"]
    user_id_num = pd.to_numeric(user_id, errors="coerce")
    return {
        "user_id": np.nan if pd.isna(user_id_num) else int(user_id_num),
        "user_attributed": 0 if pd.isna(user_id_num) else 1,
        "role": role,
        "source_surface": source_surface,
        "event_ts": event_ts,
        "event_date": event_ts.date().isoformat(),
        "month_id": str(question_meta["month_id"]),
        "month_index": month_to_index(str(question_meta["month_id"])),
        "question_id": int(question_id),
        "answer_id": np.nan if answer_id is None else int(answer_id),
        "strict_pattern_family": ";".join(sorted(strict_labels)),
        "tool_family": ";".join(sorted(tool_labels)),
        "tool_family_primary": sorted(tool_labels)[0],
        "primary_tag": question_meta["primary_tag"],
        "high_tag": int(question_meta["high_tag"]),
        "exposure_index": float(question_meta["exposure_index"]),
        "post_chatgpt": int(question_meta["post_chatgpt"]),
    }


def parse_post_events(
    question_ids: set[int],
    question_meta: dict[int, dict[str, object]],
    answer_meta: dict[int, dict[str, object]],
    strict_compiled: dict[str, re.Pattern[str]],
    tool_compiled: dict[str, re.Pattern[str]],
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    source_path = POST_CANDIDATES if POST_CANDIDATES.exists() else POSTS_XML

    with source_path.open("r", encoding="utf-8", errors="ignore", buffering=1024 * 1024) as handle:
        for line in handle:
            if "<row " not in line or COARSE_HINT.search(line) is None:
                continue
            try:
                attrs = ET.fromstring(line.strip()).attrib
            except ET.ParseError:
                continue
            post_type = attrs.get("PostTypeId")
            if post_type not in {"1", "2"}:
                continue
            raw_id = attrs.get("Id")
            if raw_id is None:
                continue
            post_id = int(raw_id)

            if post_type == "1" and post_id in question_ids:
                meta = question_meta[post_id]
                created_at = meta["question_created_at"]
                title = str(meta["title"] or "")
                body = clean_text(attrs.get("Body", "") or "")

                title_event = make_event_row(
                    user_id=meta["owner_user_id"],
                    role="asker",
                    source_surface="title",
                    event_ts=created_at,
                    text=title,
                    strict_compiled=strict_compiled,
                    tool_compiled=tool_compiled,
                    question_id=post_id,
                    answer_id=None,
                    question_meta=meta,
                )
                if title_event is not None:
                    events.append(title_event)

                body_event = make_event_row(
                    user_id=meta["owner_user_id"],
                    role="asker",
                    source_surface="question_body",
                    event_ts=created_at,
                    text=body,
                    strict_compiled=strict_compiled,
                    tool_compiled=tool_compiled,
                    question_id=post_id,
                    answer_id=None,
                    question_meta=meta,
                )
                if body_event is not None:
                    events.append(body_event)

            elif post_type == "2" and post_id in answer_meta:
                answer_info = answer_meta[post_id]
                qid = int(answer_info["question_id"])
                meta = question_meta[qid]
                created_at = answer_info["answer_created_at"]
                if pd.isna(created_at):
                    created_at = meta["question_created_at"]
                body = clean_text(attrs.get("Body", "") or "")
                answer_event = make_event_row(
                    user_id=answer_info["owner_user_id"],
                    role="answerer",
                    source_surface="answer_body",
                    event_ts=created_at,
                    text=body,
                    strict_compiled=strict_compiled,
                    tool_compiled=tool_compiled,
                    question_id=qid,
                    answer_id=post_id,
                    question_meta=meta,
                )
                if answer_event is not None:
                    events.append(answer_event)

    return events


def parse_comment_events(
    question_ids: set[int],
    question_meta: dict[int, dict[str, object]],
    answer_meta: dict[int, dict[str, object]],
    strict_compiled: dict[str, re.Pattern[str]],
    tool_compiled: dict[str, re.Pattern[str]],
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    if not COMMENTS_XML.exists():
        return events

    source_path = COMMENT_CANDIDATES if COMMENT_CANDIDATES.exists() else COMMENTS_XML
    with source_path.open("r", encoding="utf-8", errors="ignore", buffering=1024 * 1024) as handle:
        for line in handle:
            if "<row " not in line or COARSE_HINT.search(line) is None:
                continue
            try:
                attrs = ET.fromstring(line.strip()).attrib
            except ET.ParseError:
                continue
            raw_post_id = attrs.get("PostId")
            if raw_post_id is None:
                continue
            post_id = int(raw_post_id)

            if post_id in question_ids:
                qid = post_id
                role = "question_commenter"
                source_surface = "question_comment"
                answer_id = None
            elif post_id in answer_meta:
                qid = int(answer_meta[post_id]["question_id"])
                role = "answer_commenter"
                source_surface = "answer_comment"
                answer_id = post_id
            else:
                continue

            meta = question_meta[qid]
            created_raw = attrs.get("CreationDate")
            if not created_raw:
                continue
            created_at = pd.Timestamp(created_raw, tz="UTC")
            text = clean_text(attrs.get("Text", "") or "")
            event = make_event_row(
                user_id=attrs.get("UserId"),
                role=role,
                source_surface=source_surface,
                event_ts=created_at,
                text=text,
                strict_compiled=strict_compiled,
                tool_compiled=tool_compiled,
                question_id=qid,
                answer_id=answer_id,
                question_meta=meta,
            )
            if event is not None:
                events.append(event)

    return events


def build_behavior_panel(user_ids: np.ndarray) -> pd.DataFrame:
    q_cols = ["owner_user_id", "month_id", "question_id", "high_tag", "exposure_index"]
    q = pd.read_parquet(QUESTION_PANEL, columns=q_cols).copy()
    q["owner_user_id"] = pd.to_numeric(q["owner_user_id"], errors="coerce")
    q = q.loc[q["owner_user_id"].notna()].copy()
    q["user_id"] = q["owner_user_id"].astype("int64")
    q = q.loc[q["user_id"].isin(user_ids)].copy()

    ask = (
        q.groupby(["user_id", "month_id"], as_index=False)
        .agg(
            ask_questions=("question_id", "size"),
            ask_high_tag_questions=("high_tag", "sum"),
            ask_exposure_index_mean=("exposure_index", "mean"),
        )
        .sort_values(["user_id", "month_id"])
        .reset_index(drop=True)
    )
    ask["ask_high_tag_share"] = np.where(
        ask["ask_questions"] > 0,
        ask["ask_high_tag_questions"] / ask["ask_questions"],
        np.nan,
    )

    a_cols = ["answerer_user_id", "month_id", "answer_count", "accepted_current_count", "exposure_index"]
    a = pd.read_parquet(ANSWER_USER_TAG_MONTH_PANEL, columns=a_cols).copy()
    a["answerer_user_id"] = pd.to_numeric(a["answerer_user_id"], errors="coerce")
    a = a.loc[a["answerer_user_id"].notna()].copy()
    a["user_id"] = a["answerer_user_id"].astype("int64")
    a = a.loc[a["user_id"].isin(user_ids)].copy()
    a["answer_count"] = pd.to_numeric(a["answer_count"], errors="coerce").fillna(0.0)
    a["accepted_current_count"] = pd.to_numeric(a["accepted_current_count"], errors="coerce").fillna(0.0)
    a["exposure_index"] = pd.to_numeric(a["exposure_index"], errors="coerce")
    a["answer_exposure_num"] = a["answer_count"] * a["exposure_index"].fillna(0.0)

    ans = (
        a.groupby(["user_id", "month_id"], as_index=False)
        .agg(
            answer_count_total=("answer_count", "sum"),
            accepted_current_total=("accepted_current_count", "sum"),
            answer_exposure_num=("answer_exposure_num", "sum"),
        )
        .sort_values(["user_id", "month_id"])
        .reset_index(drop=True)
    )
    ans["answer_exposure_index_wavg"] = np.where(
        ans["answer_count_total"] > 0,
        ans["answer_exposure_num"] / ans["answer_count_total"],
        np.nan,
    )

    months = sorted(pd.read_parquet(QUESTION_PANEL, columns=["month_id"])["month_id"].astype(str).unique().tolist())
    base = pd.MultiIndex.from_product([user_ids, months], names=["user_id", "month_id"]).to_frame(index=False)
    panel = base.merge(ask, on=["user_id", "month_id"], how="left").merge(ans, on=["user_id", "month_id"], how="left")
    for col in ["ask_questions", "ask_high_tag_questions", "answer_count_total", "accepted_current_total", "answer_exposure_num"]:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0.0)
    panel["month_index"] = month_to_index(panel["month_id"])
    return panel


def build_outputs(events: pd.DataFrame) -> None:
    events = events.sort_values(["event_ts", "question_id", "answer_id", "source_surface"]).reset_index(drop=True)
    events.to_parquet(OUT_EVENTS_PARQUET, index=False)
    events.to_csv(OUT_EVENTS_CSV, index=False)

    attributed = events.loc[events["user_attributed"] == 1].copy()
    user_first = (
        attributed.sort_values(["user_id", "event_ts"])
        .groupby("user_id", as_index=False)
        .agg(
            first_event_ts=("event_ts", "min"),
            first_month_id=("month_id", "first"),
            n_events_total=("user_id", "size"),
            n_event_months=("month_id", "nunique"),
            n_roles=("role", "nunique"),
        )
    )
    user_first["repeat_disclosure_user"] = (user_first["n_event_months"] >= 2).astype(int)
    user_first.to_csv(OUT_USER_FIRST_HITS_CSV, index=False)

    user_month = (
        attributed.groupby(["user_id", "month_id"], as_index=False)
        .agg(
            event_count=("user_id", "size"),
            roles_nunique=("role", "nunique"),
            tool_families_nunique=("tool_family_primary", "nunique"),
            asker_events=("role", lambda s: int((s == "asker").sum())),
            answerer_events=("role", lambda s: int((s == "answerer").sum())),
            question_commenter_events=("role", lambda s: int((s == "question_commenter").sum())),
            answer_commenter_events=("role", lambda s: int((s == "answer_commenter").sum())),
            high_tag_event_share=("high_tag", "mean"),
            mean_event_exposure=("exposure_index", "mean"),
        )
        .sort_values(["user_id", "month_id"])
        .reset_index(drop=True)
    )
    user_month["month_index"] = month_to_index(user_month["month_id"])
    user_month = user_month.merge(user_first[["user_id", "first_month_id", "repeat_disclosure_user"]], on="user_id", how="left")
    user_month["event_time"] = user_month["month_index"] - month_to_index(user_month["first_month_id"])
    user_month.to_csv(OUT_USER_MONTH_PANEL_CSV, index=False)

    user_ids = user_first["user_id"].to_numpy(dtype="int64")
    behavior = build_behavior_panel(user_ids)
    behavior = behavior.merge(user_first[["user_id", "first_month_id", "repeat_disclosure_user"]], on="user_id", how="left")
    behavior["event_time"] = behavior["month_index"] - month_to_index(behavior["first_month_id"])
    behavior = behavior.merge(
        user_month[
            [
                "user_id",
                "month_id",
                "event_count",
                "roles_nunique",
                "tool_families_nunique",
                "high_tag_event_share",
                "mean_event_exposure",
            ]
        ],
        on=["user_id", "month_id"],
        how="left",
    )
    for col in ["event_count", "roles_nunique", "tool_families_nunique"]:
        behavior[col] = behavior[col].fillna(0.0)
    event_time = (
        behavior.loc[behavior["event_time"].between(-6, 6)]
        .groupby("event_time", as_index=False)
        .agg(
            users=("user_id", "nunique"),
            ask_questions_mean=("ask_questions", "mean"),
            ask_high_tag_share_mean=("ask_high_tag_share", "mean"),
            ask_exposure_index_mean=("ask_exposure_index_mean", "mean"),
            answer_count_total_mean=("answer_count_total", "mean"),
            accepted_current_total_mean=("accepted_current_total", "mean"),
            answer_exposure_index_wavg_mean=("answer_exposure_index_wavg", "mean"),
            event_count_mean=("event_count", "mean"),
            high_tag_event_share_mean=("high_tag_event_share", "mean"),
            mean_event_exposure_mean=("mean_event_exposure", "mean"),
        )
        .sort_values("event_time")
        .reset_index(drop=True)
    )
    event_time.to_csv(OUT_EVENT_TIME_CSV, index=False)

    repeat_summary = (
        behavior.assign(post_first=(behavior["event_time"] >= 0).astype(int))
        .groupby(["repeat_disclosure_user", "post_first"], as_index=False)
        .agg(
            users=("user_id", "nunique"),
            ask_questions_mean=("ask_questions", "mean"),
            ask_high_tag_share_mean=("ask_high_tag_share", "mean"),
            ask_exposure_index_mean=("ask_exposure_index_mean", "mean"),
            answer_count_total_mean=("answer_count_total", "mean"),
            accepted_current_total_mean=("accepted_current_total", "mean"),
            event_count_mean=("event_count", "mean"),
        )
    )
    repeat_summary.to_csv(OUT_REPEAT_SUMMARY_CSV, index=False)

    role_counts = attributed["role"].value_counts().to_dict()
    summary = {
        "n_events_total": int(len(events)),
        "n_events_attributed": int((events["user_attributed"] == 1).sum()),
        "n_unique_users_attributed": int(attributed["user_id"].nunique()),
        "n_unique_questions": int(events["question_id"].nunique()),
        "n_answer_side_events": int((events["role"] == "answerer").sum()),
        "n_comment_events": int(events["role"].isin(["question_commenter", "answer_commenter"]).sum()),
        "role_counts": role_counts,
        "n_repeat_users": int(user_first["repeat_disclosure_user"].sum()),
        "share_repeat_users": float(user_first["repeat_disclosure_user"].mean()),
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readout = f"""# P1 Author-Linked Direct-Disclosure Event Layer

Date: `2026-04-06`

## Purpose

This build upgrades the GenAI user layer from a question-title / exact-tag disclosure proxy to an author-linked direct-disclosure event panel using the local Stack Overflow dump.

## What This Layer Observes

- explicit direct-disclosure events in question titles
- explicit direct-disclosure events in question bodies
- explicit direct-disclosure events in answer bodies
- explicit direct-disclosure events in question comments
- explicit direct-disclosure events in answer comments

The layer attributes events to:

- askers
- answerers
- question commenters
- answer commenters

It remains a disclosure-event layer, not telemetry.

## Headline Counts

- total events: `{summary['n_events_total']}`
- attributed events: `{summary['n_events_attributed']}`
- attributed unique users: `{summary['n_unique_users_attributed']}`
- unique focal questions touched: `{summary['n_unique_questions']}`
- answer-side events: `{summary['n_answer_side_events']}`
- comment events: `{summary['n_comment_events']}`
- repeat users (`2+` disclosure months): `{summary['n_repeat_users']}` (`{summary['share_repeat_users']:.3f}`)

## Role Counts

{json.dumps(role_counts, indent=2)}

## Safe Read

This layer is materially closer to direct adoption evidence than title-only or tag-only question proxies because it records user-linked self-disclosure events across multiple content surfaces. It is still not telemetry because it misses silent, private, or non-disclosed AI use.

## Files

- events parquet: `{OUT_EVENTS_PARQUET}`
- events csv: `{OUT_EVENTS_CSV}`
- user first hits: `{OUT_USER_FIRST_HITS_CSV}`
- user-month panel: `{OUT_USER_MONTH_PANEL_CSV}`
- event-time summary: `{OUT_EVENT_TIME_CSV}`
- repeat summary: `{OUT_REPEAT_SUMMARY_CSV}`
- summary json: `{OUT_SUMMARY_JSON}`
"""
    READOUT_PATH.write_text(readout, encoding="utf-8")


def main() -> None:
    strict_compiled = compile_patterns(STRICT_PATTERNS)
    tool_compiled = compile_patterns(TOOL_PATTERNS)
    question_meta, question_ids = load_question_metadata()
    answer_meta = load_answer_metadata(question_ids)
    events = parse_post_events(question_ids, question_meta, answer_meta, strict_compiled, tool_compiled)
    events.extend(parse_comment_events(question_ids, question_meta, answer_meta, strict_compiled, tool_compiled))
    events_df = pd.DataFrame(events)
    if events_df.empty:
        raise RuntimeError("No author-linked disclosure events were built.")
    build_outputs(events_df)


if __name__ == "__main__":
    main()
