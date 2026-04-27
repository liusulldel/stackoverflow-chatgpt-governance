from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw" / "api_extension_20250306_20251231"
PROCESSED_DIR = BASE_DIR / "processed"

QUESTIONS_JSONL = RAW_DIR / "post_2023_questions.jsonl"
QUESTION_LOG_CSV = RAW_DIR / "post_2023_question_page_log.csv"
TIMELINE_JSONL = RAW_DIR / "post_2023_timelines.jsonl"
TIMELINE_LOG_CSV = RAW_DIR / "post_2023_timeline_page_log.csv"
MANIFEST_JSON = RAW_DIR / "post_2023_collection_manifest.json"

QUESTIONS_PARQUET = PROCESSED_DIR / "post_2023_api_questions.parquet"
QUESTION_REQUESTS_PARQUET = PROCESSED_DIR / "post_2023_api_question_requests.parquet"
QUESTION_EVENTS_PARQUET = PROCESSED_DIR / "post_2023_api_question_events.parquet"
QUESTION_SUMMARY_JSON = PROCESSED_DIR / "post_2023_api_collection_summary.json"

API_BASE = "https://api.stackexchange.com/2.3"
SITE = "stackoverflow"
API_KEY = os.environ.get("STACKEXCHANGE_API_KEY")
PAGESIZE = 100
DEFAULT_TIMELINE_BATCH_SIZE = 100

START_AT = datetime(2023, 3, 6, tzinfo=timezone.utc)
END_EXCLUSIVE = datetime(2026, 1, 1, tzinfo=timezone.utc)
WINDOW_DAYS = 7

SELECTED_TAGS = [
    "apache-spark",
    "android",
    "bash",
    "docker",
    "excel",
    "firebase",
    "javascript",
    "kubernetes",
    "linux",
    "memory-management",
    "multithreading",
    "numpy",
    "pandas",
    "python",
    "regex",
    "sql",
]


@dataclass(frozen=True)
class TimeWindow:
    tag: str
    start_at: datetime
    end_exclusive: datetime

    @property
    def label(self) -> str:
        return f"{self.tag}_{self.start_at.strftime('%Y%m%d')}_{(self.end_exclusive - timedelta(seconds=1)).strftime('%Y%m%d')}"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def unix_ts(dt: datetime) -> int:
    return int(dt.timestamp())


def iter_windows(tags: Iterable[str], start_at: datetime, end_exclusive: datetime) -> list[TimeWindow]:
    windows: list[TimeWindow] = []
    for tag in tags:
        cursor = start_at
        while cursor < end_exclusive:
            nxt = min(cursor + timedelta(days=WINDOW_DAYS), end_exclusive)
            windows.append(TimeWindow(tag=tag, start_at=cursor, end_exclusive=nxt))
            cursor = nxt
    return windows


def read_jsonl_ids(path: Path, field: str) -> set[int]:
    if not path.exists():
        return set()
    ids: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get(field)
            if value is not None:
                ids.add(int(value))
    return ids


def read_accepted_question_ids(path: Path) -> list[int]:
    if not path.exists():
        return []
    questions = pd.read_json(path, lines=True)
    if questions.empty:
        return []
    questions = questions.sort_values(["question_id", "collected_at"]).drop_duplicates(subset=["question_id"], keep="first")
    accepted = questions.loc[questions["accepted_answer_id"].notna(), "question_id"].astype(int).tolist()
    return sorted(accepted)


def read_completed_windows(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") == "ok":
                completed.add(row["window_label"])
    return completed


def read_timeline_completed_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != "ok":
                continue
            ids_field = row.get("question_ids", "")
            if not ids_field:
                continue
            for chunk in ids_field.split(";"):
                if chunk:
                    completed.add(int(chunk))
    return completed


def append_csv_row(path: Path, header: list[str], row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def stackexchange_params(params: dict) -> dict:
    clean_params = dict(params)
    if API_KEY:
        clean_params["key"] = API_KEY
    else:
        clean_params.pop("key", None)
    return clean_params


def request_json(endpoint: str, params: dict, session: requests.Session, max_attempts: int = 6) -> dict:
    params = stackexchange_params(params)
    for attempt in range(1, max_attempts + 1):
        response = session.get(endpoint, params=params, timeout=120)
        if response.status_code in {502, 503, 504, 520, 521, 522, 524}:
            time.sleep(min(2**attempt, 60))
            continue
        response.raise_for_status()
        payload = response.json()
        if "error_id" in payload:
            if payload.get("error_name") in {"throttle_violation", "temporarily_unavailable"}:
                time.sleep(min(2**attempt, 120))
                continue
            raise RuntimeError(f"Stack Exchange API error: {payload}")
        if payload.get("backoff"):
            time.sleep(int(payload["backoff"]) + 1)
        return payload
    raise RuntimeError(f"Failed after {max_attempts} attempts: {endpoint} {params}")


def normalize_question(item: dict, window: TimeWindow, page: int) -> dict:
    owner = item.get("owner") or {}
    return {
        "question_id": int(item["question_id"]),
        "question_created_at": datetime.fromtimestamp(item["creation_date"], tz=timezone.utc).isoformat(),
        "title": item.get("title"),
        "link": item.get("link"),
        "tags": item.get("tags", []),
        "tags_semicolon": ";".join(item.get("tags", [])),
        "accepted_answer_id": item.get("accepted_answer_id"),
        "answer_count_snapshot": item.get("answer_count"),
        "view_count_snapshot": item.get("view_count"),
        "score_snapshot": item.get("score"),
        "is_answered_snapshot": item.get("is_answered"),
        "owner_user_id": owner.get("user_id"),
        "owner_account_id": owner.get("account_id"),
        "owner_user_type": owner.get("user_type"),
        "requested_tag": window.tag,
        "window_start_at": window.start_at.isoformat(),
        "window_end_exclusive_at": window.end_exclusive.isoformat(),
        "window_label": window.label,
        "page": page,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def normalize_timeline_event(item: dict, question_id_batch: list[int], page: int) -> dict:
    owner = item.get("owner") or item.get("user") or {}
    event_ts = item.get("creation_date")
    created_at = None
    if event_ts is not None:
        created_at = datetime.fromtimestamp(event_ts, tz=timezone.utc).isoformat()
    return {
        "question_id": item.get("question_id"),
        "post_id": item.get("post_id"),
        "timeline_type": item.get("timeline_type"),
        "timeline_created_at": created_at,
        "comment_id": item.get("comment_id"),
        "revision_guid": item.get("revision_guid"),
        "owner_user_id": owner.get("user_id"),
        "owner_account_id": owner.get("account_id"),
        "batch_question_ids": ";".join(str(x) for x in question_id_batch),
        "page": page,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_questions(session: requests.Session, windows: list[TimeWindow], max_windows: int | None = None) -> None:
    completed = read_completed_windows(QUESTION_LOG_CSV)
    processed = 0
    for window in windows:
        if window.label in completed:
            continue
        if max_windows is not None and processed >= max_windows:
            break
        page = 1
        total_rows = 0
        quota_remaining = None
        status = "ok"
        error_message = ""
        try:
            while True:
                payload = request_json(
                    f"{API_BASE}/questions",
                    {
                        "site": SITE,
                        "tagged": window.tag,
                        "sort": "creation",
                        "order": "asc",
                        "fromdate": unix_ts(window.start_at),
                        "todate": unix_ts(window.end_exclusive - timedelta(seconds=1)),
                        "pagesize": PAGESIZE,
                        "page": page,
                    },
                    session,
                )
                items = payload.get("items", [])
                total_rows += append_jsonl(QUESTIONS_JSONL, (normalize_question(item, window, page) for item in items))
                quota_remaining = payload.get("quota_remaining")
                if not payload.get("has_more"):
                    break
                page += 1
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error_message = str(exc)

        append_csv_row(
            QUESTION_LOG_CSV,
            [
                "window_label",
                "requested_tag",
                "window_start_at",
                "window_end_exclusive_at",
                "pages_fetched",
                "rows_fetched",
                "status",
                "quota_remaining",
                "error_message",
                "logged_at",
            ],
            {
                "window_label": window.label,
                "requested_tag": window.tag,
                "window_start_at": window.start_at.isoformat(),
                "window_end_exclusive_at": window.end_exclusive.isoformat(),
                "pages_fetched": page,
                "rows_fetched": total_rows,
                "status": status,
                "quota_remaining": quota_remaining,
                "error_message": error_message,
                "logged_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        processed += 1
        if status != "ok":
            raise RuntimeError(f"Question fetch failed for {window.label}: {error_message}")


def fetch_single_timeline_batch(
    session: requests.Session,
    batch: list[int],
    batch_index: str,
) -> tuple[int, int, str, str]:
    ids_path = ";".join(str(qid) for qid in batch)
    page = 1
    total_rows = 0
    quota_remaining = None
    status = "ok"
    error_message = ""
    try:
        while True:
            payload = request_json(
                f"{API_BASE}/questions/{ids_path}/timeline",
                {
                    "site": SITE,
                    "pagesize": PAGESIZE,
                    "page": page,
                },
                session,
            )
            items = payload.get("items", [])
            total_rows += append_jsonl(TIMELINE_JSONL, (normalize_timeline_event(item, batch, page) for item in items))
            quota_remaining = payload.get("quota_remaining")
            if not payload.get("has_more"):
                break
            page += 1
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error_message = str(exc)

    append_csv_row(
        TIMELINE_LOG_CSV,
        [
            "batch_index",
            "question_ids",
            "batch_size",
            "pages_fetched",
            "rows_fetched",
            "status",
            "quota_remaining",
            "error_message",
            "logged_at",
        ],
        {
            "batch_index": batch_index,
            "question_ids": ids_path,
            "batch_size": len(batch),
            "pages_fetched": page,
            "rows_fetched": total_rows,
            "status": status,
            "quota_remaining": quota_remaining,
            "error_message": error_message,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return total_rows, page, status, error_message


def fetch_timeline_batch_with_split(
    session: requests.Session,
    batch: list[int],
    batch_index: str,
) -> int:
    total_rows, _, status, error_message = fetch_single_timeline_batch(session, batch, batch_index)
    if status == "ok":
        return total_rows
    if "400 Client Error" in error_message and len(batch) > 1:
        midpoint = len(batch) // 2
        left = batch[:midpoint]
        right = batch[midpoint:]
        return fetch_timeline_batch_with_split(session, left, f"{batch_index}a") + fetch_timeline_batch_with_split(
            session, right, f"{batch_index}b"
        )
    raise RuntimeError(f"Timeline fetch failed for batch {batch_index}: {error_message}")


def fetch_timelines(
    session: requests.Session,
    question_ids: list[int],
    max_batches: int | None = None,
    timeline_batch_size: int = DEFAULT_TIMELINE_BATCH_SIZE,
) -> None:
    completed_ids = read_timeline_completed_ids(TIMELINE_LOG_CSV)
    remaining_ids = [qid for qid in question_ids if qid not in completed_ids]
    if not remaining_ids:
        return

    n_batches = math.ceil(len(remaining_ids) / timeline_batch_size)
    processed = 0
    for batch_index in range(n_batches):
        if max_batches is not None and processed >= max_batches:
            break
        batch = remaining_ids[batch_index * timeline_batch_size : (batch_index + 1) * timeline_batch_size]
        processed += 1
        fetch_timeline_batch_with_split(session, batch, str(batch_index))


def build_processed_outputs() -> dict:
    if not QUESTIONS_JSONL.exists():
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_JSONL}")

    questions = pd.read_json(QUESTIONS_JSONL, lines=True)
    if questions.empty:
        raise RuntimeError("No question rows collected.")

    questions["question_created_at"] = pd.to_datetime(questions["question_created_at"], utc=True)
    questions = questions.sort_values(["question_id", "requested_tag", "window_start_at", "page"]).reset_index(drop=True)

    requests_df = questions.copy()
    dedup_questions = questions.drop_duplicates(subset=["question_id"], keep="first").copy()
    dedup_questions["question_year"] = dedup_questions["question_created_at"].dt.year
    dedup_questions["question_month"] = dedup_questions["question_created_at"].dt.strftime("%Y-%m")

    requests_df.to_parquet(QUESTION_REQUESTS_PARQUET, index=False)
    dedup_questions.to_parquet(QUESTIONS_PARQUET, index=False)

    event_summary = pd.DataFrame(columns=[
        "question_id",
        "first_answer_at",
        "first_accepted_at",
        "last_unaccepted_at",
        "n_answer_events",
        "n_accepted_events",
        "n_unaccepted_events",
    ])

    if TIMELINE_JSONL.exists():
        timelines = pd.read_json(TIMELINE_JSONL, lines=True)
        if not timelines.empty:
            timelines["timeline_created_at"] = pd.to_datetime(timelines["timeline_created_at"], utc=True, errors="coerce")

            def summarize(group: pd.DataFrame) -> pd.Series:
                answer_events = group.loc[group["timeline_type"] == "answer", "timeline_created_at"]
                accepted_events = group.loc[group["timeline_type"] == "accepted_answer", "timeline_created_at"]
                unaccepted_events = group.loc[group["timeline_type"] == "unaccepted_answer", "timeline_created_at"]
                return pd.Series(
                    {
                        "first_answer_at": answer_events.min() if not answer_events.empty else pd.NaT,
                        "first_accepted_at": accepted_events.min() if not accepted_events.empty else pd.NaT,
                        "last_unaccepted_at": unaccepted_events.max() if not unaccepted_events.empty else pd.NaT,
                        "n_answer_events": int((group["timeline_type"] == "answer").sum()),
                        "n_accepted_events": int((group["timeline_type"] == "accepted_answer").sum()),
                        "n_unaccepted_events": int((group["timeline_type"] == "unaccepted_answer").sum()),
                    }
                )

            event_summary = timelines.groupby("question_id", dropna=True).apply(summarize).reset_index()
            event_summary.to_parquet(QUESTION_EVENTS_PARQUET, index=False)

    summary = {
        "questions_jsonl": str(QUESTIONS_JSONL),
        "timelines_jsonl": str(TIMELINE_JSONL),
        "questions_parquet": str(QUESTIONS_PARQUET),
        "question_requests_parquet": str(QUESTION_REQUESTS_PARQUET),
        "question_events_parquet": str(QUESTION_EVENTS_PARQUET),
        "n_question_request_rows": int(len(requests_df)),
        "n_unique_questions": int(dedup_questions["question_id"].nunique()),
        "n_timeline_summaries": int(len(event_summary)),
        "min_question_created_at": dedup_questions["question_created_at"].min().isoformat(),
        "max_question_created_at": dedup_questions["question_created_at"].max().isoformat(),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    QUESTION_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_manifest(summary: dict, args: argparse.Namespace) -> None:
    manifest = {
        "api_base": API_BASE,
        "site": SITE,
        "api_key_source": "STACKEXCHANGE_API_KEY environment variable if set; omitted otherwise",
        "start_at": START_AT.isoformat(),
        "end_exclusive": END_EXCLUSIVE.isoformat(),
        "window_days": WINDOW_DAYS,
        "selected_tags": SELECTED_TAGS,
        "max_windows": args.max_windows,
        "max_timeline_batches": args.max_timeline_batches,
        "summary": summary,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect post-2023 Stack Overflow data for focal tags via official API.")
    parser.add_argument("--max-windows", type=int, default=None, help="Limit number of question windows for a partial run.")
    parser.add_argument("--max-timeline-batches", type=int, default=None, help="Limit number of timeline batches for a partial run.")
    parser.add_argument("--timeline-batch-size", type=int, default=DEFAULT_TIMELINE_BATCH_SIZE, help="Number of question ids per timeline batch.")
    parser.add_argument("--accepted-only", action="store_true", help="When fetching timelines, only request timelines for questions with a non-null accepted_answer_id in the raw questions file.")
    parser.add_argument("--skip-questions", action="store_true", help="Skip question collection and only build outputs from existing raw files.")
    parser.add_argument("--skip-timelines", action="store_true", help="Skip timeline collection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    windows = iter_windows(SELECTED_TAGS, START_AT, END_EXCLUSIVE)
    session = requests.Session()
    session.headers.update({"User-Agent": "codex-stackoverflow-governance/1.0"})

    if not args.skip_questions:
        fetch_questions(session, windows, max_windows=args.max_windows)

    question_ids = read_accepted_question_ids(QUESTIONS_JSONL) if args.accepted_only else sorted(read_jsonl_ids(QUESTIONS_JSONL, "question_id"))
    if question_ids and not args.skip_timelines:
        fetch_timelines(
            session,
            question_ids,
            max_batches=args.max_timeline_batches,
            timeline_batch_size=args.timeline_batch_size,
        )

    summary = build_processed_outputs()
    write_manifest(summary, args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
