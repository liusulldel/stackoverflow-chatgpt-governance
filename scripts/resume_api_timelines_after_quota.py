from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw" / "api_extension_20250306_20251231"
STATE_JSON = RAW_DIR / "accepted_timeline_resume_state.json"
LOG_PATH = RAW_DIR / "accepted_timeline_resume.log"
QUESTIONS_JSONL = RAW_DIR / "post_2023_questions.jsonl"
TIMELINE_LOG_CSV = RAW_DIR / "post_2023_timeline_page_log.csv"
COLLECTOR_SCRIPT = BASE_DIR / "scripts" / "post_2023_stackoverflow_api_collector.py"

API_URL = "https://api.stackexchange.com/2.3/info"
API_KEY = "U4DMV*8nvpm3EOpvf69Rxw(("
SITE = "stackoverflow"


def write_log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def write_state(status: str, detail: str, remaining_accepted: int | None = None) -> None:
    payload = {
        "status": status,
        "detail": detail,
        "remaining_accepted": remaining_accepted,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    STATE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_remaining_accepted() -> int:
    questions = pd.read_json(QUESTIONS_JSONL, lines=True)
    questions = questions.sort_values(["question_id", "collected_at"]).drop_duplicates(subset=["question_id"], keep="first")
    accepted_ids = set(questions.loc[questions["accepted_answer_id"].notna(), "question_id"].astype(int).tolist())

    completed_ids: set[int] = set()
    if TIMELINE_LOG_CSV.exists():
        with TIMELINE_LOG_CSV.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "ok":
                    continue
                for chunk in row.get("question_ids", "").split(";"):
                    if chunk:
                        completed_ids.add(int(chunk))
    return len(accepted_ids - completed_ids)


def get_quota_status() -> tuple[bool, str]:
    response = requests.get(API_URL, params={"site": SITE, "key": API_KEY}, timeout=120)
    payload = response.json()
    if response.status_code == 200 and "error_id" not in payload:
        quota_remaining = payload.get("quota_remaining")
        return True, f"quota_remaining={quota_remaining}"
    error_message = payload.get("error_message", "")
    return False, error_message


def parse_wait_seconds(detail: str, default_seconds: int) -> int:
    match = re.search(r"(\d+)\s+seconds", detail)
    if match:
        return int(match.group(1)) + 60
    return default_seconds


def run_collector(batch_size: int) -> int:
    cmd = [
        sys.executable,
        str(COLLECTOR_SCRIPT),
        "--skip-questions",
        "--accepted-only",
        "--timeline-batch-size",
        str(batch_size),
    ]
    write_log(f"Launching collector: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
    if result.stdout:
        write_log(f"collector stdout: {result.stdout[-4000:]}")
    if result.stderr:
        write_log(f"collector stderr: {result.stderr[-4000:]}")
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume accepted-only Stack Overflow timeline collection when API quota becomes available.")
    parser.add_argument("--poll-seconds", type=int, default=3600, help="How often to poll for quota availability when throttled.")
    parser.add_argument("--timeline-batch-size", type=int, default=100, help="Timeline batch size to use when resuming collection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    write_log("Accepted-only timeline resume watcher started.")

    while True:
        remaining_accepted = load_remaining_accepted()
        if remaining_accepted == 0:
            write_state("completed", "No accepted-question timelines remain.", remaining_accepted=0)
            write_log("All accepted-question timelines are complete.")
            return

        quota_ok, detail = get_quota_status()
        if not quota_ok:
            sleep_seconds = max(args.poll_seconds, parse_wait_seconds(detail, args.poll_seconds))
            write_state("waiting_for_quota", detail, remaining_accepted=remaining_accepted)
            write_log(f"Quota unavailable: {detail}. Sleeping {sleep_seconds} seconds.")
            time.sleep(sleep_seconds)
            continue

        write_state("running_collector", detail, remaining_accepted=remaining_accepted)
        return_code = run_collector(args.timeline_batch_size)
        if return_code == 0:
            write_log("Collector exited cleanly; rechecking remaining accepted timelines.")
            continue

        write_state("collector_error", f"collector return code={return_code}", remaining_accepted=remaining_accepted)
        write_log(f"Collector returned non-zero exit code {return_code}. Sleeping {args.poll_seconds} seconds before retry.")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
