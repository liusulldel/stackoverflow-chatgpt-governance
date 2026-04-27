from __future__ import annotations

import csv
import html
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
PAPER_DIR = BASE_DIR / "paper" / "staged_public_resolution"
RAW_POSTS_XML = BASE_DIR / "raw" / "stackexchange_20251231" / "stackoverflow.com_extracted" / "Posts.xml"

QUESTION_FILE = PROCESSED_DIR / "stackexchange_20251231_question_level_enriched.parquet"
COMPLEXITY_FILE = PROCESSED_DIR / "stackexchange_20251231_question_complexity_features.parquet"
WAVE1_LABEL_FILE = PROCESSED_DIR / "question_level_exposure_labels.csv"

SAMPLE_FILE = PROCESSED_DIR / "question_level_exposure_wave2_sample_576.csv"
BODY_FILE = PROCESSED_DIR / "question_level_exposure_wave2_sample_bodies.csv"
RAW_RESPONSE_FILE = PROCESSED_DIR / "question_level_exposure_wave2_api_responses.jsonl"
LABEL_FILE = PROCESSED_DIR / "question_level_exposure_wave2_labels.csv"
SUMMARY_JSON = PROCESSED_DIR / "question_level_exposure_wave2_summary.json"
SUMMARY_MD = PAPER_DIR / "question_level_exposure_wave2.md"
CALL_LEDGER = PAPER_DIR / "exposure_call_ledger_2026-04-04.csv"

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:2455/v1").rstrip("/")
MAIN_MODEL = "gpt-5.1-codex-mini"
CAL_MODEL = "gpt-5.4-mini"

TARGET_PER_STRATUM = 16
CALIBRATION_PER_STRATUM = 2
MAX_BODY_CHARS = 1800
CONCURRENCY = 4
SEED = 314
MAX_RETRIES = 8
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

SYSTEM_PROMPT = """You are annotating Stack Overflow questions for a research project on question-level private-AI substitutability.

Rate the question on five fields using only integers 1-5:
- snippet_solvable_no_hidden_state: 1=very unlikely to be solved from the prompt alone, 5=very likely
- environment_context_required: 1=very little hidden environment/configuration context needed, 5=a lot is needed
- iterative_clarification_needed: 1=little back-and-forth needed, 5=extensive clarification likely needed
- systems_debugging_integration: 1=mostly deterministic transformation/syntax/query logic, 5=systems/integration/runtime debugging
- overall_private_ai_substitutability: 1=low substitutability by early private GenAI, 5=high substitutability

Use the title, tags, and body excerpt only. Return strict JSON with exactly these keys plus:
- confidence: number from 0 to 1
- rationale: short string under 30 words
Do not include markdown fences or extra text."""


def load_base() -> pd.DataFrame:
    questions = pd.read_parquet(
        QUESTION_FILE,
        columns=[
            "question_id",
            "question_created_at",
            "primary_tag",
            "title",
            "question_tags",
            "post_chatgpt",
            "exposure_index",
        ],
    )
    complexity = pd.read_parquet(COMPLEXITY_FILE)
    merged = questions.merge(complexity, on="question_id", how="left")
    merged["stratum"] = merged["primary_tag"] + "__" + merged["post_chatgpt"].astype(str)
    return merged


def prior_question_ids() -> set[int]:
    ids: set[int] = set()
    if WAVE1_LABEL_FILE.exists():
        wave1 = pd.read_csv(WAVE1_LABEL_FILE, usecols=["question_id"])
        ids.update(wave1["question_id"].astype(int).tolist())
    if LABEL_FILE.exists():
        wave2 = pd.read_csv(LABEL_FILE, usecols=["question_id"])
        ids.update(wave2["question_id"].astype(int).tolist())
    return ids


def build_sample(df: pd.DataFrame) -> pd.DataFrame:
    used = prior_question_ids()
    rows = []
    for (tag, post), group in df.groupby(["primary_tag", "post_chatgpt"], observed=False):
        group = group.loc[~group["question_id"].astype(int).isin(used)].copy()
        if len(group) < TARGET_PER_STRATUM:
            raise RuntimeError(f"Not enough remaining questions in stratum {(tag, post)} for wave2.")

        sampled = group.sample(n=TARGET_PER_STRATUM, random_state=SEED)
        sampled = sampled.copy()
        sampled["sample_role"] = "main"
        sampled["label_pass"] = 1
        sampled["call_model"] = MAIN_MODEL
        rows.append(sampled)

        cal = sampled.sample(n=CALIBRATION_PER_STRATUM, random_state=SEED + 7).copy()
        cal["sample_role"] = "calibration"
        cal["label_pass"] = 2
        cal["call_model"] = CAL_MODEL
        rows.append(cal)

    sample = pd.concat(rows, ignore_index=True)
    sample["sample_row_id"] = np.arange(1, len(sample) + 1)
    sample["question_created_at"] = sample["question_created_at"].astype(str)
    sample = sample.sort_values(["primary_tag", "post_chatgpt", "label_pass", "sample_row_id"]).reset_index(drop=True)
    return sample


def strip_html(raw_html: str) -> str:
    text = raw_html or ""
    text = re.sub(r"<pre><code>(.*?)</code></pre>", lambda m: "\n[CODE]\n" + html.unescape(m.group(1)) + "\n[/CODE]\n", text, flags=re.I | re.S)
    text = re.sub(r"<code>(.*?)</code>", lambda m: " " + html.unescape(m.group(1)) + " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_BODY_CHARS]


def extract_bodies(sample: pd.DataFrame) -> pd.DataFrame:
    needed = set(sample["question_id"].astype(int).tolist())
    rows = []
    for _event, elem in ET.iterparse(RAW_POSTS_XML, events=("end",)):
        if elem.tag != "row":
            continue
        post_type = elem.attrib.get("PostTypeId")
        qid = elem.attrib.get("Id")
        if post_type == "1" and qid and int(qid) in needed:
            rows.append({"question_id": int(qid), "body_excerpt": strip_html(elem.attrib.get("Body", ""))})
            needed.remove(int(qid))
            if not needed:
                elem.clear()
                break
        elem.clear()
    return pd.DataFrame(rows)


def make_user_prompt(row: pd.Series) -> str:
    return (
        f"primary_tag: {row['primary_tag']}\n"
        f"question_tags: {row['question_tags']}\n"
        f"title: {row['title']}\n"
        f"body_excerpt: {row['body_excerpt']}\n"
    )


def parse_json_text(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start : end + 1])


def call_api(prompt: str, model: str) -> tuple[dict, dict]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = parse_json_text(content)
            return data, parsed
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in RETRYABLE_STATUS or attempt == MAX_RETRIES:
                raise
        except urllib.error.URLError as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                raise
        except TimeoutError as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                raise
        time.sleep(delay)
        delay = min(delay * 1.5, 20.0)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("API call failed without a captured exception.")


def append_jsonl(path: Path, row: dict, lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_calls(sample: pd.DataFrame) -> pd.DataFrame:
    completed_ids = set()
    if RAW_RESPONSE_FILE.exists():
        with RAW_RESPONSE_FILE.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    obj = json.loads(line)
                    if obj.get("status") == "success":
                        completed_ids.add(int(obj["sample_row_id"]))
                except Exception:
                    continue

    lock = threading.Lock()
    tasks = sample.loc[~sample["sample_row_id"].isin(completed_ids)].copy()
    if tasks.empty:
        rows = [json.loads(line) for line in RAW_RESPONSE_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)

    def worker(row_dict: dict) -> dict:
        row = pd.Series(row_dict)
        prompt = make_user_prompt(row)
        try:
            start = time.time()
            api_data, parsed = call_api(prompt, row["call_model"])
            elapsed = time.time() - start
            out = {
                "sample_row_id": int(row["sample_row_id"]),
                "question_id": int(row["question_id"]),
                "label_pass": int(row["label_pass"]),
                "call_model": row["call_model"],
                "status": "success",
                "elapsed_seconds": round(elapsed, 3),
                "usage": api_data.get("usage", {}),
                "response_id": api_data.get("id"),
                "parsed": parsed,
            }
            append_jsonl(RAW_RESPONSE_FILE, out, lock)
            return out
        except Exception as exc:
            return {
                "sample_row_id": int(row["sample_row_id"]),
                "question_id": int(row["question_id"]),
                "label_pass": int(row["label_pass"]),
                "call_model": row["call_model"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    failed_rows = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(worker, row.to_dict()) for _, row in tasks.iterrows()]
        for future in as_completed(futures):
            result = future.result()
            if result.get("status") != "success":
                failed_rows.append(result)

    if failed_rows:
        raise RuntimeError(f"Wave 2 labeling still has {len(failed_rows)} failed rows after retries: {failed_rows[:3]}")


    rows = [json.loads(line) for line in RAW_RESPONSE_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def build_label_frame(sample: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in responses.iterrows():
        parsed = row["parsed"]
        rows.append(
            {
                "sample_row_id": int(row["sample_row_id"]),
                "question_id": int(row["question_id"]),
                "label_pass": int(row["label_pass"]),
                "call_model": row["call_model"],
                "snippet_solvable_no_hidden_state": int(parsed["snippet_solvable_no_hidden_state"]),
                "environment_context_required": int(parsed["environment_context_required"]),
                "iterative_clarification_needed": int(parsed["iterative_clarification_needed"]),
                "systems_debugging_integration": int(parsed["systems_debugging_integration"]),
                "overall_private_ai_substitutability": int(parsed["overall_private_ai_substitutability"]),
                "confidence": float(parsed["confidence"]),
                "rationale": str(parsed["rationale"]),
            }
        )
    labels = pd.DataFrame(rows)
    merged = sample.merge(labels, on=["sample_row_id", "question_id", "label_pass", "call_model"], how="left")
    merged["rubric_exposure_score"] = (
        merged["snippet_solvable_no_hidden_state"]
        + (6 - merged["environment_context_required"])
        + (6 - merged["iterative_clarification_needed"])
        + (6 - merged["systems_debugging_integration"])
    ) / 4.0
    return merged


def summarize(labels: pd.DataFrame, response_count: int) -> dict:
    main = labels.loc[labels["label_pass"] == 1].copy()
    cal = labels.loc[labels["label_pass"] == 2].copy()
    agreement = {}
    if not cal.empty:
        paired = main.merge(
            cal[["question_id", "overall_private_ai_substitutability", "rubric_exposure_score"]],
            on="question_id",
            suffixes=("_p1", "_p2"),
            how="inner",
        )
        if not paired.empty:
            agreement = {
                "n_paired": int(len(paired)),
                "exact_overall_match_rate": float(np.mean(paired["overall_private_ai_substitutability_p1"] == paired["overall_private_ai_substitutability_p2"])),
                "mean_abs_overall_diff": float(np.mean(np.abs(paired["overall_private_ai_substitutability_p1"] - paired["overall_private_ai_substitutability_p2"]))),
                "mean_abs_rubric_diff": float(np.mean(np.abs(paired["rubric_exposure_score_p1"] - paired["rubric_exposure_score_p2"]))),
            }
    by_group = (
        main.groupby(["primary_tag", "post_chatgpt"], observed=False)["rubric_exposure_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "successful_api_calls": int(response_count),
        "target_calls": int(len(labels)),
        "call_target_met": bool(response_count >= 500),
        "unique_questions_labeled": int(main["question_id"].nunique()),
        "main_labels": int(len(main)),
        "calibration_labels": int(len(cal)),
        "overall_mean_rubric_exposure": float(main["rubric_exposure_score"].mean()),
        "within_tag_sd_mean": float(main.groupby("primary_tag")["rubric_exposure_score"].std().mean()),
        "agreement": agreement,
        "by_group": by_group,
    }


def append_ledger(count_increment: int, cumulative_total: int, notes: str) -> None:
    with CALL_LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(
            f"exposure_index_labeling_wave2,local_llm_api,{count_increment},{cumulative_total},question_level_exposure_wave2_api_responses.jsonl,{notes}\n"
        )


def write_summary_md(summary: dict) -> None:
    lines = [
        "# Question-Level Exposure Index Wave 2",
        "",
        "## API Labeling Summary",
        "",
        f"- Successful API calls in wave 2: `{summary['successful_api_calls']}`",
        f"- Target met (`>=500`): `{summary['call_target_met']}`",
        f"- Main labels: `{summary['main_labels']}`",
        f"- Calibration labels: `{summary['calibration_labels']}`",
        f"- Unique questions labeled: `{summary['unique_questions_labeled']}`",
        f"- Mean within-tag SD of rubric exposure: `{summary['within_tag_sd_mean']:.4f}`",
        "",
        "## Calibration Agreement",
        "",
        f"- Paired calibration items: `{summary.get('agreement', {}).get('n_paired', 0)}`",
        f"- Exact overall match rate: `{summary.get('agreement', {}).get('exact_overall_match_rate', float('nan')):.4f}`",
        f"- Mean absolute overall difference: `{summary.get('agreement', {}).get('mean_abs_overall_diff', float('nan')):.4f}`",
        f"- Mean absolute rubric difference: `{summary.get('agreement', {}).get('mean_abs_rubric_diff', float('nan')):.4f}`",
        "",
        "## Read",
        "",
        "Wave 2 expands the question-level exposure labeling pool with a non-overlapping stratified sample from the canonical 2025-backed design. It is intended to strengthen the training base and move the audited completed-call total beyond the four-digit threshold.",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    df = load_base()
    sample = build_sample(df)

    if BODY_FILE.exists():
        bodies = pd.read_csv(BODY_FILE)
    else:
        bodies = extract_bodies(sample)
        bodies.to_csv(BODY_FILE, index=False)

    sample = sample.merge(bodies, on="question_id", how="left")
    sample["body_excerpt"] = sample["body_excerpt"].fillna("")
    sample.to_csv(SAMPLE_FILE, index=False)

    previous_success_count = 0
    if RAW_RESPONSE_FILE.exists():
        with RAW_RESPONSE_FILE.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    obj = json.loads(line)
                    if obj.get("status") == "success":
                        previous_success_count += 1
                except Exception:
                    continue

    responses = run_calls(sample)
    success_count = int((responses["status"] == "success").sum())
    labels = build_label_frame(sample, responses)
    labels.to_csv(LABEL_FILE, index=False)

    summary = summarize(labels, success_count)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_md(summary)

    existing = 0
    if CALL_LEDGER.exists():
        with CALL_LEDGER.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    existing = max(existing, int(row["cumulative_total"]))
                except Exception:
                    continue
    increment = max(0, success_count - previous_success_count)
    cumulative = existing + increment
    append_ledger(
        increment,
        cumulative,
        f"Completed {summary['main_labels']} main API labels and {summary['calibration_labels']} calibration API labels in wave 2.",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
