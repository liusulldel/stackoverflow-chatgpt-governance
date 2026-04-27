from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import py7zr


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw" / "stackexchange_20251231"
PROCESSED_DIR = BASE_DIR / "processed"

ARCHIVE_7Z = RAW_DIR / "stackoverflow.com.7z"
EXTRACT_DIR = RAW_DIR / "stackoverflow.com_extracted"
MANIFEST_JSON = RAW_DIR / "stackexchange_20251231_manifest.json"
PROGRESS_JSON = RAW_DIR / "stackexchange_20251231_progress.json"

QUESTIONS_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_questions.parquet"
ANSWERS_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_answers.parquet"
ACCEPT_VOTES_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_accept_votes.parquet"
SUMMARY_JSON = PROCESSED_DIR / "stackexchange_20251231_focal_summary.json"

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

START_DATE = "2020-01-01"
END_EXCLUSIVE = "2026-01-01"
ACCEPT_VOTE_TYPE_ID = "1"


def ensure_dirs() -> None:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def emit(message: str) -> None:
    print(message, flush=True)


def save_progress(stage: str, **payload: object) -> None:
    progress = {
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    progress.update(payload)
    PROGRESS_JSON.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def find_member_path(archive: py7zr.SevenZipFile, target_name: str) -> str:
    for name in archive.getnames():
        if name.endswith(target_name):
            return name
    raise FileNotFoundError(f"{target_name} not found inside {ARCHIVE_7Z}")


def extract_required_files(force: bool = False) -> dict[str, str]:
    if not ARCHIVE_7Z.exists():
        raise FileNotFoundError(f"Archive not found: {ARCHIVE_7Z}")

    emit("Opening 2025Q4 archive and locating Posts.xml / Votes.xml members.")
    with py7zr.SevenZipFile(ARCHIVE_7Z, mode="r") as archive:
        posts_member = find_member_path(archive, "Posts.xml")
        votes_member = find_member_path(archive, "Votes.xml")
        extracted = {}
        for member in [posts_member, votes_member]:
            out_path = EXTRACT_DIR / Path(member).name
            if force and out_path.exists():
                out_path.unlink()
            if not out_path.exists():
                emit(f"Extracting {member} to {out_path}.")
                archive.extract(targets=[member], path=EXTRACT_DIR)
                nested_path = EXTRACT_DIR / member
                if nested_path.exists() and nested_path != out_path:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    nested_path.replace(out_path)
                    nested_parent = nested_path.parent
                    while nested_parent != EXTRACT_DIR and nested_parent.exists():
                        try:
                            nested_parent.rmdir()
                        except OSError:
                            break
                        nested_parent = nested_parent.parent
            extracted[Path(member).name] = str(out_path)
    save_progress("extracted_required_files", extracted_members=extracted)
    return extracted


def parse_tags(tags_raw: str | None) -> list[str]:
    if not tags_raw:
        return []
    stripped = tags_raw.strip("<>")
    if not stripped:
        return []
    return [tag for tag in stripped.split("><") if tag]


def question_in_scope(row: dict[str, str]) -> bool:
    if row.get("PostTypeId") != "1":
        return False
    created = row.get("CreationDate")
    if not created or created < START_DATE or created >= END_EXCLUSIVE:
        return False
    tags = parse_tags(row.get("Tags"))
    return any(tag in SELECTED_TAGS for tag in tags)


def parse_posts(posts_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    focal_questions: list[dict] = []
    focal_answers: list[dict] = []
    focal_question_ids: set[int] = set()
    focal_accepted_answer_ids: set[int] = set()

    emit(f"Starting single pass over {posts_path.name} for focal questions and answers.")
    save_progress("parsing_posts", posts_path=str(posts_path))
    context = ET.iterparse(posts_path, events=("end",))
    rows_seen = 0
    for _, elem in context:
        if elem.tag != "row":
            continue
        rows_seen += 1
        attrib = elem.attrib
        post_type = attrib.get("PostTypeId")
        if post_type == "1" and question_in_scope(attrib):
            question_id = int(attrib["Id"])
            accepted_answer_id = attrib.get("AcceptedAnswerId")
            tags = parse_tags(attrib.get("Tags"))
            selected_tags = [tag for tag in tags if tag in SELECTED_TAGS]
            focal_questions.append(
                {
                    "question_id": question_id,
                    "question_created_at": attrib.get("CreationDate"),
                    "accepted_answer_id": int(accepted_answer_id) if accepted_answer_id else None,
                    "score": int(attrib["Score"]) if attrib.get("Score") else None,
                    "view_count": int(attrib["ViewCount"]) if attrib.get("ViewCount") else None,
                    "answer_count": int(attrib["AnswerCount"]) if attrib.get("AnswerCount") else None,
                    "owner_user_id": int(attrib["OwnerUserId"]) if attrib.get("OwnerUserId") else None,
                    "title": attrib.get("Title"),
                    "question_tags": ";".join(tags),
                    "selected_tags": ";".join(selected_tags),
                    "selected_tag_overlap": len(selected_tags),
                }
            )
            focal_question_ids.add(question_id)
            if accepted_answer_id:
                focal_accepted_answer_ids.add(int(accepted_answer_id))
        elif post_type == "2":
            parent_id = attrib.get("ParentId")
            if parent_id:
                parent_id_int = int(parent_id)
                if parent_id_int in focal_question_ids:
                    answer_id = int(attrib["Id"])
                    focal_answers.append(
                        {
                            "answer_id": answer_id,
                            "question_id": parent_id_int,
                            "answer_created_at": attrib.get("CreationDate"),
                            "score": int(attrib["Score"]) if attrib.get("Score") else None,
                            "owner_user_id": int(attrib["OwnerUserId"]) if attrib.get("OwnerUserId") else None,
                            "is_current_accepted_answer": 1 if answer_id in focal_accepted_answer_ids else 0,
                        }
                    )
        if rows_seen % 2_000_000 == 0:
            emit(
                f"Posts pass progress: {rows_seen:,} rows seen; "
                f"{len(focal_questions):,} focal questions; "
                f"{len(focal_answers):,} focal answers; "
                f"{len(focal_accepted_answer_ids):,} accepted-answer ids."
            )
            save_progress(
                "parsing_posts",
                rows_seen=rows_seen,
                focal_questions=len(focal_questions),
                focal_answers=len(focal_answers),
                focal_accepted_answer_ids=len(focal_accepted_answer_ids),
            )
        elem.clear()

    questions_df = pd.DataFrame(focal_questions)
    answers_df = pd.DataFrame(focal_answers)
    emit(
        f"Completed Posts.xml pass: {len(questions_df):,} focal questions, "
        f"{len(answers_df):,} focal answers, and "
        f"{len(focal_accepted_answer_ids):,} accepted-answer ids retained."
    )
    return questions_df, answers_df


def parse_accept_votes(votes_path: Path, accepted_answer_ids: set[int]) -> pd.DataFrame:
    accept_votes: list[dict] = []
    emit(f"Starting Votes.xml pass for accepted-vote dates across {len(accepted_answer_ids):,} accepted answers.")
    save_progress(
        "parsing_accept_votes",
        votes_path=str(votes_path),
        accepted_answer_ids=len(accepted_answer_ids),
    )
    context = ET.iterparse(votes_path, events=("end",))
    rows_seen = 0
    for _, elem in context:
        if elem.tag != "row":
            continue
        rows_seen += 1
        attrib = elem.attrib
        if attrib.get("VoteTypeId") != ACCEPT_VOTE_TYPE_ID:
            elem.clear()
            continue
        post_id = attrib.get("PostId")
        if not post_id:
            elem.clear()
            continue
        post_id_int = int(post_id)
        if post_id_int not in accepted_answer_ids:
            elem.clear()
            continue
        accept_votes.append(
            {
                "vote_id": int(attrib["Id"]),
                "answer_id": post_id_int,
                "accept_vote_date": attrib.get("CreationDate"),
            }
        )
        if rows_seen % 2_000_000 == 0:
            emit(f"Votes pass progress: {rows_seen:,} rows seen; {len(accept_votes):,} accepted-vote rows.")
            save_progress(
                "parsing_accept_votes",
                rows_seen=rows_seen,
                accept_votes=len(accept_votes),
            )
        elem.clear()

    emit(f"Completed votes pass: {len(accept_votes):,} accepted-vote rows retained.")
    return pd.DataFrame(accept_votes)


def build_outputs(questions_df: pd.DataFrame, answers_df: pd.DataFrame, accept_votes_df: pd.DataFrame) -> dict:
    emit("Writing focal questions, answers, accepted-vote rows, and summary outputs.")
    questions_df.to_parquet(QUESTIONS_PARQUET, index=False)
    answers_df.to_parquet(ANSWERS_PARQUET, index=False)
    accept_votes_df.to_parquet(ACCEPT_VOTES_PARQUET, index=False)

    summary = {
        "questions_parquet": str(QUESTIONS_PARQUET),
        "answers_parquet": str(ANSWERS_PARQUET),
        "accept_votes_parquet": str(ACCEPT_VOTES_PARQUET),
        "n_questions": int(len(questions_df)),
        "n_answers": int(len(answers_df)),
        "n_accept_votes": int(len(accept_votes_df)),
        "min_question_created_at": questions_df["question_created_at"].min() if not questions_df.empty else None,
        "max_question_created_at": questions_df["question_created_at"].max() if not questions_df.empty else None,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_progress("outputs_written", **summary)
    return summary


def write_manifest(extracted: dict[str, str], summary: dict | None) -> None:
    manifest = {
        "archive_7z": str(ARCHIVE_7Z),
        "extract_dir": str(EXTRACT_DIR),
        "start_date": START_DATE,
        "end_exclusive": END_EXCLUSIVE,
        "selected_tags": SELECTED_TAGS,
        "accept_vote_type_id": ACCEPT_VOTE_TYPE_ID,
        "extracted_members": extracted,
        "summary": summary,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and parse Stack Overflow 2025Q4 dump for focal-tag extension.")
    parser.add_argument("--extract-only", action="store_true", help="Only extract Posts.xml and Votes.xml from the 7z archive.")
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of Posts.xml and Votes.xml.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    save_progress("starting", argv=sys.argv[1:])
    extracted = extract_required_files(force=args.force_extract)
    write_manifest(extracted, None)
    summary = None
    if not args.extract_only:
        posts_path = Path(extracted["Posts.xml"])
        votes_path = Path(extracted["Votes.xml"])
        questions_df, answers_df = parse_posts(posts_path)
        accepted_answer_ids = {
            int(answer_id)
            for answer_id in questions_df["accepted_answer_id"].dropna().astype(int).tolist()
        }
        accept_votes_df = parse_accept_votes(votes_path, accepted_answer_ids)
        summary = build_outputs(questions_df, answers_df, accept_votes_df)
        print(json.dumps(summary, indent=2))
    write_manifest(extracted, summary)


if __name__ == "__main__":
    main()
