from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import py7zr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract focal question, answer, and accept-vote rows from a Stack Exchange site dump.")
    parser.add_argument("--archive", required=True, help="Path to the site .7z archive.")
    parser.add_argument("--site-key", required=True, help="Short site key used in output filenames, e.g. dba_20251231.")
    parser.add_argument("--selected-tags-json", required=True, help="JSON array of selected tags.")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-exclusive", default="2026-01-01")
    parser.add_argument("--output-dir", default=None, help="Output directory for parquet/json files.")
    parser.add_argument("--extract-dir", default=None, help="Directory to extract Posts.xml / Votes.xml into.")
    parser.add_argument("--accept-vote-type-id", default="1")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def emit(message: str) -> None:
    print(message, flush=True)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_tags(tags_raw: str | None) -> list[str]:
    if not tags_raw:
        return []
    stripped = tags_raw.strip("<>")
    if not stripped:
        return []
    return [tag for tag in stripped.split("><") if tag]


def find_member_path(archive: py7zr.SevenZipFile, target_name: str) -> str:
    for name in archive.getnames():
        if name.endswith(target_name):
            return name
    raise FileNotFoundError(f"{target_name} not found in archive")


def extract_required_files(archive_path: Path, extract_dir: Path) -> dict[str, str]:
    ensure_dir(extract_dir)
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        posts_member = find_member_path(archive, "Posts.xml")
        votes_member = find_member_path(archive, "Votes.xml")
        extracted = {}
        for member in [posts_member, votes_member]:
            target = extract_dir / Path(member).name
            if not target.exists():
                emit(f"Extracting {member} -> {target}")
                archive.extract(targets=[member], path=extract_dir)
                nested = extract_dir / member
                if nested.exists() and nested != target:
                    nested.replace(target)
                    parent = nested.parent
                    while parent != extract_dir and parent.exists():
                        try:
                            parent.rmdir()
                        except OSError:
                            break
                        parent = parent.parent
            extracted[Path(member).name] = str(target)
    return extracted


def question_in_scope(row: dict[str, str], selected_tags: set[str], start_date: str, end_exclusive: str) -> bool:
    if row.get("PostTypeId") != "1":
        return False
    created = row.get("CreationDate")
    if not created or created < start_date or created >= end_exclusive:
        return False
    tags = parse_tags(row.get("Tags"))
    return any(tag in selected_tags for tag in tags)


def parse_posts(posts_path: Path, selected_tags: set[str], start_date: str, end_exclusive: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    focal_questions: list[dict] = []
    focal_answers: list[dict] = []
    focal_question_ids: set[int] = set()
    accepted_answer_ids: set[int] = set()
    rows_seen = 0
    for _, elem in ET.iterparse(posts_path, events=("end",)):
        if elem.tag != "row":
            continue
        rows_seen += 1
        attrib = elem.attrib
        post_type = attrib.get("PostTypeId")
        if post_type == "1" and question_in_scope(attrib, selected_tags, start_date, end_exclusive):
            qid = int(attrib["Id"])
            tags = parse_tags(attrib.get("Tags"))
            selected = [tag for tag in tags if tag in selected_tags]
            accepted_answer_id = attrib.get("AcceptedAnswerId")
            focal_questions.append(
                {
                    "question_id": qid,
                    "question_created_at": attrib.get("CreationDate"),
                    "accepted_answer_id": int(accepted_answer_id) if accepted_answer_id else None,
                    "score": int(attrib["Score"]) if attrib.get("Score") else None,
                    "view_count": int(attrib["ViewCount"]) if attrib.get("ViewCount") else None,
                    "answer_count": int(attrib["AnswerCount"]) if attrib.get("AnswerCount") else None,
                    "comment_count": int(attrib["CommentCount"]) if attrib.get("CommentCount") else 0,
                    "owner_user_id": int(attrib["OwnerUserId"]) if attrib.get("OwnerUserId") else None,
                    "title": attrib.get("Title"),
                    "body": attrib.get("Body"),
                    "question_tags": ";".join(tags),
                    "selected_tags": ";".join(selected),
                    "selected_tag_overlap": len(selected),
                    "last_edit_date": attrib.get("LastEditDate"),
                }
            )
            focal_question_ids.add(qid)
            if accepted_answer_id:
                accepted_answer_ids.add(int(accepted_answer_id))
        elif post_type == "2":
            parent_id = attrib.get("ParentId")
            if parent_id and int(parent_id) in focal_question_ids:
                aid = int(attrib["Id"])
                focal_answers.append(
                    {
                        "answer_id": aid,
                        "question_id": int(parent_id),
                        "answer_created_at": attrib.get("CreationDate"),
                        "score": int(attrib["Score"]) if attrib.get("Score") else None,
                        "owner_user_id": int(attrib["OwnerUserId"]) if attrib.get("OwnerUserId") else None,
                        "is_current_accepted_answer": 1 if aid in accepted_answer_ids else 0,
                    }
                )
        if rows_seen % 1_000_000 == 0:
            emit(
                f"Posts progress: {rows_seen:,} rows, "
                f"{len(focal_questions):,} focal questions, {len(focal_answers):,} focal answers."
            )
        elem.clear()
    return pd.DataFrame(focal_questions), pd.DataFrame(focal_answers)


def parse_accept_votes(votes_path: Path, accepted_answer_ids: set[int], vote_type_id: str) -> pd.DataFrame:
    accept_votes: list[dict] = []
    rows_seen = 0
    for _, elem in ET.iterparse(votes_path, events=("end",)):
        if elem.tag != "row":
            continue
        rows_seen += 1
        attrib = elem.attrib
        if attrib.get("VoteTypeId") != vote_type_id:
            elem.clear()
            continue
        post_id = attrib.get("PostId")
        if post_id and int(post_id) in accepted_answer_ids:
            accept_votes.append(
                {
                    "vote_id": int(attrib["Id"]),
                    "answer_id": int(post_id),
                    "accept_vote_date": attrib.get("CreationDate"),
                }
            )
        if rows_seen % 1_000_000 == 0:
            emit(f"Votes progress: {rows_seen:,} rows, {len(accept_votes):,} accepted votes.")
        elem.clear()
    return pd.DataFrame(accept_votes)


def main() -> None:
    args = parse_args()
    archive_path = Path(args.archive)
    base_output = Path(args.output_dir) if args.output_dir else archive_path.resolve().parents[1] / "processed"
    extract_dir = Path(args.extract_dir) if args.extract_dir else archive_path.with_suffix("")
    ensure_dir(base_output)
    ensure_dir(extract_dir)

    selected_tags = set(json.loads(args.selected_tags_json))
    extracted = extract_required_files(archive_path, extract_dir)

    questions_df, answers_df = parse_posts(
        Path(extracted["Posts.xml"]), selected_tags, args.start_date, args.end_exclusive
    )
    accepted_ids = set(questions_df["accepted_answer_id"].dropna().astype(int).tolist())
    accept_votes_df = parse_accept_votes(Path(extracted["Votes.xml"]), accepted_ids, args.accept_vote_type_id)

    stem = args.site_key
    q_path = base_output / f"{stem}_focal_questions.parquet"
    a_path = base_output / f"{stem}_focal_answers.parquet"
    v_path = base_output / f"{stem}_focal_accept_votes.parquet"
    s_path = base_output / f"{stem}_focal_summary.json"

    questions_df.to_parquet(q_path, index=False)
    answers_df.to_parquet(a_path, index=False)
    accept_votes_df.to_parquet(v_path, index=False)

    summary = {
        "archive": str(archive_path),
        "site_key": stem,
        "selected_tags": sorted(selected_tags),
        "start_date": args.start_date,
        "end_exclusive": args.end_exclusive,
        "n_questions": int(len(questions_df)),
        "n_answers": int(len(answers_df)),
        "n_accept_votes": int(len(accept_votes_df)),
        "min_question_created_at": questions_df["question_created_at"].min() if not questions_df.empty else None,
        "max_question_created_at": questions_df["question_created_at"].max() if not questions_df.empty else None,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "questions_parquet": str(q_path),
        "answers_parquet": str(a_path),
        "accept_votes_parquet": str(v_path),
    }
    save_json(s_path, summary)
    emit(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
