import json
import os

import duckdb


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "raw")

SOURCE_POSTS = os.path.join(RAW_DIR, "stackoverflow_2023_05_posts.parquet")
SOURCE_TAGS = os.path.join(RAW_DIR, "stackoverflow_2023_05_tags.parquet")
OUTPUT_CSV = os.path.join(RAW_DIR, "stackoverflow_large_design_questions_raw.csv")
OUTPUT_PARQUET = os.path.join(RAW_DIR, "stackoverflow_large_design_questions_raw.parquet")
OUTPUT_METADATA = os.path.join(RAW_DIR, "stackoverflow_large_design_tag_metadata.json")

START_DATE = "2022-01-01"
END_DATE_EXCLUSIVE = "2023-06-01"

SELECTED_TAGS = [
    "apache-spark",
    "android",
    "bash",
    "docker",
    "excel",
    "firebase",
    "kubernetes",
    "linux",
    "memory-management",
    "multithreading",
    "numpy",
    "pandas",
    "python",
    "regex",
    "javascript",
    "sql",
]


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)


def escape_tag(tag: str) -> str:
    return tag.replace("'", "''")


def build_condition(tags):
    return " OR ".join([f"Tags LIKE '%<{escape_tag(tag)}>%' " for tag in tags])


def main():
    ensure_dirs()
    con = duckdb.connect()
    condition = build_condition(SELECTED_TAGS)

    base_query = f"""
        SELECT
            Id AS question_id,
            strftime(CreationDate, '%Y-%m-%dT%H:%M:%SZ') AS created_at_iso,
            Score AS score,
            ViewCount AS view_count,
            AnswerCount AS answer_count,
            CASE WHEN AcceptedAnswerId IS NULL THEN 0 ELSE 1 END AS accepted,
            OwnerUserId AS owner_user_id,
            Title AS title,
            left(regexp_replace(coalesce(Body, ''), '<[^>]+>', ' ', 'g'), 1500) AS body_text,
            regexp_replace(regexp_replace(Tags, '^<|>$', '', 'g'), '><', ';', 'g') AS question_tags
        FROM read_parquet('{SOURCE_POSTS}')
        WHERE PostTypeId = 1
          AND CreationDate >= TIMESTAMP '{START_DATE}'
          AND CreationDate < TIMESTAMP '{END_DATE_EXCLUSIVE}'
          AND ({condition})
    """

    con.execute(
        f"COPY ({base_query}) TO '{OUTPUT_CSV}' (HEADER, DELIMITER ',')"
    )
    con.execute(
        f"COPY ({base_query}) TO '{OUTPUT_PARQUET}' (FORMAT PARQUET)"
    )

    count_query = f"""
        SELECT
            TagName AS tag,
            Count AS all_time_tag_count
        FROM read_parquet('{SOURCE_TAGS}')
        WHERE TagName IN ({", ".join([f"'{escape_tag(tag)}'" for tag in SELECTED_TAGS])})
        ORDER BY TagName
    """
    tag_rows = con.execute(count_query).fetchdf().to_dict(orient="records")
    sample_rows = con.execute(f"SELECT COUNT(*) AS n_rows FROM ({base_query}) q").fetchone()[0]

    with open(OUTPUT_METADATA, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_posts": SOURCE_POSTS,
                "source_tags": SOURCE_TAGS,
                "selected_tags": SELECTED_TAGS,
                "start_date": START_DATE,
                "end_date_exclusive": END_DATE_EXCLUSIVE,
                "sample_rows": int(sample_rows),
                "tag_rows": tag_rows,
            },
            handle,
            indent=2,
        )

    print(
        json.dumps(
            {
                "output_csv": OUTPUT_CSV,
                "output_parquet": OUTPUT_PARQUET,
                "sample_rows": int(sample_rows),
                "selected_tags": len(SELECTED_TAGS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
