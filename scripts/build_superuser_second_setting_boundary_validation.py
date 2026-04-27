from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

import score_question_level_exposure_index as exposure_model


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

SITE_DIR = RAW / "superuser_extract"
POSTS_XML = SITE_DIR / "Posts.xml"
VOTES_XML = SITE_DIR / "Votes.xml"

OUT_QUESTION = PROCESSED / "superuser_second_setting_question_level.parquet"
OUT_PANEL = PROCESSED / "superuser_second_setting_tag_month_panel.csv"
OUT_RESULTS = PROCESSED / "superuser_second_setting_results.csv"
OUT_SUMMARY = PROCESSED / "superuser_second_setting_summary.json"
OUT_MEMO = PAPER / "who_still_answers_superuser_second_setting_readout_2026-04-04.md"

START_DATE = "2020-01-01"
END_EXCLUSIVE = "2026-01-01"
SHOCK_DATE = pd.Timestamp("2022-11-30T00:00:00Z")
ACCEPT_VOTE_TYPE_ID = "1"
USER_AGENT = "CodexResearchBot/1.0 (liusully@gmail.com)"

SELECTED_TAGS = [
    "batch",
    "batch-file",
    "bash",
    "cmd.exe",
    "command-line",
    "drivers",
    "ethernet",
    "filesystems",
    "microsoft-excel",
    "partitioning",
    "permissions",
    "port-forwarding",
    "powershell",
    "proxy",
    "regex",
    "routing",
    "shell",
    "ssl",
    "wireless-networking",
    "worksheet-function",
]

TAG_RE = re.compile(r"<[^>]+>")
CODE_BLOCK_RE = re.compile(r"<pre(?:\\s[^>]*)?>(.*?)</pre>", flags=re.IGNORECASE | re.DOTALL)
INLINE_CODE_RE = re.compile(r"<code(?:\\s[^>]*)?>(.*?)</code>", flags=re.IGNORECASE | re.DOTALL)
ERROR_RE = re.compile(
    r"\\b(?:error|exception|traceback|stack trace|segmentation fault|segfault|syntaxerror|typeerror|valueerror|keyerror|nullpointer|undefined|failed|cannot|can't)\\b",
    flags=re.IGNORECASE,
)


def parse_tags(tags_raw: str | None) -> list[str]:
    if not tags_raw:
        return []
    stripped = tags_raw.strip("<>")
    if not stripped:
        return []
    return [tag for tag in stripped.split("><") if tag]


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    text = TAG_RE.sub(" ", html.unescape(value))
    return re.sub(r"\\s+", " ", text).strip()


def compute_complexity_metrics(
    body_html: str | None,
    comment_count_raw: str | None,
    tags_raw: str | None,
    last_edit_raw: str | None,
    title_raw: str | None,
) -> dict[str, float]:
    body_html = body_html or ""
    title_text = normalize_text(title_raw)
    body_text = normalize_text(body_html)
    code_blocks = CODE_BLOCK_RE.findall(body_html)
    inline_codes = INLINE_CODE_RE.findall(body_html)
    code_char_count = sum(len(normalize_text(block)) for block in code_blocks) + sum(
        len(normalize_text(code)) for code in inline_codes
    )
    body_word_count = len(body_text.split())
    error_keyword_count = len(ERROR_RE.findall(body_text))
    tags = parse_tags(tags_raw)
    return {
        "title_length_chars": len(title_text),
        "body_length_chars": len(body_text),
        "body_word_count": body_word_count,
        "code_block_count": len(code_blocks),
        "inline_code_count": len(inline_codes),
        "code_char_count": code_char_count,
        "error_keyword_count": error_keyword_count,
        "error_keyword_density": (error_keyword_count / body_word_count) if body_word_count > 0 else 0.0,
        "comment_count": int(comment_count_raw) if comment_count_raw else 0,
        "has_edit": 1 if last_edit_raw else 0,
        "tag_count_full": len(tags),
    }


def parse_questions_and_answers() -> tuple[pd.DataFrame, pd.DataFrame]:
    focal_questions: list[dict] = []
    focal_answers: list[dict] = []
    focal_question_ids: set[int] = set()

    context = ET.iterparse(POSTS_XML, events=("end",))
    for _, elem in context:
        if elem.tag != "row":
            continue
        attrib = elem.attrib
        post_type = attrib.get("PostTypeId")
        if post_type == "1":
            created = attrib.get("CreationDate")
            if not created or created < START_DATE or created >= END_EXCLUSIVE:
                elem.clear()
                continue
            all_tags = parse_tags(attrib.get("Tags"))
            selected = [tag for tag in all_tags if tag in SELECTED_TAGS]
            if not selected:
                elem.clear()
                continue
            qid = int(attrib["Id"])
            accepted_answer_id = int(attrib["AcceptedAnswerId"]) if attrib.get("AcceptedAnswerId") else None
            metrics = compute_complexity_metrics(
                attrib.get("Body"),
                attrib.get("CommentCount"),
                attrib.get("Tags"),
                attrib.get("LastEditDate"),
                attrib.get("Title"),
            )
            focal_questions.append(
                {
                    "question_id": qid,
                    "question_created_at": created,
                    "title": attrib.get("Title"),
                    "question_tags": ";".join(all_tags),
                    "selected_tags": ";".join(selected),
                    "selected_tag_overlap": len(selected),
                    "primary_tag": selected[0] if len(selected) == 1 else None,
                    "accepted_answer_id": accepted_answer_id,
                    **metrics,
                }
            )
            focal_question_ids.add(qid)
        elif post_type == "2":
            parent_id = attrib.get("ParentId")
            if parent_id and int(parent_id) in focal_question_ids:
                focal_answers.append(
                    {
                        "answer_id": int(attrib["Id"]),
                        "question_id": int(parent_id),
                        "answer_created_at": attrib.get("CreationDate"),
                    }
                )
        elem.clear()
    return pd.DataFrame(focal_questions), pd.DataFrame(focal_answers)


def parse_accept_votes(accepted_answer_ids: set[int]) -> pd.DataFrame:
    rows: list[dict] = []
    context = ET.iterparse(VOTES_XML, events=("end",))
    for _, elem in context:
        if elem.tag != "row":
            continue
        attrib = elem.attrib
        if attrib.get("VoteTypeId") != ACCEPT_VOTE_TYPE_ID:
            elem.clear()
            continue
        post_id = attrib.get("PostId")
        if post_id and int(post_id) in accepted_answer_ids:
            rows.append(
                {
                    "answer_id": int(post_id),
                    "accept_vote_date": attrib.get("CreationDate"),
                }
            )
        elem.clear()
    return pd.DataFrame(rows)


def build_exposure_scores(df: pd.DataFrame) -> pd.DataFrame:
    base = exposure_model.load_base()
    labels = exposure_model.load_labels()
    train_df = base.merge(labels, on=["question_id", "primary_tag", "post_chatgpt"], how="inner")
    train_df = train_df.drop_duplicates(subset=["question_id"]).copy()

    site_df = df.copy()
    site_df["question_created_at"] = pd.to_datetime(site_df["question_created_at"], utc=True)
    site_df["month_id"] = site_df["question_created_at"].dt.strftime("%Y-%m")
    site_df["post_chatgpt"] = (site_df["question_created_at"] >= SHOCK_DATE).astype(int)
    site_df["combined_text"] = (
        site_df["title"].fillna("").astype(str).str.strip()
        + " [TAGS] "
        + site_df["question_tags"].fillna("").astype(str).str.replace(";", " ", regex=False)
    )

    x_train, vectorizer, encoder = exposure_model.build_design(train_df)
    y_train = train_df[exposure_model.TARGET_COL].to_numpy(dtype=float)
    best_alpha, _ = exposure_model.select_alpha(x_train, y_train)
    model = exposure_model.Ridge(alpha=best_alpha, random_state=exposure_model.SEED)
    model.fit(x_train, y_train)

    score_cols = [
        "question_id",
        "primary_tag",
        "post_chatgpt",
        "title_length_chars",
        "body_length_chars",
        "body_word_count",
        "code_block_count",
        "inline_code_count",
        "code_char_count",
        "error_keyword_count",
        "error_keyword_density",
        "comment_count",
        "has_edit",
        "tag_count_full",
        "combined_text",
    ]
    x_site, _, _ = exposure_model.build_design(site_df[score_cols], vectorizer=vectorizer, encoder=encoder)
    site_df["predicted_exposure_score"] = model.predict(x_site)
    site_df["predicted_exposure_within_tag_z"] = site_df.groupby("primary_tag")[
        "predicted_exposure_score"
    ].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0))
    return site_df


def build_question_level() -> pd.DataFrame:
    questions, answers = parse_questions_and_answers()
    accepted_answer_ids = set(questions["accepted_answer_id"].dropna().astype(int).tolist())
    accept_votes = parse_accept_votes(accepted_answer_ids)

    questions = build_exposure_scores(questions)
    answers["answer_created_at"] = pd.to_datetime(answers["answer_created_at"], utc=True)
    if not accept_votes.empty:
        accept_votes["accept_vote_date"] = pd.to_datetime(accept_votes["accept_vote_date"], utc=True)
    else:
        accept_votes = pd.DataFrame(columns=["answer_id", "accept_vote_date"])

    first_answers = (
        answers.groupby("question_id", as_index=False)["answer_created_at"]
        .min()
        .rename(columns={"answer_created_at": "first_answer_at"})
    )
    accept_dates = pd.DataFrame(columns=["accepted_answer_id", "accept_vote_date"])
    if not accept_votes.empty:
        accept_dates = (
            accept_votes.groupby("answer_id", as_index=False)["accept_vote_date"]
            .min()
            .rename(columns={"answer_id": "accepted_answer_id"})
        )

    df = questions.merge(first_answers, on="question_id", how="left").merge(
        accept_dates, on="accepted_answer_id", how="left"
    )
    cutoff = max(
        pd.Timestamp(df["question_created_at"].max()),
        pd.Timestamp(answers["answer_created_at"].max()) if not answers.empty else pd.Timestamp(df["question_created_at"].max()),
        pd.Timestamp(accept_votes["accept_vote_date"].max()) if not accept_votes.empty else pd.Timestamp(df["question_created_at"].max()),
    )
    df["keep_single_focal"] = (df["selected_tag_overlap"] == 1).astype(int)

    for name, source_col, days in [
        ("first_answer_1d", "first_answer_at", 1),
        ("accepted_30d", "accept_vote_date", 30),
    ]:
        delta = (df[source_col] - df["question_created_at"]).dt.total_seconds() / 3600.0
        df[name] = ((delta >= 0) & (delta <= 24 * days)).fillna(False).astype(int)
        df[f"{name}_eligible"] = (df["question_created_at"] <= (cutoff - pd.Timedelta(days=days))).astype(int)

    pre_means = (
        df.loc[(df["question_created_at"] < SHOCK_DATE) & df["keep_single_focal"].eq(1)]
        .groupby("primary_tag", as_index=False)["predicted_exposure_score"]
        .mean()
        .rename(columns={"predicted_exposure_score": "pre_exposure_mean"})
    )
    pre_means["high_tag"] = (
        pre_means["pre_exposure_mean"].rank(method="first", ascending=False) <= (len(pre_means) / 2)
    ).astype(int)
    df = df.merge(pre_means, on="primary_tag", how="left")
    return df


def fetch_pageviews() -> pd.DataFrame:
    headers = {"User-Agent": USER_AGENT}
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        "en.wikipedia.org/all-access/all-agents/ChatGPT/monthly/20221101/20251231"
    )
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    items = response.json()["items"]
    views = pd.DataFrame(
        {"month_id": [f"{x['timestamp'][:4]}-{x['timestamp'][4:6]}" for x in items], "chatgpt_views": [x["views"] for x in items]}
    )
    full_months = pd.DataFrame({"month_id": pd.period_range("2020-01", "2025-12", freq="M").astype(str)})
    views = full_months.merge(views, on="month_id", how="left").fillna({"chatgpt_views": 0})
    views["chatgpt_z"] = (
        np.log1p(views["chatgpt_views"]) - np.log1p(views["chatgpt_views"]).mean()
    ) / np.log1p(views["chatgpt_views"]).std(ddof=0)
    return views


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    sample = df.loc[df["keep_single_focal"] == 1].copy()
    sample["month_id"] = sample["question_created_at"].dt.strftime("%Y-%m")
    rows = []
    for (tag, month_id), g in sample.groupby(["primary_tag", "month_id"], sort=True):
        def rate(event: str, elig: str) -> tuple[float, float]:
            denom = float(g[elig].sum())
            if denom <= 0:
                return np.nan, 0.0
            return float(g.loc[g[elig] == 1, event].sum() / denom), denom
        fa1, fa1_n = rate("first_answer_1d", "first_answer_1d_eligible")
        acc30, acc30_n = rate("accepted_30d", "accepted_30d_eligible")
        rows.append(
            {
                "primary_tag": tag,
                "month_id": month_id,
                "n_questions": int(len(g)),
                "time_index": int(pd.Period(month_id, freq="M").ordinal - pd.Period("2020-01", freq="M").ordinal + 1),
                "high_tag": int(g["high_tag"].iloc[0]),
                "exposure_index": float(g["pre_exposure_mean"].iloc[0]),
                "predicted_exposure_mean": float(g["predicted_exposure_score"].mean()),
                "residual_queue_complexity_index_mean": float(
                    pd.Series(g["body_word_count"]).rank(pct=True).mean()
                    + pd.Series(g["tag_count_full"]).rank(pct=True).mean()
                    + pd.Series(g["error_keyword_density"]).rank(pct=True).mean()
                )
                / 3.0,
                "first_answer_1d_rate": fa1,
                "first_answer_1d_denom": fa1_n,
                "accepted_30d_rate": acc30,
                "accepted_30d_denom": acc30_n,
                "response_cert_gap": fa1 - acc30 if pd.notna(fa1) and pd.notna(acc30) else np.nan,
            }
        )
    panel = pd.DataFrame(rows)
    panel["post_chatgpt"] = (pd.to_datetime(panel["month_id"] + "-01", utc=True) >= SHOCK_DATE).astype(int)
    panel["high_post"] = panel["high_tag"] * panel["post_chatgpt"]
    return panel


def clustered_result(formula: str, data: pd.DataFrame, weight_col: str, term: str) -> tuple[float, float, float]:
    model = smf.wls(formula, data=data, weights=data[weight_col]).fit()
    used = data.loc[model.model.data.row_labels]
    cov = cov_cluster_2groups(
        model,
        used["primary_tag"].astype("category").cat.codes,
        used["month_id"].astype("category").cat.codes,
    )[0]
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    idx = model.model.exog_names.index(term)
    coef = float(model.params.iloc[idx])
    se_term = float(se[idx])
    pval = float(2 * stats.norm.sf(abs(coef / se_term))) if se_term > 0 else np.nan
    return coef, se_term, pval


def main() -> None:
    question_level = build_question_level()
    question_level.to_parquet(OUT_QUESTION, index=False)
    panel = build_panel(question_level)
    panel = panel.merge(fetch_pageviews()[["month_id", "chatgpt_z"]], on="month_id", how="left")
    panel.to_csv(OUT_PANEL, index=False)

    specs = [
        (
            "predicted_exposure_mean",
            "predicted_exposure_mean ~ high_tag * post_chatgpt + C(primary_tag) + C(month_id)",
            "n_questions",
            "high_tag:post_chatgpt",
            "post_residualization",
        ),
        (
            "response_cert_gap",
            "response_cert_gap ~ high_tag * post_chatgpt + C(primary_tag) + C(month_id)",
            "first_answer_1d_denom",
            "high_tag:post_chatgpt",
            "post_gap",
        ),
        (
            "accepted_30d_rate",
            "accepted_30d_rate ~ high_tag * post_chatgpt + C(primary_tag) + C(month_id)",
            "accepted_30d_denom",
            "high_tag:post_chatgpt",
            "post_accepted30",
        ),
        (
            "response_cert_gap",
            "response_cert_gap ~ exposure_index:chatgpt_z + residual_queue_complexity_index_mean + C(primary_tag) + C(month_id) + C(primary_tag):time_index",
            "first_answer_1d_denom",
            "exposure_index:chatgpt_z",
            "external_gap",
        ),
        (
            "accepted_30d_rate",
            "accepted_30d_rate ~ exposure_index:chatgpt_z + residual_queue_complexity_index_mean + C(primary_tag) + C(month_id) + C(primary_tag):time_index",
            "accepted_30d_denom",
            "exposure_index:chatgpt_z",
            "external_accepted30",
        ),
        (
            "first_answer_1d_rate",
            "first_answer_1d_rate ~ exposure_index:chatgpt_z + residual_queue_complexity_index_mean + C(primary_tag) + C(month_id) + C(primary_tag):time_index",
            "first_answer_1d_denom",
            "exposure_index:chatgpt_z",
            "external_first1d",
        ),
    ]

    rows = []
    for outcome, formula, weight_col, term, spec_name in specs:
        data = panel[[outcome, weight_col, "high_tag", "post_chatgpt", "primary_tag", "month_id", "exposure_index", "chatgpt_z", "residual_queue_complexity_index_mean", "time_index"]].dropna().copy()
        coef, se, pval = clustered_result(formula, data, weight_col, term)
        rows.append(
            {
                "specification": spec_name,
                "outcome": outcome,
                "coef": coef,
                "cluster_se": se,
                "cluster_pval": pval,
                "nobs": len(data),
                "n_tags": data["primary_tag"].nunique(),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(OUT_RESULTS, index=False)

    summary = {
        "selected_tags": SELECTED_TAGS,
        "n_question_rows": int(len(question_level)),
        "n_single_focal_questions": int(question_level["keep_single_focal"].sum()),
        "n_panel_rows": int(len(panel)),
        "n_tags": int(panel["primary_tag"].nunique()),
        "results": results.to_dict(orient="records"),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    promoted = results.set_index("specification")
    memo = f"""# Super User Second-Setting Boundary Validation\n\nDate: April 4, 2026\n\n## Purpose\n\nThis second setting is a boundary-setting validation on Super User, chosen because it is a public technical troubleshooting platform with visible rapid response and accepted-answer certification.\n\n## Data\n\n- Archive: [superuser_download.7z](D:/AI alignment/projects/stackoverflow_chatgpt_governance/raw/superuser_download.7z)\n- Extracted inputs: [Posts.xml](D:/AI alignment/projects/stackoverflow_chatgpt_governance/raw/superuser_extract/Posts.xml), [Votes.xml](D:/AI alignment/projects/stackoverflow_chatgpt_governance/raw/superuser_extract/Votes.xml)\n- Question-level file: [superuser_second_setting_question_level.parquet](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/superuser_second_setting_question_level.parquet)\n- Tag-month panel: [superuser_second_setting_tag_month_panel.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/superuser_second_setting_tag_month_panel.csv)\n- Results: [superuser_second_setting_results.csv](D:/AI alignment/projects/stackoverflow_chatgpt_governance/processed/superuser_second_setting_results.csv)\n\n## Selected Tags\n\n{', '.join(SELECTED_TAGS)}\n\n## Key Estimates\n\n- `post_residualization`: `{promoted.loc['post_residualization', 'coef']:.4f}`, `p={promoted.loc['post_residualization', 'cluster_pval']:.4f}`\n- `post_gap`: `{promoted.loc['post_gap', 'coef']:.4f}`, `p={promoted.loc['post_gap', 'cluster_pval']:.4f}`\n- `post_accepted30`: `{promoted.loc['post_accepted30', 'coef']:.4f}`, `p={promoted.loc['post_accepted30', 'cluster_pval']:.4f}`\n- `external_gap`: `{promoted.loc['external_gap', 'coef']:.4f}`, `p={promoted.loc['external_gap', 'cluster_pval']:.4f}`\n- `external_accepted30`: `{promoted.loc['external_accepted30', 'coef']:.4f}`, `p={promoted.loc['external_accepted30', 'cluster_pval']:.4f}`\n- `external_first1d`: `{promoted.loc['external_first1d', 'coef']:.4f}`, `p={promoted.loc['external_first1d', 'cluster_pval']:.4f}`\n\n## Intended Use\n\nUse this second setting only if it sharpens the same core story as Stack Overflow: more exposed public troubleshooting domains retain a different residual queue and become more weakly coupled from rapid response to later certification in the generative-AI period.\n"""
    OUT_MEMO.write_text(memo, encoding="utf-8")


if __name__ == "__main__":
    main()
