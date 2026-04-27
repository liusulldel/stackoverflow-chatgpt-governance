from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(r"D:\AI alignment\projects\stackoverflow_chatgpt_governance")
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

RAW_ZIP = PROCESSED / "who_still_answers_devgpt_raw.zip"
THREAD_PANEL_PARQUET = PROCESSED / "who_still_answers_devgpt_sidecar_thread_panel.parquet"
PR_PANEL_PARQUET = PROCESSED / "who_still_answers_devgpt_sidecar_pr_panel.parquet"
RESULTS_CSV = PROCESSED / "who_still_answers_devgpt_sidecar_results.csv"
SUMMARY_JSON = PROCESSED / "who_still_answers_devgpt_sidecar_summary.json"
READOUT_PATH = PAPER / "who_still_answers_devgpt_sidecar_readout_2026-04-05.md"


ERROR_PATTERN = re.compile(r"\berror\b|\bexception\b|traceback|stack\s*trace|\bbug\b|\bfail(?:ed|ing)?\b", re.IGNORECASE)
ENV_PATTERN = re.compile(
    r"\bversion\b|\bpython\b|\bnode(?:js)?\b|\bjava\b|\bubuntu\b|\blinux\b|\bwindows\b|docker|kubernetes|pip|npm|conda",
    re.IGNORECASE,
)
STATE_PATTERN = re.compile(
    r"\bconfig\b|\bconfiguration\b|\benvironment\b|\bsetup\b|\binstall\b|\bdependency\b|requirements\.txt|package\.json|pom\.xml",
    re.IGNORECASE,
)
CODE_PATTERN = re.compile(r"`{3}|</?[a-zA-Z]+>|::|=>|==|!=|\bSELECT\b|\bfunction\b|\bclass\b", re.IGNORECASE)


def _snapshot_rank(snapshot_name: str) -> int:
    digits = "".join(ch for ch in snapshot_name if ch.isdigit())
    if not digits:
        return -1
    return int(digits)


def _safe_ts(value: Any) -> pd.Timestamp | pd.NaT:
    if value in (None, "", "None"):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce", utc=True)


def _safe_len(text: Any) -> int:
    if not isinstance(text, str):
        return 0
    return len(text)


@dataclass
class ChatFeatures:
    n_chat_links: int
    n_prompts: int
    n_turns: int
    prompt_chars: int
    answer_chars: int
    prompt_tokens: float
    answer_tokens: float
    n_models: int
    n_code_blocks: int
    context_error_hits: int
    context_env_hits: int
    context_state_hits: int
    context_code_hits: int


def _extract_chat_features(chat_list: Any) -> ChatFeatures:
    if not isinstance(chat_list, list):
        chat_list = []

    n_chat_links = len(chat_list)
    n_prompts = 0
    n_turns = 0
    prompt_chars = 0
    answer_chars = 0
    prompt_tokens = 0.0
    answer_tokens = 0.0
    models: set[str] = set()
    n_code_blocks = 0
    context_error_hits = 0
    context_env_hits = 0
    context_state_hits = 0
    context_code_hits = 0

    for share in chat_list:
        if not isinstance(share, dict):
            continue
        model = share.get("Model")
        if isinstance(model, str) and model.strip():
            models.add(model.strip())

        share_prompts = share.get("NumberOfPrompts")
        if isinstance(share_prompts, (int, float)) and math.isfinite(float(share_prompts)):
            n_prompts += int(share_prompts)

        p_tok = share.get("TokensOfPrompts")
        a_tok = share.get("TokensOfAnswers")
        if isinstance(p_tok, (int, float)) and math.isfinite(float(p_tok)):
            prompt_tokens += float(p_tok)
        if isinstance(a_tok, (int, float)) and math.isfinite(float(a_tok)):
            answer_tokens += float(a_tok)

        conversations = share.get("Conversations")
        if not isinstance(conversations, list):
            conversations = []

        for conv in conversations:
            if not isinstance(conv, dict):
                continue
            n_turns += 1
            prompt = conv.get("Prompt")
            answer = conv.get("Answer")
            prompt_chars += _safe_len(prompt)
            answer_chars += _safe_len(answer)

            prompt_text = prompt if isinstance(prompt, str) else ""
            if ERROR_PATTERN.search(prompt_text):
                context_error_hits += 1
            if ENV_PATTERN.search(prompt_text):
                context_env_hits += 1
            if STATE_PATTERN.search(prompt_text):
                context_state_hits += 1
            if CODE_PATTERN.search(prompt_text):
                context_code_hits += 1

            loc = conv.get("ListOfCode")
            if isinstance(loc, list):
                n_code_blocks += len(loc)

    return ChatFeatures(
        n_chat_links=n_chat_links,
        n_prompts=n_prompts,
        n_turns=n_turns,
        prompt_chars=prompt_chars,
        answer_chars=answer_chars,
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        n_models=len(models),
        n_code_blocks=n_code_blocks,
        context_error_hits=context_error_hits,
        context_env_hits=context_env_hits,
        context_state_hits=context_state_hits,
        context_code_hits=context_code_hits,
    )


def _canon_thread_type(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else ""
    if text in {"issue", "discussion", "pull request", "pr"}:
        return text.replace("pull request", "pr")
    return text or "unknown"


def _bucket_language(series: pd.Series, top_n: int = 12) -> pd.Series:
    freq = series.fillna("Unknown").value_counts()
    keep = set(freq.head(top_n).index)
    return series.fillna("Unknown").map(lambda x: x if x in keep else "Other")


def _load_json_from_zip(zip_file: zipfile.ZipFile, path: str) -> list[dict[str, Any]]:
    with zip_file.open(path) as f:
        payload = json.load(f)
    sources = payload.get("Sources", [])
    if not isinstance(sources, list):
        return []
    return [row for row in sources if isinstance(row, dict)]


def load_devgpt_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RAW_ZIP.exists():
        raise FileNotFoundError(f"DevGPT zip not found: {RAW_ZIP}")

    thread_rows: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(RAW_ZIP) as zf:
        names = zf.namelist()
        issue_files = sorted(n for n in names if n.endswith("issue_sharings.json"))
        discussion_files = sorted(n for n in names if n.endswith("discussion_sharings.json"))
        pr_files = sorted(n for n in names if n.endswith("pr_sharings.json"))

        for path in issue_files + discussion_files:
            snapshot = path.split("/")[0]
            for row in _load_json_from_zip(zf, path):
                f = _extract_chat_features(row.get("ChatgptSharing"))
                thread_rows.append(
                    {
                        "snapshot": snapshot,
                        "snapshot_rank": _snapshot_rank(snapshot),
                        "source_file": path,
                        "thread_type": _canon_thread_type(row.get("Type")),
                        "url": row.get("URL"),
                        "repo_name": row.get("RepoName"),
                        "repo_language": row.get("RepoLanguage"),
                        "number": row.get("Number"),
                        "author": row.get("Author"),
                        "state": row.get("State"),
                        "closed_flag_raw": row.get("Closed"),
                        "created_at": _safe_ts(row.get("CreatedAt")),
                        "updated_at": _safe_ts(row.get("UpdatedAt")),
                        "closed_at": _safe_ts(row.get("ClosedAt")),
                        "title": row.get("Title"),
                        "body": row.get("Body"),
                        "upvote_count": row.get("UpvoteCount"),
                        "n_chat_links": f.n_chat_links,
                        "n_prompts": f.n_prompts,
                        "n_turns": f.n_turns,
                        "prompt_chars": f.prompt_chars,
                        "answer_chars": f.answer_chars,
                        "prompt_tokens": f.prompt_tokens,
                        "answer_tokens": f.answer_tokens,
                        "n_models": f.n_models,
                        "n_code_blocks": f.n_code_blocks,
                        "context_error_hits": f.context_error_hits,
                        "context_env_hits": f.context_env_hits,
                        "context_state_hits": f.context_state_hits,
                        "context_code_hits": f.context_code_hits,
                    }
                )

        for path in pr_files:
            snapshot = path.split("/")[0]
            for row in _load_json_from_zip(zf, path):
                f = _extract_chat_features(row.get("ChatgptSharing"))
                pr_rows.append(
                    {
                        "snapshot": snapshot,
                        "snapshot_rank": _snapshot_rank(snapshot),
                        "source_file": path,
                        "url": row.get("URL"),
                        "repo_name": row.get("RepoName"),
                        "repo_language": row.get("RepoLanguage"),
                        "number": row.get("Number"),
                        "state": row.get("State"),
                        "created_at": _safe_ts(row.get("CreatedAt")),
                        "updated_at": _safe_ts(row.get("UpdatedAt")),
                        "closed_at": _safe_ts(row.get("ClosedAt")),
                        "merged_at": _safe_ts(row.get("MergedAt")),
                        "title": row.get("Title"),
                        "body": row.get("Body"),
                        "additions": row.get("Additions"),
                        "deletions": row.get("Deletions"),
                        "changed_files": row.get("ChangedFiles"),
                        "n_chat_links": f.n_chat_links,
                        "n_prompts": f.n_prompts,
                        "n_turns": f.n_turns,
                        "prompt_chars": f.prompt_chars,
                        "answer_chars": f.answer_chars,
                        "prompt_tokens": f.prompt_tokens,
                        "answer_tokens": f.answer_tokens,
                        "n_models": f.n_models,
                        "n_code_blocks": f.n_code_blocks,
                        "context_error_hits": f.context_error_hits,
                        "context_env_hits": f.context_env_hits,
                        "context_state_hits": f.context_state_hits,
                        "context_code_hits": f.context_code_hits,
                    }
                )

    thread_df = pd.DataFrame(thread_rows)
    pr_df = pd.DataFrame(pr_rows)
    return thread_df, pr_df


def _dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.sort_values(["url", "snapshot_rank", "updated_at"], ascending=[True, False, False])
        .drop_duplicates(subset=["url"], keep="first")
        .reset_index(drop=True)
    )


def prepare_thread_panel(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    panel = _dedupe_latest(df).copy()
    panel["created_month"] = panel["created_at"].dt.to_period("M").astype(str)

    panel["title_len"] = panel["title"].map(_safe_len)
    panel["body_len"] = panel["body"].map(_safe_len)
    panel["log_title_len"] = np.log1p(panel["title_len"])
    panel["log_body_len"] = np.log1p(panel["body_len"])

    panel["total_ai_tokens"] = panel["prompt_tokens"].fillna(0.0) + panel["answer_tokens"].fillna(0.0)
    panel["log_ai_tokens"] = np.log1p(panel["total_ai_tokens"])
    panel["total_context_hits"] = (
        panel["context_error_hits"].fillna(0)
        + panel["context_env_hits"].fillna(0)
        + panel["context_state_hits"].fillna(0)
        + panel["context_code_hits"].fillna(0)
    )
    panel["context_density"] = panel["total_context_hits"] / panel["n_turns"].clip(lower=1)
    panel["context_heavy"] = (panel["context_density"] >= panel["context_density"].median()).astype(int)
    panel["multi_turn"] = (panel["n_prompts"] >= 4).astype(int)
    panel["multi_model"] = (panel["n_models"] >= 2).astype(int)

    close_hours = (panel["closed_at"] - panel["created_at"]).dt.total_seconds() / 3600.0
    update_hours = (panel["updated_at"] - panel["created_at"]).dt.total_seconds() / 3600.0
    panel["close_hours"] = close_hours
    panel["first_update_hours"] = update_hours

    panel["updated_7d"] = update_hours.between(0, 24 * 7, inclusive="both").fillna(False).astype(int)
    panel["closed_30d"] = close_hours.between(0, 24 * 30, inclusive="both").fillna(False).astype(int)
    panel["closed_90d"] = close_hours.between(0, 24 * 90, inclusive="both").fillna(False).astype(int)
    panel["ever_closed"] = (
        panel["closed_at"].notna()
        | panel["state"].astype(str).str.upper().eq("CLOSED")
        | panel["closed_flag_raw"].astype(str).str.lower().eq("true")
    ).astype(int)
    panel["response_close_gap_30d"] = panel["updated_7d"] - panel["closed_30d"]
    panel["closed_30d_given_updated"] = np.where(panel["updated_7d"] == 1, panel["closed_30d"], np.nan)

    panel["thread_kind"] = panel["thread_type"].map(
        {
            "issue": "issue",
            "discussion": "discussion",
        }
    ).fillna("other")
    panel["repo_language_bucket"] = _bucket_language(panel["repo_language"])
    return panel


def prepare_pr_panel(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    panel = _dedupe_latest(df).copy()
    panel["created_month"] = panel["created_at"].dt.to_period("M").astype(str)
    panel["title_len"] = panel["title"].map(_safe_len)
    panel["body_len"] = panel["body"].map(_safe_len)
    panel["log_title_len"] = np.log1p(panel["title_len"])
    panel["log_body_len"] = np.log1p(panel["body_len"])
    panel["total_ai_tokens"] = panel["prompt_tokens"].fillna(0.0) + panel["answer_tokens"].fillna(0.0)
    panel["log_ai_tokens"] = np.log1p(panel["total_ai_tokens"])
    panel["total_context_hits"] = (
        panel["context_error_hits"].fillna(0)
        + panel["context_env_hits"].fillna(0)
        + panel["context_state_hits"].fillna(0)
        + panel["context_code_hits"].fillna(0)
    )
    panel["context_density"] = panel["total_context_hits"] / panel["n_turns"].clip(lower=1)
    panel["context_heavy"] = (panel["context_density"] >= panel["context_density"].median()).astype(int)
    panel["multi_turn"] = (panel["n_prompts"] >= 4).astype(int)
    merge_hours = (panel["merged_at"] - panel["created_at"]).dt.total_seconds() / 3600.0
    panel["merged_hours"] = merge_hours
    panel["merged_30d"] = merge_hours.between(0, 24 * 30, inclusive="both").fillna(False).astype(int)
    panel["merged_7d"] = merge_hours.between(0, 24 * 7, inclusive="both").fillna(False).astype(int)
    panel["repo_language_bucket"] = _bucket_language(panel["repo_language"])
    return panel


def _run_cluster_lpm(
    df: pd.DataFrame,
    outcome: str,
    sample_name: str,
    setting: str,
    condition: str = "",
) -> list[dict[str, Any]]:
    frame = df.copy()
    if condition:
        frame = frame.query(condition).copy()
    frame = frame.dropna(
        subset=[outcome, "context_heavy", "multi_turn", "log_ai_tokens", "log_title_len", "log_body_len", "repo_name"]
    )
    if frame.empty or frame["repo_name"].nunique() < 20:
        return []

    formula = (
        f"{outcome} ~ context_heavy + multi_turn + log_ai_tokens + "
        "log_title_len + log_body_len + C(repo_language_bucket) + C(created_month)"
    )
    model = smf.ols(formula=formula, data=frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame["repo_name"]}
    )

    rows = []
    for var in ["context_heavy", "multi_turn", "log_ai_tokens"]:
        if var not in model.params.index:
            continue
        rows.append(
            {
                "setting": setting,
                "sample": sample_name,
                "outcome": outcome,
                "predictor": var,
                "coef": float(model.params[var]),
                "se": float(model.bse[var]),
                "pval": float(model.pvalues[var]),
                "nobs": int(model.nobs),
                "n_repos": int(frame["repo_name"].nunique()),
                "n_months": int(frame["created_month"].nunique()),
            }
        )
    return rows


def fit_results(thread_panel: pd.DataFrame, pr_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not thread_panel.empty:
        rows += _run_cluster_lpm(
            thread_panel.loc[thread_panel["thread_kind"].isin(["issue", "discussion"])].copy(),
            outcome="updated_7d",
            sample_name="issue_discussion_all",
            setting="devgpt_thread",
        )
        rows += _run_cluster_lpm(
            thread_panel.loc[thread_panel["thread_kind"].isin(["issue", "discussion"])].copy(),
            outcome="closed_30d",
            sample_name="issue_discussion_all",
            setting="devgpt_thread",
        )
        rows += _run_cluster_lpm(
            thread_panel.loc[thread_panel["thread_kind"].isin(["issue", "discussion"])].copy(),
            outcome="response_close_gap_30d",
            sample_name="issue_discussion_all",
            setting="devgpt_thread",
        )
        rows += _run_cluster_lpm(
            thread_panel.loc[thread_panel["thread_kind"].eq("issue")].copy(),
            outcome="closed_30d_given_updated",
            sample_name="issue_only_updated",
            setting="devgpt_thread",
            condition="updated_7d == 1",
        )

    if not pr_panel.empty:
        rows += _run_cluster_lpm(
            pr_panel.copy(),
            outcome="merged_30d",
            sample_name="pr_all",
            setting="devgpt_pr_support",
        )
        rows += _run_cluster_lpm(
            pr_panel.copy(),
            outcome="merged_7d",
            sample_name="pr_all",
            setting="devgpt_pr_support",
        )

    return pd.DataFrame(rows)


def _build_topline_summary(thread_panel: pd.DataFrame, pr_panel: pd.DataFrame, results: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "data_source": str(RAW_ZIP),
        "thread_sample_n": int(len(thread_panel)),
        "thread_issue_n": int(thread_panel["thread_kind"].eq("issue").sum()) if not thread_panel.empty else 0,
        "thread_discussion_n": int(thread_panel["thread_kind"].eq("discussion").sum()) if not thread_panel.empty else 0,
        "thread_repo_n": int(thread_panel["repo_name"].nunique()) if not thread_panel.empty else 0,
        "pr_sample_n": int(len(pr_panel)),
        "pr_repo_n": int(pr_panel["repo_name"].nunique()) if not pr_panel.empty else 0,
    }

    if not thread_panel.empty:
        summary.update(
            {
                "thread_updated_7d_mean": float(thread_panel["updated_7d"].mean()),
                "thread_closed_30d_mean": float(thread_panel["closed_30d"].mean()),
                "thread_closed_90d_mean": float(thread_panel["closed_90d"].mean()),
                "thread_gap_mean": float(thread_panel["response_close_gap_30d"].mean()),
                "thread_context_heavy_share": float(thread_panel["context_heavy"].mean()),
                "thread_median_ai_tokens": float(thread_panel["total_ai_tokens"].median()),
            }
        )
        context_stats = (
            thread_panel.groupby("context_heavy", as_index=False)
            .agg(
                n=("url", "size"),
                updated_7d=("updated_7d", "mean"),
                closed_30d=("closed_30d", "mean"),
                closed_90d=("closed_90d", "mean"),
                response_close_gap_30d=("response_close_gap_30d", "mean"),
            )
            .to_dict("records")
        )
        summary["thread_context_split"] = context_stats

    if not pr_panel.empty:
        summary.update(
            {
                "pr_merged_30d_mean": float(pr_panel["merged_30d"].mean()),
                "pr_merged_7d_mean": float(pr_panel["merged_7d"].mean()),
            }
        )

    if not results.empty:
        def pick(setting: str, sample: str, outcome: str, predictor: str) -> dict[str, Any] | None:
            subset = results.loc[
                (results["setting"] == setting)
                & (results["sample"] == sample)
                & (results["outcome"] == outcome)
                & (results["predictor"] == predictor)
            ]
            if subset.empty:
                return None
            return subset.iloc[0].to_dict()

        summary["headline_coefficients"] = {
            "thread_closed_30d_context_heavy": pick(
                "devgpt_thread", "issue_discussion_all", "closed_30d", "context_heavy"
            ),
            "thread_gap_context_heavy": pick(
                "devgpt_thread", "issue_discussion_all", "response_close_gap_30d", "context_heavy"
            ),
            "thread_closed_given_updated_context_heavy": pick(
                "devgpt_thread", "issue_only_updated", "closed_30d_given_updated", "context_heavy"
            ),
            "pr_merged_30d_context_heavy": pick(
                "devgpt_pr_support", "pr_all", "merged_30d", "context_heavy"
            ),
        }
    return summary


def write_readout(summary: dict[str, Any], results: pd.DataFrame) -> None:
    lines = [
        "# Who Still Answers DevGPT Sidecar Readout",
        "",
        "Date: 2026-04-05",
        "",
        "## Feasibility Verdict",
        "",
        "Feasible. DevGPT is locally downloadable and includes issue/discussion threads with direct ChatGPT-sharing traces.",
        "This makes it more question-like than AIDev PR-only data for a sidecar focused on public response and closure/certification logic.",
        "",
        "## Data Built",
        "",
        f"- Thread panel (issues + discussions, deduped to latest snapshot): `{summary.get('thread_sample_n', 0):,}` rows",
        f"- Issue rows: `{summary.get('thread_issue_n', 0):,}`; discussion rows: `{summary.get('thread_discussion_n', 0):,}`",
        f"- PR support panel (deduped): `{summary.get('pr_sample_n', 0):,}` rows",
        "",
        "## Sidecar Logic",
        "",
        "Question-like layer: `updated_7d` (early public response proxy) vs `closed_30d` (formalized closure).",
        "Divergence metric: `response_close_gap_30d = updated_7d - closed_30d`.",
        "Direct-AI feature layer: prompt/answer tokens, multi-turn chat, and context-heaviness parsed from shared prompts.",
        "",
    ]

    if "thread_updated_7d_mean" in summary:
        lines.extend(
            [
                "## Topline Rates",
                "",
                f"- Thread updated within 7d: `{summary['thread_updated_7d_mean']:.3f}`",
                f"- Thread closed within 30d: `{summary['thread_closed_30d_mean']:.3f}`",
                f"- Thread closed within 90d: `{summary['thread_closed_90d_mean']:.3f}`",
                f"- Mean response-close gap (30d): `{summary['thread_gap_mean']:.3f}`",
                "",
            ]
        )

    if summary.get("thread_context_split"):
        lines.append("## Context-Heavy Split (Descriptive)")
        lines.append("")
        for row in summary["thread_context_split"]:
            label = "high-context" if int(row["context_heavy"]) == 1 else "low-context"
            lines.append(
                f"- {label}: n=`{int(row['n'])}`, updated_7d=`{row['updated_7d']:.3f}`, "
                f"closed_30d=`{row['closed_30d']:.3f}`, gap=`{row['response_close_gap_30d']:.3f}`"
            )
        lines.append("")

    lines.extend(["## Clustered LPM Results", ""])
    if results.empty:
        lines.append("No estimable models were produced.")
    else:
        for _, row in results.iterrows():
            lines.append(
                f"- `{row['setting']}` / `{row['sample']}` / `{row['outcome']}` / `{row['predictor']}`: "
                f"coef `{row['coef']:.4f}`, clustered `p = {row['pval']:.4f}` "
                f"(nobs `{int(row['nobs']):,}`, repos `{int(row['n_repos'])}`)"
            )

    lines.extend(
        [
            "",
            "## What This Adds (and What It Does Not)",
            "",
            "Adds: a direct-AI-use sidecar with question-like threads and an explicit response-versus-closure decomposition.",
            "Does not add: complete direct observation of all AI use on GitHub (still a selective self-sharing sample).",
            "Therefore this is a mechanism-validating sidecar, not a standalone causal replacement for the main Stack Overflow design.",
        ]
    )

    READOUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    thread_raw, pr_raw = load_devgpt_records()
    thread_panel = prepare_thread_panel(thread_raw)
    pr_panel = prepare_pr_panel(pr_raw)
    results = fit_results(thread_panel, pr_panel)
    summary = _build_topline_summary(thread_panel, pr_panel, results)

    thread_panel.to_parquet(THREAD_PANEL_PARQUET, index=False)
    pr_panel.to_parquet(PR_PANEL_PARQUET, index=False)
    results.to_csv(RESULTS_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readout(summary, results)

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {THREAD_PANEL_PARQUET}")
    print(f"Wrote: {PR_PANEL_PARQUET}")
    print(f"Wrote: {RESULTS_CSV}")
    print(f"Wrote: {SUMMARY_JSON}")
    print(f"Wrote: {READOUT_PATH}")


if __name__ == "__main__":
    main()
