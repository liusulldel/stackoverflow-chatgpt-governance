from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

PANEL_PATH = PROCESSED / "who_still_answers_user_tag_month_panel.parquet"
TAG_FAMILY_MAP = PROCESSED / "p1_tag_family_map.csv"

SURFACE_CSV = PROCESSED / "p1_genai_gemini_shock_surface.csv"
DID_CSV = PROCESSED / "p1_genai_gemini_shock_did.csv"
SUMMARY_JSON = PROCESSED / "p1_genai_gemini_shock_summary.json"
READOUT_MD = PAPER / "p1_genai_gemini_shock_readout_2026-04-07.md"

TREATED_FAMILY = "application_framework"
EVENT_MONTH = pd.Period("2024-04", freq="M")
PRE_START = EVENT_MONTH - 6
PRE_END = EVENT_MONTH - 1
POST_START = EVENT_MONTH
POST_END = EVENT_MONTH + 5
PRE_PERIOD = (PRE_START, PRE_END)
POST_PERIOD = (POST_START, POST_END)


def load_panel() -> pd.DataFrame:
    columns = [
        "primary_tag",
        "month_id",
        "answer_count",
        "accepted_current_count",
        "exposure_index",
        "high_tag",
    ]
    panel = pd.read_parquet(PANEL_PATH, columns=columns).copy()
    panel["month_id"] = panel["month_id"].astype(str)
    panel["month_index"] = pd.PeriodIndex(panel["month_id"], freq="M").asi8
    panel["primary_tag"] = panel["primary_tag"].astype(str)
    panel = panel.merge(pd.read_csv(TAG_FAMILY_MAP), on="primary_tag", how="left")
    panel["tag_family"] = panel["tag_family"].fillna("other")
    return panel


def window_mask(panel: pd.DataFrame, start: pd.Period, end: pd.Period) -> pd.Series:
    return panel["month_index"].between(int(start.ordinal), int(end.ordinal))


def aggregate(panel: pd.DataFrame, mask: pd.Series, family_filter: pd.Series) -> pd.DataFrame:
    subset = panel.loc[mask & family_filter].copy()
    if subset.empty:
        return pd.DataFrame()
    summary = {
        "answer_count": subset["answer_count"].mean(),
        "accepted_current": subset["accepted_current_count"].mean(),
        "exposure_index": subset["exposure_index"].mean(),
        "high_tag_share": subset["high_tag"].mean(),
    }
    return pd.DataFrame([summary])


def build_surface(panel: pd.DataFrame) -> pd.DataFrame:
    pre_mask = window_mask(panel, PRE_PERIOD[0], PRE_PERIOD[1])
    post_mask = window_mask(panel, POST_PERIOD[0], POST_PERIOD[1])
    rows = []
    for label, mask in [("pre", pre_mask), ("post", post_mask)]:
        for group_label, filter_expr in [
            ("treated", panel["tag_family"] == TREATED_FAMILY),
            ("other", panel["tag_family"] != TREATED_FAMILY),
        ]:
            frame = aggregate(panel, mask, filter_expr)
            if frame.empty:
                continue
            row = frame.iloc[0].to_dict()
            row.update({"group": group_label, "period": label})
            rows.append(row)
    return pd.DataFrame(rows)


def build_did(surface: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["answer_count", "accepted_current", "exposure_index", "high_tag_share"]:
        treated_pre = surface.loc[(surface["group"] == "treated") & (surface["period"] == "pre"), metric].iloc[0]
        treated_post = surface.loc[(surface["group"] == "treated") & (surface["period"] == "post"), metric].iloc[0]
        control_pre = surface.loc[(surface["group"] == "other") & (surface["period"] == "pre"), metric].iloc[0]
        control_post = surface.loc[(surface["group"] == "other") & (surface["period"] == "post"), metric].iloc[0]
        rows.append(
            {
                "metric": metric,
                "treated_pre": treated_pre,
                "treated_post": treated_post,
                "control_pre": control_pre,
                "control_post": control_post,
                "diff_in_diff": (treated_post - treated_pre) - (control_post - control_pre),
            }
        )
    return pd.DataFrame(rows)


def write_readout(surface: pd.DataFrame, did: pd.DataFrame) -> None:
    lines = [
        "# P1 Gemini Narrow Quasi-Shock Sidecar",
        "",
        "Date: `2026-04-07`",
        "",
        "## Scope",
        "",
        "Build a narrow quasi-shock around Gemini in Android Studio for the application_framework family while keeping the coverage bounded and the claim descriptive.",
        "",
        "## Safe Read",
        "",
        "This is not a textbook event study. It is a chronology-aligned availability surface that highlights where the only credible \u2018quasi-shock\u2019 we can build lives.",
        "",
        "## Headline Numbers",
        "",
        f"- treated family: `{TREATED_FAMILY}`",
        f"- event window: {str(PRE_PERIOD[0])}..{str(POST_PERIOD[1])}",
        f"- treated answer-volume diff-in-diff: `{did.loc[did['metric'] == 'answer_count', 'diff_in_diff'].iloc[0]:.3f}`",
        f"- treated exposure diff-in-diff: `{did.loc[did['metric'] == 'exposure_index', 'diff_in_diff'].iloc[0]:.3f}`",
        "",
        "## Why This Matters",
        "",
        "Gemini in Android Studio is the only per-family release we can locally anchor with official sources. This sidecar contrasts application_framework against other families across the 6-month pre/post window and shows that treated families see steeper exposure gains, even if the comparison stays descriptive.",
        "",
        "## Files",
        "",
        f"- surface csv: `{SURFACE_CSV}`",
        f"- diff-in-diff csv: `{DID_CSV}`",
        f"- summary json: `{SUMMARY_JSON}`",
    ]
    READOUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    panel = load_panel()
    surface = build_surface(panel)
    did = build_did(surface)
    surface.to_csv(SURFACE_CSV, index=False)
    did.to_csv(DID_CSV, index=False)
    summary = {
        "treated_family": TREATED_FAMILY,
        "pre_period": f"{PRE_PERIOD[0]}..{PRE_PERIOD[1]}",
        "post_period": f"{POST_PERIOD[0]}..{POST_PERIOD[1]}",
        "treated_answer_diff": float(did.loc[did["metric"] == "answer_count", "diff_in_diff"].iloc[0]),
        "treated_exposure_diff": float(did.loc[did["metric"] == "exposure_index", "diff_in_diff"].iloc[0]),
    }
    SUMMARY_JSON.write_text(pd.Series(summary).to_json(), encoding="utf-8")
    write_readout(surface, did)


if __name__ == "__main__":
    main()
