from __future__ import annotations

import html
import shutil
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = PROJECT_ROOT / "paper" / "staged_public_resolution"
PROCESSED_DIR = PROJECT_ROOT / "processed"
DESKTOP = Path.home() / "Desktop"
PACKAGE_DIR = DESKTOP / f"StackOverflow_ISR_Submission_Package_{date.today().isoformat()}"
LATEST_PACKAGE_DIR = DESKTOP / "StackOverflow_ISR_Submission_Package_latest"
SOURCE_DIR = PACKAGE_DIR / "source_markdown"
SUPPORT_DIR = PACKAGE_DIR / "supporting_outputs"


MAIN_FILES = [
    ("00_Package_Guide", PAPER_DIR / "isr_final_package.md"),
    ("01_Manuscript", PAPER_DIR / "isr_submission_manuscript.md"),
    ("02_Online_Appendix", PAPER_DIR / "isr_submission_online_appendix.md"),
    ("03_Cover_Letter", PAPER_DIR / "isr_cover_letter.md"),
    ("04_Submission_Memo", PAPER_DIR / "isr_submission_package.md"),
]

OPTIONAL_SUPPORT_FILES = [
    ("05_Closure_Ladder_Readout", PAPER_DIR / "closure_ladder_results.md"),
    ("06_Selection_Composition_Readout", PAPER_DIR / "selection_composition_evidence.md"),
    ("07_Exposure_Index_Readout", PAPER_DIR / "question_level_exposure_index.md"),
    ("08_Validation_Report", PROCESSED_DIR / "stackexchange_20251231_validation_report.md"),
    ("09_Submission_Gate_Check", PAPER_DIR / "submission_gate_check_2026-04-04.md"),
    ("10_Strict_ISR_Reclearance", PAPER_DIR / "strict_isr_editor_reclearance_2026-04-04.md"),
    ("11_Citation_Integrity_Audit", PAPER_DIR / "citation_integrity_audit_2026-04-04.md"),
    ("12_Strict_Call_Audit", PAPER_DIR / "strict_call_audit_2026-04-04.md"),
]

RAW_SUPPORT_FILES = [
    ("closure_ladder_model_results.csv", PROCESSED_DIR / "closure_ladder_model_results.csv"),
    ("closure_ladder_window_trajectory.csv", PROCESSED_DIR / "closure_ladder_window_trajectory.csv"),
    ("selection_composition_model_results.csv", PROCESSED_DIR / "selection_composition_model_results.csv"),
    ("selection_composition_results.json", PROCESSED_DIR / "selection_composition_results.json"),
    ("question_level_exposure_validation_regressions.csv", PROCESSED_DIR / "question_level_exposure_validation_regressions.csv"),
    ("question_level_exposure_model_validation.json", PROCESSED_DIR / "question_level_exposure_model_validation.json"),
    ("stack_overflow_2020_2025_validation_report.json", PROCESSED_DIR / "stackexchange_20251231_validation_report.json"),
    ("stack_overflow_2020_2025_validation_report.md", PROCESSED_DIR / "stackexchange_20251231_validation_report.md"),
]


def register_fonts() -> tuple[str, str]:
    sans_candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]
    mono_candidates = [
        Path(r"C:\Windows\Fonts\consola.ttf"),
        Path(r"C:\Windows\Fonts\cour.ttf"),
    ]

    sans_name = "Helvetica"
    mono_name = "Courier"

    for candidate in sans_candidates:
        if candidate.exists():
            pdfmetrics.registerFont(TTFont("PkgSans", str(candidate)))
            sans_name = "PkgSans"
            break

    for candidate in mono_candidates:
        if candidate.exists():
            pdfmetrics.registerFont(TTFont("PkgMono", str(candidate)))
            mono_name = "PkgMono"
            break

    return sans_name, mono_name


def build_styles(sans_name: str, mono_name: str):
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "PkgBody",
        parent=styles["BodyText"],
        fontName=sans_name,
        fontSize=11,
        leading=22,
        spaceAfter=10,
    )
    title = ParagraphStyle(
        "PkgTitle",
        parent=styles["Title"],
        fontName=sans_name,
        fontSize=19,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    h1 = ParagraphStyle(
        "PkgH1",
        parent=styles["Heading1"],
        fontName=sans_name,
        fontSize=15,
        leading=24,
        textColor=colors.HexColor("#17324d"),
        spaceBefore=12,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "PkgH2",
        parent=styles["Heading2"],
        fontName=sans_name,
        fontSize=12.5,
        leading=20,
        textColor=colors.HexColor("#1f4c73"),
        spaceBefore=10,
        spaceAfter=6,
    )
    bullet = ParagraphStyle(
        "PkgBullet",
        parent=body,
        leftIndent=18,
        bulletIndent=6,
        spaceAfter=6,
    )
    code = ParagraphStyle(
        "PkgCode",
        parent=body,
        fontName=mono_name,
        fontSize=9.2,
        leading=16,
        leftIndent=10,
        rightIndent=10,
        backColor=colors.HexColor("#f4f6f8"),
        borderPadding=6,
        borderColor=colors.HexColor("#d8dee4"),
        borderWidth=0.5,
        borderRadius=2,
        spaceAfter=8,
    )
    return {
        "body": body,
        "title": title,
        "h1": h1,
        "h2": h2,
        "bullet": bullet,
        "code": code,
    }


def normalize_inline(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = escaped.replace("`", "")
    return escaped


def flush_paragraph(paragraph_lines: list[str], story: list, styles) -> None:
    if not paragraph_lines:
        return
    text = " ".join(line.strip() for line in paragraph_lines if line.strip())
    if text:
        story.append(Paragraph(normalize_inline(text), styles["body"]))
    paragraph_lines.clear()


def flush_codeblock(code_lines: list[str], story: list, styles) -> None:
    if not code_lines:
        return
    story.append(Preformatted("\n".join(code_lines), styles["code"]))
    code_lines.clear()

def flush_image(image_path: str, story: list, styles, doc_width: float) -> None:
    normalized = image_path.strip()
    if normalized.startswith("/") and len(normalized) > 3 and normalized[2] == ":":
        normalized = normalized[1:]
    path = Path(normalized)
    if not path.exists():
        story.append(Paragraph(normalize_inline(f"[Missing image: {image_path}]"), styles["body"]))
        return
    reader = ImageReader(str(path))
    width_px, height_px = reader.getSize()
    max_width = doc_width
    max_height = 4.6 * inch
    scale = min(max_width / width_px, max_height / height_px)
    story.append(Image(str(path), width=width_px * scale, height=height_px * scale))
    story.append(Spacer(1, 0.08 * inch))


def add_markdown_to_story(text: str, story: list, styles, doc_width: float) -> None:
    paragraph_lines: list[str] = []
    code_lines: list[str] = []
    in_code = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph(paragraph_lines, story, styles)
            if in_code:
                flush_codeblock(code_lines, story, styles)
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            flush_paragraph(paragraph_lines, story, styles)
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph(paragraph_lines, story, styles)
            code_lines.append(line)
            continue
        elif code_lines:
            flush_codeblock(code_lines, story, styles)

        if stripped.startswith("# "):
            flush_paragraph(paragraph_lines, story, styles)
            story.append(Paragraph(normalize_inline(stripped[2:].strip()), styles["title"]))
            continue
        if stripped.startswith("## "):
            flush_paragraph(paragraph_lines, story, styles)
            story.append(Paragraph(normalize_inline(stripped[3:].strip()), styles["h1"]))
            continue
        if stripped.startswith("### "):
            flush_paragraph(paragraph_lines, story, styles)
            story.append(Paragraph(normalize_inline(stripped[4:].strip()), styles["h2"]))
            continue

        if stripped.startswith("![") and "](" in stripped and stripped.endswith(")"):
            flush_paragraph(paragraph_lines, story, styles)
            alt, path = stripped[2:].split("](", 1)
            path = path[:-1]
            if alt.strip():
                story.append(Paragraph(normalize_inline(alt.strip()), styles["h2"]))
            flush_image(path, story, styles, doc_width)
            continue

        if stripped.startswith("- "):
            flush_paragraph(paragraph_lines, story, styles)
            story.append(
                Paragraph(
                    normalize_inline(stripped[2:].strip()),
                    styles["bullet"],
                    bulletText="-",
                )
            )
            continue

        if stripped[:2].isdigit() and stripped[2:3] == ".":
            flush_paragraph(paragraph_lines, story, styles)
            marker, content = stripped.split(".", 1)
            story.append(
                Paragraph(
                    normalize_inline(content.strip()),
                    styles["bullet"],
                    bulletText=f"{marker}.",
                )
            )
            continue

        paragraph_lines.append(line)

    flush_paragraph(paragraph_lines, story, styles)
    flush_codeblock(code_lines, story, styles)


def add_page_number(canvas, doc):
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#586069"))
    canvas.drawRightString(doc.pagesize[0] - 1.0 * inch, 0.7 * inch, f"Page {doc.page}")


def render_pdf(markdown_path: Path, pdf_path: Path, styles) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    story: list = []
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        leftMargin=1.0 * inch,
        rightMargin=1.0 * inch,
        topMargin=1.0 * inch,
        bottomMargin=1.0 * inch,
        title=markdown_path.stem,
        author="Codex research workflow",
    )
    add_markdown_to_story(text, story, styles, doc.width)
    if story and not isinstance(story[-1], Spacer):
        story.append(Spacer(1, 0.12 * inch))
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def ensure_clean_package_dir() -> None:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    SUPPORT_DIR.mkdir(parents=True, exist_ok=True)


def copy_support_files() -> None:
    for export_name, source in RAW_SUPPORT_FILES:
        if source.exists():
            shutil.copy2(source, SUPPORT_DIR / export_name)


def refresh_latest_alias() -> None:
    if LATEST_PACKAGE_DIR.exists():
        shutil.rmtree(LATEST_PACKAGE_DIR)
    shutil.copytree(PACKAGE_DIR, LATEST_PACKAGE_DIR)


def main() -> None:
    ensure_clean_package_dir()
    sans_name, mono_name = register_fonts()
    styles = build_styles(sans_name, mono_name)

    all_render_files = MAIN_FILES + OPTIONAL_SUPPORT_FILES
    for label, source in all_render_files:
        if not source.exists():
            continue
        target_md = SOURCE_DIR / source.name
        shutil.copy2(source, target_md)
        target_pdf = PACKAGE_DIR / f"{label}.pdf"
        render_pdf(source, target_pdf, styles)

    copy_support_files()
    refresh_latest_alias()
    print(PACKAGE_DIR)


if __name__ == "__main__":
    main()
