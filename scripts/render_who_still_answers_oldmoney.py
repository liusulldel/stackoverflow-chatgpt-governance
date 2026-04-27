from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(r"D:\AI alignment\projects\stackoverflow_chatgpt_governance")
PAPER_DIR = PROJECT_ROOT / "paper"
RENDER_DIR = PROJECT_ROOT / "rendered" / "who_still_answers_oldmoney_2026-04-05"
SEND_READY_DIR = Path(r"D:\AI alignment\overflow whostill answer")

DOCS = [
    ("01_Manuscript", PAPER_DIR / "who_still_answers_option_c_manuscript.md", "article"),
    ("02_Online_Appendix", PAPER_DIR / "who_still_answers_online_appendix.md", "article"),
    ("03_Cover_Letter", PAPER_DIR / "who_still_answers_isr_cover_letter.md", "letter"),
]


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def inline_to_latex(text: str) -> str:
    placeholders: list[str] = []

    def breakable_code(raw: str) -> str:
        escaped = latex_escape(raw)
        escaped = escaped.replace(r"\_", r"\_\allowbreak{}")
        escaped = escaped.replace("/", r"/\allowbreak{}")
        escaped = escaped.replace("-", r"-\allowbreak{}")
        escaped = escaped.replace("=", r"=\allowbreak{}")
        escaped = escaped.replace(">", r">\allowbreak{}")
        return r"\texttt{" + escaped + "}"

    def store(rendered: str) -> str:
        token = f"@@PH{len(placeholders)}@@"
        placeholders.append(rendered)
        return token

    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: store(latex_escape(m.group(1))),
        text,
    )
    text = re.sub(
        r"`([^`]+)`",
        lambda m: store(breakable_code(m.group(1))),
        text,
    )
    text = re.sub(
        r"\*\*([^*]+)\*\*",
        lambda m: store(r"\textbf{" + latex_escape(m.group(1)) + "}"),
        text,
    )
    text = re.sub(
        r"\*([^*]+)\*",
        lambda m: store(r"\emph{" + latex_escape(m.group(1)) + "}"),
        text,
    )

    text = latex_escape(text)
    for idx, rendered in enumerate(placeholders):
        text = text.replace(latex_escape(f"@@PH{idx}@@"), rendered)
    return text


def strip_title_number(title: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title).strip()


def parse_table(lines: list[str], start: int) -> tuple[str, int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].startswith("|"):
        raw = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
        rows.append(raw)
        i += 1

    if len(rows) < 2:
        return inline_to_latex(lines[start]), start + 1

    header = rows[0]
    body = [row for row in rows[2:] if row != ["---"] * len(header)]
    cols = len(header)
    if cols == 6:
        widths = [0.28, 0.10, 0.10, 0.10, 0.10, 0.22]
    elif cols == 5:
        widths = [0.31, 0.12, 0.12, 0.12, 0.23]
    elif cols == 4:
        widths = [0.34, 0.16, 0.14, 0.26]
    elif cols == 7:
        widths = [0.25, 0.105, 0.105, 0.085, 0.085, 0.085, 0.185]
    else:
        widths = [0.90 / cols for _ in range(cols)]
    colspec = "".join([rf">{{\raggedright\arraybackslash}}p{{{w:.3f}\textwidth}}" for w in widths])

    out = [
        r"\begin{center}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        rf"\begin{{longtable}}{{{colspec}}}",
        r"\toprule",
        " & ".join(inline_to_latex(cell) for cell in header) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(inline_to_latex(cell) for cell in header) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in body:
        padded = row + [""] * (cols - len(row))
        out.append(" & ".join(inline_to_latex(cell) for cell in padded[:cols]) + r" \\")
    out.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize", r"\end{center}"])
    return "\n".join(out), i


def parse_list(lines: list[str], start: int, numbered: bool) -> tuple[str, int]:
    env = "enumerate" if numbered else "itemize"
    out = [rf"\begin{{{env}}}"]
    i = start
    pattern = re.compile(r"^\d+\.\s+") if numbered else re.compile(r"^- ")
    while i < len(lines) and pattern.match(lines[i]):
        item = pattern.sub("", lines[i], count=1)
        out.append(r"\item " + inline_to_latex(item.strip()))
        i += 1
    out.append(rf"\end{{{env}}}")
    return "\n".join(out), i


def parse_paragraph(lines: list[str], start: int) -> tuple[str, int]:
    buf: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            break
        if (
            line.startswith("#")
            or line.startswith("|")
            or line.startswith("- ")
            or re.match(r"^\d+\.\s+", line)
        ):
            break
        buf.append(line.strip())
        i += 1
    paragraph = " ".join(buf).strip()
    if re.fullmatch(r"`[^`]+`", paragraph):
        code_text = paragraph[1:-1]
        return (
            r"\begin{quote}" "\n"
            r"\small" "\n"
            + inline_to_latex(f"`{code_text}`")
            + "\n"
            + r"\normalsize" "\n"
            + r"\end{quote}" "\n",
            i,
        )
    return inline_to_latex(paragraph) + "\n", i


def render_article(md_text: str, short_title: str) -> str:
    lines = md_text.splitlines()
    title = ""
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        lines = lines[1:]

    preamble = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1.1in]{{geometry}}
\usepackage{{fontspec}}
\setmainfont{{Palatino Linotype}}
\usepackage{{microtype}}
\usepackage{{setspace}}
\setstretch{{1.16}}
\usepackage{{indentfirst}}
\setlength{{\parindent}}{{1.2em}}
\setlength{{\parskip}}{{0.38em}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{booktabs,longtable,array}}
\setlength{{\LTleft}}{{0pt}}
\setlength{{\LTright}}{{0pt}}
\usepackage{{enumitem}}
\setlist[itemize]{{leftmargin=1.45em,itemsep=0.24em,topsep=0.34em}}
\setlist[enumerate]{{leftmargin=1.65em,itemsep=0.24em,topsep=0.34em}}
\usepackage{{titlesec}}
    \titleformat{{\section}}{{\normalfont\large\scshape}}{{}}{{0em}}{{}}
    \titleformat{{\subsection}}{{\normalfont\large\bfseries}}{{}}{{0em}}{{}}
    \titleformat{{\subsubsection}}{{\normalfont\normalsize\itshape}}{{}}{{0em}}{{}}
    \titlespacing*{{\section}}{{0pt}}{{0.8\baselineskip}}{{0.35\baselineskip}}
    \usepackage{{fancyhdr}}
    \pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\footnotesize\scshape {latex_escape(short_title)}}}
\fancyhead[R]{{\footnotesize\thepage}}
\renewcommand{{\headrulewidth}}{{0.3pt}}
\emergencystretch=2em
\widowpenalty=10000
\clubpenalty=10000
\begin{{document}}
\begin{{center}}
{{\LARGE\bfseries {latex_escape(title)}\par}}
    \vspace{{0.15em}}
\end{{center}}
\thispagestyle{{empty}}
"""

    out: list[str] = [preamble]
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if line.startswith("## Abstract"):
            i += 1
            abstract_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("## "):
                abstract_lines.append(lines[i])
                i += 1
            keywords_line = None
            filtered_lines: list[str] = []
            for a_line in abstract_lines:
                if a_line.strip().startswith("Keywords:"):
                    keywords_line = a_line.strip()[len("Keywords:"):].strip()
                else:
                    filtered_lines.append(a_line)
            paragraphs = [inline_to_latex(" ".join(p.strip() for p in block if p.strip())) for block in split_blocks(filtered_lines)]
            out.append(r"\begin{abstract}")
            if paragraphs:
                out.append(r"\par\medskip ".join(paragraphs))
            if keywords_line:
                out.append(
                    r"\par\smallskip\noindent{\small\textsc{Keywords:} "
                    + inline_to_latex(keywords_line)
                    + r"}"
                )
            out.append(r"\end{abstract}")
            continue
        if line.startswith("## "):
            out.append(r"\section*{" + latex_escape(strip_title_number(line[3:].strip())) + "}")
            i += 1
            continue
        if line.startswith("### "):
            out.append(r"\subsection*{" + latex_escape(strip_title_number(line[4:].strip())) + "}")
            i += 1
            continue
        if line.startswith("#### "):
            out.append(r"\subsubsection*{" + latex_escape(strip_title_number(line[5:].strip())) + "}")
            i += 1
            continue
        if line.startswith("|"):
            table_tex, i = parse_table(lines, i)
            out.append(table_tex)
            continue
        if line.startswith("- "):
            list_tex, i = parse_list(lines, i, numbered=False)
            out.append(list_tex)
            continue
        if re.match(r"^\d+\.\s+", line):
            list_tex, i = parse_list(lines, i, numbered=True)
            out.append(list_tex)
            continue
        para_tex, i = parse_paragraph(lines, i)
        out.append(para_tex)
    out.append(r"\end{document}")
    return "\n".join(out)


def split_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def render_letter(md_text: str, short_title: str) -> str:
    lines = md_text.splitlines()
    title = ""
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        lines = lines[1:]

    preamble = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1.15in]{{geometry}}
\usepackage{{fontspec}}
\setmainfont{{Palatino Linotype}}
\usepackage{{microtype}}
\usepackage{{setspace}}
\setstretch{{1.13}}
\usepackage[hidelinks]{{hyperref}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.55em}}
\begin{{document}}
\begin{{center}}
{{\Large\bfseries {latex_escape(title)}\par}}
\end{{center}}
\vspace{{0.6em}}
"""
    out: list[str] = [preamble]
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("## "):
            out.append(r"\textbf{" + latex_escape(strip_title_number(line[3:].strip())) + "}\n\n")
            i += 1
            continue
        if line.startswith("- "):
            list_tex, i = parse_list(lines, i, numbered=False)
            out.append(list_tex)
            continue
        if re.match(r"^\d+\.\s+", line):
            list_tex, i = parse_list(lines, i, numbered=True)
            out.append(list_tex)
            continue
        para_tex, i = parse_paragraph(lines, i)
        out.append(para_tex)
    out.append(r"\end{document}")
    return "\n".join(out)


def compile_tex(tex_path: Path) -> None:
    for _ in range(2):
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )


def prepare_render_dir() -> Path:
    if not RENDER_DIR.exists():
        RENDER_DIR.mkdir(parents=True, exist_ok=True)
        return RENDER_DIR
    try:
        shutil.rmtree(RENDER_DIR)
        RENDER_DIR.mkdir(parents=True, exist_ok=True)
        return RENDER_DIR
    except PermissionError:
        fallback = RENDER_DIR.parent / f"{RENDER_DIR.name}_{datetime.now():%H%M%S}"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def main() -> None:
    active_render_dir = prepare_render_dir()

    for stem, md_path, doc_type in DOCS:
        md_text = md_path.read_text(encoding="utf-8")
        short_title = "Private AI and Certification"
        tex_text = render_letter(md_text, short_title) if doc_type == "letter" else render_article(md_text, short_title)
        tex_path = active_render_dir / f"{stem}.tex"
        tex_path.write_text(tex_text, encoding="utf-8")
        compile_tex(tex_path)

    # refresh the easy-to-find folder with the old-money styled PDFs
    shutil.copy2(active_render_dir / "01_Manuscript.pdf", SEND_READY_DIR / "01_Manuscript.pdf")
    shutil.copy2(active_render_dir / "02_Online_Appendix.pdf", SEND_READY_DIR / "supporting materials" / "02_Online_Appendix.pdf")
    shutil.copy2(active_render_dir / "03_Cover_Letter.pdf", SEND_READY_DIR / "supporting materials" / "03_Cover_Letter.pdf")
    source_dir = SEND_READY_DIR / "source files"
    source_dir.mkdir(parents=True, exist_ok=True)
    for _, md_path, _ in DOCS:
        shutil.copy2(md_path, source_dir / md_path.name)

    preprint_source = PAPER_DIR / "who_still_answers_preprint_title_abstract_keywords_2026-04-07.md"
    if preprint_source.exists():
        shutil.copy2(preprint_source, SEND_READY_DIR / "supporting materials" / "preprint_title_abstract_keywords.md")
    insider_source = PAPER_DIR / "who_still_answers_insider_packet_2026-04-07.md"
    if insider_source.exists():
        shutil.copy2(insider_source, SEND_READY_DIR / "supporting materials" / "insider_packet.md")
    final_rec_source = PAPER_DIR / "who_still_answers_final_finish_recommendation_2026-04-16.md"
    if final_rec_source.exists():
        shutil.copy2(final_rec_source, SEND_READY_DIR / "supporting materials" / "final_finish_recommendation.md")


if __name__ == "__main__":
    main()
