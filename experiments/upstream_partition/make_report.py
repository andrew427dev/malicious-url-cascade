"""Render this run's findings using the team's November 23 report format."""

import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from diagram import draw
from report_content import build

HERE = Path(__file__).resolve().parent


def field(p, instruction):
    el = OxmlElement("w:fldSimple")
    el.set(qn("w:instr"), instruction)
    p._p.append(el)


def bookmark(p, index):
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(index))
    start.set(qn("w:name"), f"section{index}")
    p._p.insert(1, start)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(index))
    p._p.append(end)


def numbered_sections(pages):
    chapter, subsection = 0, 0
    for page in pages[1:]:
        if page["chapter"]:
            chapter += 1
            subsection = 0
            number = str(chapter)
        else:
            subsection += 1
            number = f"{chapter}.{subsection}"
        yield number, page


def table(doc, caption, heads, rows):
    p = doc.add_paragraph(caption, style="Caption")
    p.paragraph_format.keep_with_next = True
    count = len(heads)
    widths = {
        2: [0.43, 0.57],
        3: [0.34, 0.29, 0.37],
        4: [0.28, 0.28, 0.22, 0.22],
        5: [0.24, 0.16, 0.20, 0.20, 0.20],
        6: [0.19, 0.14, 0.16, 0.15, 0.22, 0.14],
        7: [0.17, 0.13, 0.14, 0.14, 0.14, 0.14, 0.14],
        8: [0.165, 0.12, 0.12, 0.12, 0.11, 0.12, 0.14, 0.105],
    }[count]
    if heads[:2] == ["순서", "원저자 열 이름"]:
        widths = [0.068, 0.258, 0.674]
    elif heads[0] == "파일":
        widths = [0.43, 0.14, 0.14, 0.29]
    elif heads[0] == "단계":
        widths = [0.60, 0.20, 0.20]
    elif heads[0] == "학습 블록":
        widths = [0.17, 0.32, 0.17, 0.17, 0.17]
    tab = doc.add_table(rows=0, cols=count)
    tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    tab.autofit = False
    for col, w in zip(tab.columns, widths):
        col.width = Cm(16 * w)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:" + side)
        if side in ("top", "bottom", "insideH"):
            color, size = ("D6D6D6", "3") if side == "insideH" else ("303030", "7")
            for k, v in {
                "val": "single",
                "sz": size,
                "space": "0",
                "color": color,
            }.items():
                el.set(qn("w:" + k), v)
        else:
            el.set(qn("w:val"), "nil")
        borders.append(el)
    tab._tbl.tblPr.append(borders)
    numeric = [
        all(re.fullmatch(r"[0-9.,%+−–—/ \-]+", str(row[j])) for row in rows)
        for j in range(count)
    ]
    for i, values in enumerate([heads, *rows]):
        row = tab.add_row()
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        if i == 0:
            row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for j, (cell, value, w) in enumerate(zip(row.cells, values, widths)):
            cell.width = Cm(16 * w)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if i == 0:
                fill = OxmlElement("w:shd")
                for k, v in {"val": "clear", "color": "auto", "fill": "F1F1F1"}.items():
                    fill.set(qn("w:" + k), v)
                cell._tc.get_or_add_tcPr().append(fill)
            p = cell.paragraphs[0]
            p.style = "Cell"
            if numeric[j]:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            # Dense result matrices and the 18-feature table keep the same font and rules.
            if count >= 6 or heads[:2] == ["순서", "원저자 열 이름"]:
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = Pt(14)
            p.add_run(str(value)).bold = i == 0


def make(root, regular, bold):
    pages = build(root)
    dest = root / "report"
    dest.mkdir(exist_ok=True)
    (dest / "content.json").write_text(
        json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf8"
    )
    figure = dest / "cascade.png"
    draw(figure, regular, bold)
    doc = Document(HERE / "nov23-format.docx")
    # The reference uses Calibri for Latin headings and Malgun Gothic for Korean.
    for name in ("Title", "Heading 1", "Heading 2"):
        fonts = doc.styles[name].element.get_or_add_rPr().rFonts
        for k in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
            fonts.attrib.pop(qn("w:" + k), None)
        for k, v in (
            ("ascii", "Calibri"),
            ("hAnsi", "Calibri"),
            ("eastAsia", "맑은 고딕"),
        ):
            fonts.set(qn("w:" + k), v)
    written = datetime.now(timezone(timedelta(hours=9))).date()
    doc.paragraphs[1].text = "원저자 전처리 기준\n악성 URL 탐지 재실험"
    doc.paragraphs[
        2
    ].text = f"같은 GRU를 사용한 경량 분류기의 선택적 검사  ·  {written.isoformat()}"
    toc = doc.add_paragraph("목차")
    toc.paragraph_format.page_break_before = True
    toc.paragraph_format.space_after = Pt(13)
    toc.paragraph_format.line_spacing = Pt(33)
    toc.runs[0].bold = True
    toc.runs[0].font.size = Pt(23)
    sections = list(numbered_sections(pages))
    for i, (number, page) in enumerate(sections, 1):
        p = doc.add_paragraph(style="toc 1" if page["chapter"] else "toc 2")
        p.paragraph_format.tab_stops.add_tab_stop(
            Cm(15.9), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS
        )
        p.add_run(number + "  " + page["title"] + "\t")
        field(p, f"PAGEREF section{i} \\h")
    for i, (number, page) in enumerate(sections, 1):
        p = doc.add_paragraph(
            number + "  " + page["title"],
            style="Heading 1" if page["chapter"] else "Heading 2",
        )
        p.paragraph_format.page_break_before = True
        bookmark(p, i)
        for tag, *values in page["blocks"]:
            if tag == "paragraph":
                doc.add_paragraph(values[0])
            elif tag == "table":
                table(doc, *values)
            elif tag == "figure":
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.line_spacing = 1
                p.paragraph_format.keep_with_next = True
                p.add_run().add_picture(str(figure), width=Cm(16))
                doc.add_paragraph(values[0], style="Caption")
            elif tag == "sub":
                doc.add_paragraph(values[0], style="Minor heading")
            else:
                raise ValueError(f"Unexpected body block {tag}")
    doc.core_properties.title = pages[0]["title"]
    doc.core_properties.author = "컴퓨터보안 팀프로젝트"
    doc.save(dest / "원저자 전처리 기준 악성 URL 탐지 재실험 보고서.docx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regular", type=Path, default=Path("C:/Windows/Fonts/malgun.ttf")
    )
    parser.add_argument(
        "--bold", type=Path, default=Path("C:/Windows/Fonts/malgunbd.ttf")
    )
    args = parser.parse_args()
    make(Path(os.environ["URLCASCADE_RUN_DIR"]), args.regular, args.bold)
