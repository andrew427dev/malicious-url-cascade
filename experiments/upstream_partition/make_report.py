"""Compose the Korean DOCX and semantic report content from the verified results."""

import argparse
import json
import os
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from diagram import draw
from report_content import build, CHAPTERS

FONT = "맑은 고딕"


def font(run, size=None, bold=None):
    run.font.name = FONT
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def field(p, instruction):
    f = OxmlElement("w:fldSimple")
    f.set(qn("w:instr"), instruction)
    p._p.append(f)


def table(doc, caption, heads, rows):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    font(p.add_run(caption), 10, True)
    count = len(heads)
    widths = {
        2: [0.43, 0.57],
        3: [0.34, 0.29, 0.37],
        4: [0.27, 0.28, 0.22, 0.23],
        5: [0.24, 0.16, 0.20, 0.20, 0.20],
        6: [0.205, 0.125, 0.17, 0.15, 0.20, 0.15],
        7: [0.165, 0.13, 0.14, 0.14, 0.14, 0.14, 0.145],
        8: [0.18, 0.12, 0.12, 0.12, 0.11, 0.12, 0.13, 0.10],
    }[count]
    if heads[:2] == ["순서", "원저자 열 이름"]:
        widths = [0.07, 0.26, 0.67]
    elif heads[0] == "파일":
        widths = [0.43, 0.14, 0.14, 0.29]
    elif heads[0] == "단계":
        widths = [0.64, 0.18, 0.18]
    tab = doc.add_table(rows=0, cols=count)
    tab.style = "Table Grid"
    tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    tab.autofit = False
    for col, w in zip(tab.columns, widths):
        col.width = Cm(17.4 * w)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:" + side)
        for k, v in {"val": "single", "sz": "4", "color": "D9D9D9"}.items():
            el.set(qn("w:" + k), v)
        borders.append(el)
    tab._tbl.tblPr.append(borders)
    for i, values in enumerate([heads, *rows]):
        row = tab.add_row()
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        if not i:
            row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for j, (cell, value, w) in enumerate(zip(row.cells, values, widths)):
            cell.width = Cm(17.4 * w)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if i == 0:
                sh = OxmlElement("w:shd")
                sh.set(qn("w:fill"), "E2EAF0")
                cell._tc.get_or_add_tcPr().append(sh)
            p = cell.paragraphs[0]
            fmt = p.paragraph_format
            fmt.space_before = Pt(2)
            fmt.space_after = Pt(2)
            fmt.line_spacing = 1.08
            if j and (
                count >= 5 or heads[j] in ("정상", "악성", "남은 행", "이번 단계 제외")
            ):
                p.alignment = (
                    WD_ALIGN_PARAGRAPH.RIGHT if i else WD_ALIGN_PARAGRAPH.CENTER
                )
            font(p.add_run(str(value)), 8.6 if count >= 7 else 9.1, i == 0)


def make(root, regular, bold):
    from datetime import datetime, timedelta, timezone

    written = datetime.now(timezone(timedelta(hours=9))).date()
    pages = build(root)
    pages[0]["blocks"].insert(1, ("paragraph", f"작성일 {written.isoformat()}"))
    directory = root / "report"
    directory.mkdir(exist_ok=True)
    (directory / "content.json").write_text(
        json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf8"
    )
    figure = directory / "cascade.png"
    draw(figure, regular, bold)
    doc = Document()
    sec = doc.sections[0]
    sec.page_height = Cm(29.7)
    sec.page_width = Cm(21)
    sec.top_margin = Cm(2.15)
    sec.bottom_margin = Cm(2.05)
    sec.left_margin = Cm(1.8)
    sec.right_margin = Cm(1.8)
    sec.header_distance = Cm(0.9)
    sec.footer_distance = Cm(0.9)
    for name in ("Normal", "Title", "Heading 1", "Heading 2"):
        style = doc.styles[name]
        style.font.name = FONT
        style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
        style.font.color.rgb = RGBColor.from_string(
            "142735" if name == "Normal" else "111111"
        )
    for border in doc.styles["Title"]._element.xpath(".//w:pBdr"):
        border.getparent().remove(border)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.3
    for name, sz in (("Title", 29), ("Heading 1", 21), ("Heading 2", 15)):
        s = doc.styles[name]
        s.font.size = Pt(sz)
        s.font.bold = True
        s.paragraph_format.space_before = Pt(0)
        s.paragraph_format.space_after = Pt(10)
    header = sec.header.paragraphs[0]
    font(header.add_run("컴퓨터보안 팀프로젝트   /   악성 URL 탐지"), 8.5)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    field(footer, "PAGE")
    for i, page in enumerate(pages):
        p = doc.add_paragraph(
            style="Title" if i == 0 else "Heading 1" if page["chapter"] else "Heading 2"
        )
        p.paragraph_format.page_break_before = bool(i)
        p.add_run(page["title"])
        if page["chapter"]:
            tag = OxmlElement("w:bookmarkStart")
            tag.set(qn("w:id"), str(i))
            tag.set(qn("w:name"), f"chapter{CHAPTERS.index(page['title']) + 1}")
            p._p.insert(1, tag)
            end = OxmlElement("w:bookmarkEnd")
            end.set(qn("w:id"), str(i))
            p._p.append(end)
        for tag, *values in page["blocks"]:
            if tag == "paragraph":
                doc.add_paragraph(values[0])
            elif tag == "table":
                table(doc, *values)
            elif tag == "figure":
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(str(figure), width=Cm(17.2))
                font(doc.add_paragraph().add_run(values[0]), 10, True)
            elif tag == "subtitle":
                font(doc.add_paragraph().add_run(values[0]), 14)
            elif tag == "sub":
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.keep_with_next = True
                font(p.add_run(values[0]), 12, True)
            elif tag == "toc":
                for n, title in enumerate(values[0], 1):
                    p = doc.add_paragraph()
                    p.paragraph_format.tab_stops.add_tab_stop(
                        Cm(16.5), WD_TAB_ALIGNMENT.RIGHT
                    )
                    font(p.add_run(f"{n}   {title}\t"), 10)
                    field(p, f"PAGEREF chapter{n} \\h")
    doc.core_properties.title = pages[0]["title"]
    doc.core_properties.author = "컴퓨터보안 팀프로젝트"
    doc.save(directory / "원저자 전처리 기준 악성 URL 탐지 재실험 보고서.docx")


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
