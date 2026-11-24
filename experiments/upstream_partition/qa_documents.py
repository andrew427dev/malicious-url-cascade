"""Verify ordered headings, TOC page numbers, tables, figure and feature example."""

import argparse
import json
import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from make_report import numbered_sections
import pymupdf
from PIL import Image, ImageChops
from feature_table import FEATURE_ROWS, example
from table_layout import present
from report_content import CHAPTERS


def tight(s):
    return re.sub(r"\s+", "", s)


def verify(root):
    dest = root / "report"
    docx = next(dest.glob("*.docx"))
    doc = Document(docx)
    pdf = pymupdf.open(docx.with_suffix(".pdf"))
    pages = [tight(p.get_text()) for p in pdf]
    assert len(pdf) >= 8 and len(doc.inline_shapes) == 1
    actual = [p.text for p in doc.paragraphs if p.style.name == "Heading 1"]
    assert actual == [f"{i}  {title}" for i, title in enumerate(CHAPTERS, 1)]
    assert "컴퓨터보안팀프로젝트결과보고서" in pages[0]
    assert "목차" in pages[1]
    toc = re.sub(r"\.{2,}", "", pages[1])
    content = json.loads((dest / "content.json").read_text(encoding="utf8"))
    index_pages = {}
    for number, page in numbered_sections(content):
        title = page["title"]
        heading = tight(f"{number} {title}")
        hits = [n for n, s in enumerate(pages[2:], 3) if heading in s]
        assert len(hits) == 1, (title, hits)
        n = hits[0]
        assert heading + str(n) in toc, (title, n)
        index_pages[title] = n
    assert list(index_pages.values()) == sorted(index_pages.values())
    for p in doc.paragraphs:
        if p.style.name not in ("Heading 2", "Minor heading", "Caption"):
            continue
        forced = p.paragraph_format.page_break_before
        style = p.style
        while forced is None and style is not None:
            forced = style.paragraph_format.page_break_before
            style = style.base_style
        assert not forced, ("forced break", p.text)
    expected = [
        (caption, heads, rows)
        for page in content
        for typ, *payload in page["blocks"]
        if typ == "table"
        for caption, heads, rows in [payload]
    ]
    printed = [part for data in expected for part in present(*data)]
    assert len(doc.tables) == 1 + len(printed)
    assert doc.tables[0].cell(0, 0).text == "소속"
    all_pdf = "".join(pages)
    offsets = [all_pdf.index(tight(part.caption)) for part in printed] + [len(all_pdf)]
    assert offsets == sorted(offsets)
    for i, (tab, part) in enumerate(zip(doc.tables[1:], printed)):
        caption, headers, rows = part.caption, part.headers, part.rows
        section = all_pdf[offsets[i] : offsets[i + 1]]
        assert len(headers) <= 5 and tight("".join(headers)) in section, caption
        for row in rows:
            needle = tight("".join(row))
            assert needle in section, (caption, row)
            section = section[section.index(needle) + len(needle) :]
        borders = tab._tbl.tblPr.find(qn("w:tblBorders"))
        for side, color in (
            ("top", "303030"),
            ("bottom", "303030"),
            ("insideH", "D6D6D6"),
        ):
            assert borders.find(qn("w:" + side)).get(qn("w:color")) == color
        for side in ("left", "right", "insideV"):
            assert borders.find(qn("w:" + side)).get(qn("w:val")) == "nil"
        assert all(
            cell._tc.tcPr.find(qn("w:shd")).get(qn("w:fill")) == "F1F1F1"
            for cell in tab.rows[0].cells
        )
        assert [[c.text for c in row.cells] for row in tab.rows] == [headers, *rows], (
            caption
        )
        assert tight(caption) in all_pdf, caption
    # A second calculation from exported result rows verifies the generated tables.
    import csv
    from report_content import POLICY, GROUP, FORM

    perf = list(
        csv.DictReader((root / "summary/performance.csv").open(encoding="utf8"))
    )
    timing = list(csv.DictReader((root / "summary/timing.csv").open(encoding="utf8")))

    def matching_table(condition, timing=False):
        group, form = condition.split("-")
        caption = GROUP[group] + " 분할, " + FORM[form]
        section = [
            t
            for t in expected
            if caption in t[0] and ("동일 1,000건" in t[0]) == timing
        ]
        assert len(section) == 1, (condition, timing)
        return {tuple(row) for row in section[0][2]}

    for r in perf:
        data_cells = matching_table(r["condition"])
        cells = [
            POLICY[r["policy"]],
            *[f"{100 * float(r[k]):.2f}" for k in ("accuracy", "recall", "fpr")],
            *[f"{int(r[k]):,}" for k in ("fn", "additional_misses", "t2_calls")],
            f"{100 * float(r['t2_call_rate']):.2f}",
        ]
        assert tuple(cells) in data_cells, (r["condition"], r["policy"])
    for r in timing:
        data_cells = matching_table(r["condition"], timing=True)
        cells = [
            POLICY[r["policy"]],
            f"{int(r['calls']):,}",
            f"{float(r['calls_saved_pct']):.1f}",
            f"{float(r['ms']):.2f}",
            f"{float(r['min_ms']):.2f}–{float(r['max_ms']):.2f}",
            f"{float(r['time_saved_pct']):.2f}",
        ]
        assert tuple(cells) in data_cells
    setup = pages[index_pages["System Setup"] - 1 : index_pages[CHAPTERS[3]] - 1]
    setup_text = "".join(setup)
    frag = next(t for t in doc.tables if t.cell(0, 1).text == "원저자 열 이름")
    assert [(r.cells[1].text, r.cells[2].text) for r in frag.rows[1:]] == FEATURE_ROWS
    for n, definition in FEATURE_ROWS:
        assert tight(n) in setup_text and tight(definition) in setup_text, n
    assert tight(example(root)) in setup_text
    assert any(p.text == example(root) for p in doc.paragraphs)
    end = pages[index_pages[CHAPTERS[-1]] - 1]
    labels = [tight(s) for s in ("이번 재실험의 결론", "참고문헌")]
    assert all(s in end for s in labels) and [end.index(s) for s in labels] == sorted(
        end.index(s) for s in labels
    )
    for historical in ("11월23일", "이전보고서", "107,325"):
        assert historical not in all_pdf and historical not in tight(
            "".join(p.text for p in doc.paragraphs)
        )
    assert all(len(s) > 35 for s in pages)
    qa = dest / "rendered"
    qa.mkdir(exist_ok=True)
    for n, p in enumerate(pdf, 1):
        pix = p.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5))
        rgb = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        red, green, blue = rgb.split()
        assert not ImageChops.difference(red, green).getbbox(), (n, "red/green differ")
        assert not ImageChops.difference(green, blue).getbbox(), (
            n,
            "green/blue differ",
        )
        pix.save(qa / f"page-{n:02}.png")
    (dest / "document-checks.json").write_text(
        json.dumps(
            {
                "pages": len(pdf),
                "tables": len(printed),
                "source_tables": len(expected),
                "max_columns": max(len(p.headers) for p in printed),
                "cover_tables": 1,
                "nov23_table_format": "PASS",
                "neutral_rgb_all_pages": True,
                "figures": 1,
                "toc": index_pages,
                "features": len(FEATURE_ROWS),
                "example": "PASS",
                "table_cells_match": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    verify(parser.parse_args().root)
