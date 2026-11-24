"""Verify ordered headings, TOC page numbers, tables, figure and feature example."""

import argparse
import json
import re
from pathlib import Path

from docx import Document
import pymupdf
from feature_table import FEATURE_ROWS, example
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
    assert actual == CHAPTERS
    index_pages = {}
    for i, title in enumerate(CHAPTERS, 1):
        hits = [n for n, s in enumerate(pages[1:], 2) if tight(title) in s]
        assert len(hits) == 1, (title, hits)
        p = hits[0]
        assert tight(f"{i}{title}{p}") in pages[0], (title, p)
        index_pages[title] = p
    assert list(index_pages.values()) == sorted(index_pages.values())
    content = json.loads((dest / "content.json").read_text(encoding="utf8"))
    expected = [
        (caption, heads, rows)
        for page in content
        for typ, *payload in page["blocks"]
        if typ == "table"
        for caption, heads, rows in [payload]
    ]
    assert len(doc.tables) == len(expected)
    all_pdf = "".join(pages)
    for tab, (caption, headers, rows) in zip(doc.tables, expected):
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
        title, _, rows = section[0]
        # Word exports a title once and writes rows in order. Limit the PDF check
        # to this table, including any continuation up to the next table caption.
        start = all_pdf.index(tight(title)) + len(tight(title))
        end = min(
            [
                i
                for other, _, _ in expected
                if (i := all_pdf.find(tight(other), start)) >= 0
            ],
            default=len(all_pdf),
        )
        return {tuple(r) for r in rows}, all_pdf[start:end]

    for r in perf:
        data_cells, pdf_section = matching_table(r["condition"])
        cells = [
            POLICY[r["policy"]],
            *[f"{100 * float(r[k]):.2f}" for k in ("accuracy", "recall", "fpr")],
            *[f"{int(r[k]):,}" for k in ("fn", "additional_misses", "t2_calls")],
            f"{100 * float(r['t2_call_rate']):.2f}",
        ]
        assert tuple(cells) in data_cells, (r["condition"], r["policy"])
        assert tight("".join(cells)) in pdf_section, (
            r["condition"],
            r["policy"],
            "PDF performance",
        )
    for r in timing:
        data_cells, pdf_section = matching_table(r["condition"], timing=True)
        cells = [
            POLICY[r["policy"]],
            f"{int(r['calls']):,}",
            f"{float(r['calls_saved_pct']):.1f}",
            f"{float(r['ms']):.2f}",
            f"{float(r['min_ms']):.2f}–{float(r['max_ms']):.2f}",
            f"{float(r['time_saved_pct']):.2f}",
        ]
        assert tuple(cells) in data_cells
        assert tight("".join(cells)) in pdf_section, (
            r["condition"],
            r["policy"],
            "PDF timing",
        )
    setup = pages[index_pages["System Setup"] - 1 : index_pages[CHAPTERS[3]] - 1]
    setup_text = "".join(setup)
    frag = next(t for t in doc.tables if t.cell(0, 1).text == "원저자 열 이름")
    assert [(r.cells[1].text, r.cells[2].text) for r in frag.rows[1:]] == FEATURE_ROWS
    for n, definition in FEATURE_ROWS:
        assert tight(n) in setup_text and tight(definition) in setup_text, n
    assert tight(example(root)) in setup_text
    assert any(p.text == example(root) for p in doc.paragraphs)
    assert all(len(s) > 35 for s in pages)
    qa = dest / "rendered"
    qa.mkdir(exist_ok=True)
    for n, p in enumerate(pdf, 1):
        pix = p.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5))
        pix.save(qa / f"page-{n:02}.png")
    (dest / "document-checks.json").write_text(
        json.dumps(
            {
                "pages": len(pdf),
                "tables": len(expected),
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
