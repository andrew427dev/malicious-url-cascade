"""Render every PDF page to PNG for layout checks (report/layout-qa/page-NN.png)."""
import sys
from pathlib import Path

pdf = Path(sys.argv[1])
if not pdf.is_file():
    raise SystemExit(f"refused[missing_pdf]: {pdf} is not a file")
from v2paths import refuse_protected

dest = refuse_protected(pdf.parent / "layout-qa")
import pymupdf

doc = pymupdf.open(pdf)
dest.mkdir(exist_ok=True)
for old in dest.glob("page-*.png"):
    old.unlink()
for i, page in enumerate(doc, 1):
    page.get_pixmap(dpi=int(sys.argv[2]) if len(sys.argv) > 2 else 80).save(dest / f"page-{i:02d}.png")
print(len(doc), "pages")
