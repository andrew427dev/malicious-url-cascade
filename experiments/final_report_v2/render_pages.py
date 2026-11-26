"""Render every PDF page to PNG for layout checks (report/layout-qa/page-NN.png)."""
import sys
from pathlib import Path

import pymupdf

pdf = Path(sys.argv[1])
dest = pdf.parent / "layout-qa"
dest.mkdir(exist_ok=True)
for old in dest.glob("page-*.png"):
    old.unlink()
doc = pymupdf.open(pdf)
for i, page in enumerate(doc, 1):
    page.get_pixmap(dpi=int(sys.argv[2]) if len(sys.argv) > 2 else 80).save(dest / f"page-{i:02d}.png")
print(len(doc), "pages")
