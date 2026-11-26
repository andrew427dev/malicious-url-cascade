"""Read the rendered PDF and saved Word tables, comparing both to current results."""
import re
import json
from pathlib import Path
from docx import Document
from report_data import *
from report_v1 import TITLE,HEAD
import pymupdf as fitz
from PIL import Image,ImageOps,ImageDraw

def norm(s):return "".join(s.split()).replace("−","-").replace("–","-")
def main():
    ev=Evidence()
    out=ROOT/"report"; qa=ROOT/"layout-qa"
    qa.mkdir(exist_ok=True)
    doc=Document(out/(TITLE+".docx"))
    d=fitz.open(out/(TITLE+".pdf"))
    assert len(d)>=8 and norm(TITLE) in norm(d[0].get_text())
    assert TITLE in doc.paragraphs[1].text
    hs=[p.text for p in doc.paragraphs if p.style.name=="Heading 1"]
    assert hs==[str(i)+"  "+h for i,h in enumerate(HEAD,1)]
    plain=[p.get_text() for p in d]
    alltext=norm(" ".join(plain))
    toc=plain[1]
    for h in HEAD:assert norm(h) in norm(toc)
    assert not any(s in alltext for s in ("C:\\Users","오류!","Error!Reference","Error!Bookmark","outputs/final-v1"))
    for summary in ev.summary:
        assert norm(summary) in alltext
    for tab in doc.tables[1:]:
        cells=[[c.text for c in r.cells] for r in tab.rows]
        if AB[0] in cells[0]:
            assert AB[1] in cells[0]
    checked=[]
    for split in ("random","domain"):
        for scope in ("full","common"):
            expected=ev.metric_rows(split,scope,[("accuracy",100,2),("recall",100,2)])
            match=[t for t in doc.tables if len(t.rows)==len(expected)+1 and [[c.text for c in row.cells] for row in t.rows[1:]]==expected]
            assert match, (split,scope,"Word table")
            # Both procedures must be visible on the same PDF page as each pair of scores.
            for name,a,b,delta_ in expected:
                assert any(norm(a) in norm(page) and norm(b) in norm(page) and all(norm(s) in norm(page) for s in AB) for page in plain),(name,a,b)
            checked.append((split,scope,len(expected)))
    notion=json.loads((out/"notion-content.json").read_text(encoding="utf8"))
    assert notion["paragraphs"][1:]==ev.summary
    for split in ("random","domain"):
        a,b=[ev.row(k,"common","gru") for k in kinds(split)]
        assert [label(split)+" · "+num(a.n),"악성 재현율 (%)",pct(a.recall),pct(b.recall),delta(100*(b.recall-a.recall))+"%p"] in notion["table"]
    # preserve full page images and make readable overviews for layout review
    for i,page in enumerate(d,1):
        page.get_pixmap(matrix=fitz.Matrix(1.35,1.35),alpha=False).save(qa/f"page-{i:02d}.png")
    for offset in range(0,len(d),6):
        thumbnails=[]
        for i in range(offset+1,min(offset+7,len(d)+1)):
            im=Image.open(qa/f"page-{i:02d}.png").convert("RGB")
            im.thumbnail((332,470));thumbnails.append(im)
        sheet=Image.new("RGB",(3*344,2*485),"#d0d0d0")
        for j,im in enumerate(thumbnails):sheet.paste(im,(j%3*344+5,j//3*485+4))
        sheet.save(qa/f"overview-{offset+1:02d}.png")
    print(json.dumps(dict(pages=len(d),tables=len(doc.tables),figures=len(doc.inline_shapes),score_pairs=checked),ensure_ascii=False))
    (qa/"report-numbers-verified.json").write_text(json.dumps(dict(pages=len(d),tables=len(doc.tables),figures=len(doc.inline_shapes),score_pairs=checked,notion_summary=True),indent=2),encoding="utf8")
if __name__=="__main__":main()
