"""Pack exact new input arrays and their ordered labels without pickle dependencies."""
import json,hashlib,os
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
ROOT=Path(os.environ["URLCASCADE_RUN_DIR"])
sets={"원저자 재현":["author-validation","author-final"],"기본 표기":[p+s+"-original" for p in ("","team-") for s in ("random","domain")],"접두어 제거":[p+s+"-stripped" for p in ("","team-") for s in ("random","domain")]}
def hash_stream(f):
    h=hashlib.sha256()
    for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()
def main():
    for name,kinds in sets.items():
        dest=ROOT/"share"/(name+" 학습·평가 입력 v1.zip")
        hashes={}
        with ZipFile(dest,"x",compression=ZIP_DEFLATED,compresslevel=3) as z:
            for kind in kinds:
                source=ROOT/kind
                for p in [*source.glob("*-tokens.npy"),*source.glob("*-features.npy"),*source.glob("*-rows.csv.gz"),source/"vocab.json"]:
                    key=p.relative_to(ROOT).as_posix()
                    with p.open("rb") as f:hashes[key]=hash_stream(f)
                    z.write(p,key)
            z.writestr("SHA256.json",json.dumps(hashes,indent=2))
        with ZipFile(dest) as z:
            for path,h in hashes.items():
                with z.open(path) as f:assert hash_stream(f)==h
        print(dest.name,dest.stat().st_size,flush=True)
if __name__=="__main__":main()
