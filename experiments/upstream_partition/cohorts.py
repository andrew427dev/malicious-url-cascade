"""Build procedure-specific cohorts on preassigned raw URLs."""
import argparse
import gc
import hashlib
import json
import re
from collections import defaultdict

from inputs import ROOT, PREFIX, tf, np, pd, read_frames, prepared, save_parts, encode, vectorizer, extract_features, preprocess_link, counts, write_json
from prepare import normalize

ROLES = ("train", "validation", "test")
SPLITS = ("random", "domain")
def model_url(raw, form, team):
    value = PREFIX.sub("", raw, count=1) if form == "stripped" else raw
    return normalize(value)[0] if team else value


def conflicts(d, col):
    n = d.groupby(col, sort=False).type.nunique()
    return d[col].isin(n[n > 1].index)


def remove(d, mask, reason, ledger):
    gone = d.loc[mask]
    ledger.append(dict(rule=reason, removed=counts(gone), remaining=len(d)-len(gone)))
    return d.loc[~mask].copy()


def build(form, team):
    raw = pd.concat(read_frames())
    assignment = pd.read_pickle(ROOT/"assignment.pkl")
    raw = raw.join(assignment.drop(columns="y_mal"), on="url", validate="many_to_one")
    assert raw.random_role.notna().all()
    ledger = []
    raw_bad = set(raw.loc[conflicts(raw,"url"),"url"])
    if team:
        canon = {}
        for url in raw.url.unique():
            try:
                canon[url] = model_url(url,form,True)
            except (ValueError, UnicodeError):
                canon[url] = None
        raw["input_url"] = raw.url.map(canon)
        del canon
        raw = remove(raw, raw.input_url.isna(), "invalid_url",ledger)
        normalized_bad = set(raw.loc[conflicts(raw,"input_url"),"input_url"])
    raw = remove(raw, raw.url.isin(raw_bad), "raw_label_conflict",ledger)
    raw = remove(raw, raw.duplicated(["url","type"]), "identical_raw_label_repeat",ledger)
    if not team:
        for split in SPLITS:
            out = ROOT/(split+"-"+form)
            parts = {r:raw.loc[raw[split+"_role"]==r].sort_values(split+"_order")[["url","type","domain"]] for r in ROLES}
            save_parts(out,parts,form=="stripped")
            extra = []
            for role,d in parts.items():
                value = d.url.map(lambda u: model_url(u,form,False)).map(preprocess_link)
                mixed = value.map(d.type.groupby(value).nunique()).gt(1)
                duplicates = value.duplicated()
                extra.append(dict(role=role, mixed_text_groups=int(value[mixed].nunique()),mixed_text_removed=int((mixed & duplicates).sum()),same_label_text_removed=int((~mixed & duplicates).sum()),remaining=int((~duplicates).sum())))
            write_json(out/"cleaning.json",dict(stages=ledger,first_text=extra))
        return
    raw = remove(raw,raw.input_url.isin(normalized_bad),"normalized_label_conflict",ledger)
    raw = remove(raw,raw.input_url.duplicated(),"normalized_repeat",ledger)
    values, representations, valid, texts = [],[],[],[]
    for u in raw.input_url:
        try:
            x = extract_features(u)
            t = preprocess_link(u)
            if not t:
                raise ValueError("empty_model_text")
            k = hashlib.sha256((t[:52]+"|"+json.dumps(x)).encode()).digest()
        except (ValueError,TypeError,re.error):
            x,t,k=None,"",None
        if x is not None:
            values.append(x)
        texts.append(t)
        representations.append(k)
        valid.append(x is not None)
    raw["text"], raw["rep"] = texts, representations
    raw = remove(raw, ~np.array(valid), "unusable_input",ledger)
    x = np.asarray(values,np.float32)
    del values,representations,texts
    raw["pos"] = np.arange(len(raw))
    raw = remove(raw,conflicts(raw,"rep"),"character_numeric_label_conflict",ledger)
    raw = remove(raw,raw.rep.duplicated(),"character_numeric_repeat",ledger)
    raw["canonical_hash"] = [hashlib.sha256(u.encode()).hexdigest() for u in raw.input_url]
    raw = raw.sort_values("canonical_hash")
    raw = raw.drop(columns=["rep"])
    raw,x = token_audit(raw,x,ledger)
    for split in SPLITS:
        out = ROOT/("team-"+split+"-"+form)
        out.mkdir(exist_ok=False)
        layer, tokens = fresh_tokens(raw,split)
        keys = {hashlib.sha256(a.tobytes()+b.tobytes()).digest() for a,b in zip(tokens,x)}
        assert len(keys)==len(raw)
        write_json(out/"vocab.json",layer.get_vocabulary())
        report={}
        for role in ROLES:
            idx=np.flatnonzero(raw[split+"_role"].to_numpy()==role)
            d=raw.iloc[idx][["url","type","text","domain","input_url"]]
            d.to_pickle(out/(role+".pkl"))
            d[["type"]].assign(y_mal=1-d.type).to_csv(out/(role+"-rows.csv.gz"))
            np.save(out/(role+"-tokens.npy"),tokens[idx])
            np.save(out/(role+"-features.npy"),x[idx])
            report[role]=counts(d)
        write_json(out/"cleaning.json",dict(stages=ledger,final=report))
        del tokens,keys,layer
        gc.collect()


def fresh_tokens(d,split):
    layer=vectorizer()
    layer.adapt(tf.data.Dataset.from_tensor_slices(d.loc[d[split+"_role"]=="train","text"].tolist()).batch(2048))
    return layer,encode(d.text,layer)


def token_audit(d,x,ledger):
    x=x[d.pos.to_numpy()]
    for round_number in range(1,21):
        parent={}
        def find(i):
            p=parent.setdefault(i,i)
            if p!=i:
                parent[i]=find(p)
            return parent[i]
        for split in SPLITS:
            layer,z=fresh_tokens(d,split)
            seen={}
            for i,(a,b) in enumerate(zip(z,x)):
                k=hashlib.sha256(a.tobytes()+b.tobytes()).digest()
                if k in seen:
                    j=find(seen[k]); q=find(i)
                    parent[max(j,q)]=min(j,q)
                else:
                    seen[k]=i
            del seen,z,layer
        comps=defaultdict(list)
        for i in parent:
            comps[find(i)].append(i)
        bad=np.zeros(len(d),bool)
        dupe=bad.copy()
        labels=d.type.to_numpy()
        for group in comps.values():
            g=sorted(group)
            if len(set(labels[g]))>1:
                bad[g]=True
            else:
                dupe[g[1:]]=True
        ledger.append(dict(rule="token_round_"+str(round_number),conflict=int(bad.sum()),repeat=int(dupe.sum()),remaining=int((~(bad|dupe)).sum())))
        if not (bad|dupe).any():
            break
        keep=~(bad|dupe)
        d,x=d.iloc[np.flatnonzero(keep)],x[keep]
    else:
        raise RuntimeError("Token refinement failed to converge")
    stable=np.array([normalize(u)[0]==u for u in d.input_url])
    return remove(d,~stable,"unstable_normalization",ledger),x[stable]


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("procedure",choices=["reference","team"])
    parser.add_argument("form",choices=["original","stripped"])
    a=parser.parse_args()
    build(a.form,a.procedure=="team")
