"""Rebuild the expected ordered cohort using the original implementation."""

import argparse
from itertools import combinations
from inputs import (
    ROOT,
    PREFIX,
    DataProcessing,
    FeatureCreation,
    FEATURE_NAMES,
    read_frames,
    vectorizer,
    tf,
    sha,
    write_json,
    np,
    pd,
)
from evaluate import read


def audit(kind):
    pool = pd.read_pickle(ROOT / "team-pool.pkl")
    out = ROOT / kind
    group, form = kind.split("-")
    data = read(out / "data.json")
    parts = {}
    source = pd.concat(read_frames())
    vocab = read(out / "vocab.json")
    layer = vectorizer(vocab)
    for part in ("train", "validation", "test"):
        assigned = pd.read_csv(ROOT / (group + "-" + part + "-assigned.csv.gz")).id
        df = pool.loc[assigned]
        pd.testing.assert_frame_equal(
            df[["url", "type"]], source.loc[df.index, ["url", "type"]]
        )

        raw = (
            df.url.str.replace(PREFIX, "", n=1, regex=True)
            if form == "stripped"
            else df.url
        )
        text = raw.map(DataProcessing.preprocess_link)
        k = ~text.duplicated()
        expected = df.loc[k, ["url", "type", "domain"]].assign(text=text[k])
        saved = pd.read_pickle(out / (part + ".pkl"))
        pd.testing.assert_frame_equal(
            saved[["url", "type", "text"]], expected[["url", "type", "text"]]
        )
        # Domain and role are distinct. Always derive the audit from the registered domain.
        pd.testing.assert_series_equal(saved.domain, expected.domain)
        assert (df[group + "_role"] == part).all()
        rows = pd.read_csv(out / (part + "-rows.csv.gz"), index_col="id")
        assert rows.index.equals(expected.index) and np.array_equal(
            rows.type, expected.type
        )
        assert expected.text.is_unique and expected.index.is_unique
        np.testing.assert_array_equal(rows.y_mal, 1 - expected.type)
        if part == "train":
            learned = vectorizer()
            learned.adapt(
                tf.data.Dataset.from_tensor_slices(expected.text.tolist()).batch(128)
            )
            assert learned.get_vocabulary() == vocab
        numeric = np.load(out / (part + "-features.npy"), mmap_mode="r")
        tokens = np.load(out / (part + "-tokens.npy"), mmap_mode="r")
        assert numeric.shape == (len(expected), 18) and tokens.shape == (
            len(expected),
            50,
        )
        represented = raw[k]
        for start in range(0, len(expected), 4096):
            chunk = slice(start, start + 4096)
            batch = pd.DataFrame(
                {"url": represented.iloc[chunk], "type": expected.type.iloc[chunk]}
            )
            ref = FeatureCreation.create_features(batch)
            assert tuple(ref.columns[2:]) == FEATURE_NAMES
            np.testing.assert_array_equal(ref.iloc[:, 2:].to_numpy(), numeric[chunk])
            paired, _ = DataProcessing.vectorize_text(
                tf.constant(expected.text.iloc[chunk].tolist()),
                numeric[chunk],
                rows.type.iloc[chunk],
                layer,
            )
            np.testing.assert_array_equal(paired[0].numpy(), tokens[chunk])
        parts[part] = expected
    for a, b in combinations(parts, 2):
        overlap = {
            c: len(set(parts[a][c]) & set(parts[b][c]))
            for c in ("url", "text", "domain")
        }
        assert data["overlaps"][a + "/" + b] == overlap
        assert not overlap["url"] and (group != "domain" or not overlap["domain"])
    hashes = {
        p.name: sha(p)
        for p in out.iterdir()
        if p.suffix in (".npy", ".gz", ".pkl") or p.name in ("vocab.json", "data.json")
    }
    write_json(
        out / "full-input-audit.json",
        {
            "status": "PASS",
            "files": hashes,
            "rows_rebuilt_from_source": {p: len(f) for p, f in parts.items()},
            "registered_domains": {
                p: int(f.domain.nunique()) for p, f in parts.items()
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind")
    audit(parser.parse_args().kind)
