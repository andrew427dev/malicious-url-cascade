"""Reload the previously saved smoke model in a separate Kaggle session."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if hashlib.sha256(args.archive.read_bytes()).hexdigest()!=args.sha256:
        raise ValueError('restore archive SHA-256 mismatch')
    from urlcascade.adapters import KerasT2
    from urlcascade.features import extract_features
    import numpy as np
    import tensorflow as tf
    with tempfile.TemporaryDirectory() as temp, zipfile.ZipFile(args.archive) as archive:
        names={'development-only.h5','vocab.json','restore-fixture.json'}
        if set(archive.namelist())!=names:
            raise ValueError('unexpected restore archive members')
        root=Path(temp)
        for name in names:
            (root/name).write_bytes(archive.read(name))
        expected=json.loads((root/'restore-fixture.json').read_text())
        for name,key in [('development-only.h5','weights_sha256'),('vocab.json','vocab_sha256')]:
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected[key]:
                raise ValueError('hash mismatch: '+name)
        model=tf.keras.models.load_model(root/'development-only.h5',compile=False)
        adapter=KerasT2(model,json.loads((root/'vocab.json').read_text()))
        urls=expected['urls']
        scores=np.array(adapter(urls,[extract_features(url) for url in urls]))
        before=np.array(expected['safe_scores'])
        np.testing.assert_allclose(scores,before,rtol=1e-6,atol=1e-7)
        assert np.array_equal(scores<0.5,before<0.5)
        record=dict(status='passed',archive_sha256=args.sha256,weights_sha256=expected['weights_sha256'],
            vocabulary_sha256=expected['vocab_sha256'],tensorflow=tf.__version__,probe_rows=len(urls),
            max_abs_diff=float(np.abs(scores-before).max()),decisions_equal=True)
    args.out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':
    main()
