"""Select B's T1 and T2 using only the frozen train and model-selection files."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'upstream')]
from urlcascade.adapters import preprocess_link, vectorizer
from urlcascade.data import sha256
from urlcascade.features import FEATURE_NAMES, extract_features
from urlcascade.runtime import compute_context

SPLIT_SHA256 = '6440ae2948353ea42d5adb70a0e7a686d592496da94a8d24828707a0c6eef955'
TOLERANCE = 1e-6


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_partition(directory, manifest, name):
    import numpy as np

    path = directory / (name + '.jsonl')
    if sha256(path) != manifest['files'][path.name]:
        raise ValueError(f'Frozen partition hash changed: {name}')
    count = manifest['summary'][name]['rows']
    numeric = np.empty((count, len(FEATURE_NAMES)), dtype=np.float32)
    safe_labels = np.empty(count, dtype=np.int32)
    texts = []
    with path.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if row['split'] != name or row['y_mal'] != 1 - row['y_original']:
                raise ValueError(f'Invalid frozen row {name}/{index}')
            numeric[index] = extract_features(row['raw_url'])
            safe_labels[index] = row['y_original']
            texts.append(preprocess_link(row['raw_url']))
    if len(texts) != count or set(np.unique(safe_labels)) != {0, 1}:
        raise ValueError(f'Invalid row or class count in {name}')
    return numeric, safe_labels, texts


def select_logistic(train, validation, out):
    import joblib
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train[0])
    x_train = scaler.transform(train[0])
    x_val = scaler.transform(validation[0])
    attempts = []
    for max_iter in (2000, 5000):
        models, scores, converged = {}, {}, True
        for c in (0.1, 1.0, 10.0):
            model = LogisticRegression(C=c, solver='lbfgs', penalty='l2', max_iter=max_iter,
                                       tol=1e-4, class_weight=None, random_state=42)
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(x_train, 1 - train[1])
            messages = [str(item.message) for item in observed]
            failed = any(issubclass(item.category, ConvergenceWarning) for item in observed)
            converged = converged and not failed
            malicious_column = list(model.classes_).index(1)
            scores[c] = float(average_precision_score(1 - validation[1],
                              model.predict_proba(x_val)[:, malicious_column]))
            models[c] = model
            attempts.append(dict(C=c, max_iter=max_iter, ap=scores[c],
                                 iterations=int(np.max(model.n_iter_)), warnings=messages))
        write_json(out / 't1-candidates.json', attempts)
        if converged:
            break
    else:
        raise RuntimeError('Logistic regression did not converge for every C')
    best = min(c for c, ap in scores.items() if max(scores.values()) - ap <= TOLERANCE)
    pipeline = Pipeline([('scaler', scaler), ('logistic', models[best])])
    joblib.dump(pipeline, out / 't1.joblib')
    return best


def select_gru(train, validation, out):
    import numpy as np
    import tensorflow as tf
    from Utils.Model import create_model

    tf.keras.utils.set_random_seed(42)
    batch = 128
    layer = vectorizer()
    text_train = tf.data.Dataset.from_tensor_slices(train[2]).batch(batch)
    layer.adapt(text_train)
    write_json(out / 'b-vocab.json', layer.get_vocabulary())

    def encode(text):
        dataset = tf.data.Dataset.from_tensor_slices(text).batch(batch)
        return np.concatenate([layer(block).numpy().astype(np.int32) for block in dataset])

    x_train = [encode(train[2]), train[0]]
    x_val = [encode(validation[2]), validation[0]]
    model = create_model(50, 10000, 18)
    if model.count_params() != 162481:
        raise ValueError('Original architecture changed')
    model.compile(loss='binary_crossentropy', optimizer='adam')

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.start = time.monotonic()

        def on_epoch_end(self, epoch, logs=None):
            elapsed = time.monotonic() - self.start
            print('B_EPOCH', epoch + 1, 'seconds', elapsed, logs, flush=True)
            if epoch == 0 and elapsed * 20 * 1.5 > 4 * 3600:
                write_json(out / 'resource-review.json', dict(seconds=elapsed, epoch=1))
                self.model.stop_training = True

    with tempfile.TemporaryDirectory(dir=out) as temp:
        checkpoints = Path(temp)
        progress = model.fit(x_train, train[1], validation_data=(x_val, validation[1]),
            epochs=20, batch_size=batch, shuffle=True, verbose=2,
            callbacks=[Progress(), tf.keras.callbacks.CSVLogger(str(out / 'b-epochs.csv')),
                       tf.keras.callbacks.ModelCheckpoint(str(checkpoints / 'epoch-{epoch:02d}.h5'))])
        losses = np.asarray(progress.history['val_loss'])
        if len(losses) != 20 or not np.isfinite(losses).all():
            if len(losses):
                last = checkpoints / f'epoch-{len(losses):02d}.h5'
                if last.exists():
                    last.replace(out / 'partial-t2.h5')
            raise RuntimeError('Incomplete B run. A final model was not selected.')
        best = int(np.flatnonzero(losses <= losses.min() + TOLERANCE)[0]) + 1
        (checkpoints / f'epoch-{best:02d}.h5').replace(out / 't2.h5')
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runtime', choices=['kaggle', 'colab'], required=True)
    args = parser.parse_args()
    compute_context(args.runtime, args.out)
    import sklearn
    import tensorflow as tf
    if sklearn.__version__ != '1.2.2' or tf.__version__ != '2.12.0':
        raise RuntimeError('B requires the pinned environment')
    manifest = args.split_dir / 'manifest.json'
    if sha256(manifest) != SPLIT_SHA256:
        raise ValueError('Expected the verified W2 split manifest')
    spec = json.loads(manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    train = load_partition(args.split_dir, spec, 'train')
    validation = load_partition(args.split_dir, spec, 'model_selection')
    best_c = select_logistic(train, validation, args.out)
    best_epoch = select_gru(train, validation, args.out)
    write_json(args.out / 'models-frozen.json', dict(
        experiment='B', source_commit=spec['source_commit'], split_sha256=SPLIT_SHA256,
        feature_names=FEATURE_NAMES, fit_partition='train', selection_partition='model_selection',
        scaler_fit_partition='train', vocabulary_fit_partition='train', seed=42, batch_size=128,
        selected_C=best_c, selected_epoch=best_epoch, malicious_class_t1=1,
        t2_safe_threshold=0.5, t2_malicious_rule='s_safe < 0.5', gate_status='unselected',
        files={name: sha256(args.out / name) for name in ('t1.joblib', 't2.h5', 'b-vocab.json',
                                                            't1-candidates.json', 'b-epochs.csv')}))


if __name__ == '__main__':
    main()
