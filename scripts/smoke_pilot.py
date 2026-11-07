"""W3 only: development models, never final A/B weights. Requires a frozen B split."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from urlcascade.data import sha256
from urlcascade.features import FEATURE_NAMES, extract_features
from urlcascade.adapters import preprocess_link, vectorizer, KerasT2
from urlcascade.runtime import compute_context


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['smoke','pilot'], required=True)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--require-gpu', action='store_true')
    p.add_argument('--runtime', choices=['auto','kaggle','colab','school'], default='auto')
    args = p.parse_args()
    runtime = compute_context(args.runtime, args.out)
    if args.batch_size < 1:
        raise ValueError('invalid_batch')
    manifest_path = args.split_dir/'manifest.json'
    split = json.loads(manifest_path.read_text())
    if split.get('status') != 'frozen' or split.get('experiment') != 'B':
        raise ValueError('frozen_B_split_required')
    for name, digest in split['files'].items():
        if sha256(args.split_dir/name) != digest:
            raise ValueError('split_integrity_failed:' + name)
    rows = [json.loads(line) for line in (args.split_dir/'train.jsonl').read_text().splitlines()]
    if any(r['split'] != 'train' for r in rows) or {r['y_original'] for r in rows} != {0, 1}:
        raise ValueError('invalid_B_training_rows')
    rows.sort(key=lambda r:r['input_id'])
    maximum, epochs = (2000,1) if args.stage == 'smoke' else (50000,2)
    rows = random.Random(42).sample(rows,min(maximum,len(rows)))
    if {r['y_original'] for r in rows} != {0, 1}:
        raise ValueError('development_sample_missing_class')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'selected-ids.json').write_text(json.dumps([r['input_id'] for r in rows]))
    import numpy as np
    import tensorflow as tf
    if tf.__version__ != '2.12.0':
        raise RuntimeError('TensorFlow_2.12.0_required')
    tf.keras.utils.set_random_seed(42)
    gpus = tf.config.list_physical_devices('GPU')
    for device in gpus:
        tf.config.experimental.set_memory_growth(device, True)
    if args.require_gpu and not gpus:
        raise RuntimeError('requested_GPU_not_visible')
    gpu_probe = None
    if gpus:
        with tf.device('/GPU:0'):
            probe = tf.matmul(tf.ones((4,4)),tf.ones((4,4)))
        gpu_probe = dict(device=probe.device, value=probe.numpy().tolist())
        if args.require_gpu and 'GPU' not in probe.device:
            raise RuntimeError('GPU_operation_not_placed_on_GPU')
    source = ROOT/'upstream/Utils/Model.py'
    spec = importlib.util.spec_from_file_location('pinned_model', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = [preprocess_link(r['raw_url']) for r in rows]
    features = np.asarray([extract_features(r['raw_url']) for r in rows],dtype=np.float32)
    y = np.asarray([r['y_original'] for r in rows],dtype=np.float32)
    vl = vectorizer()
    vl.adapt(tf.data.Dataset.from_tensor_slices(text).batch(args.batch_size))
    encoded = vl(tf.constant(text))
    vocab = vl.get_vocabulary()
    (args.out/'vocab.json').write_text(json.dumps(vocab))
    (args.out/'development-config.json').write_text(json.dumps(dict(
        stage=args.stage,development_only=True,seed=42,n=len(rows),epochs=epochs,
        batch_size=args.batch_size,runtime=runtime,split_manifest_sha256=sha256(manifest_path),
        resume_support='Exact stochastic resume not implemented; restart development run.'),indent=2))
    model = module.create_model(50,10000,18)
    if model.count_params() != 162481:
        raise AssertionError('original_parameter_count_changed')
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    epoch_times = []
    class Timer(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.start = time.perf_counter()
        def on_epoch_end(self, epoch, logs=None):
            epoch_times.append(time.perf_counter()-self.start)
    history = model.fit([encoded,features],y,batch_size=args.batch_size,
                        epochs=epochs,shuffle=True,callbacks=[Timer(),
                        tf.keras.callbacks.CSVLogger(str(args.out/'epochs.csv')),
                        tf.keras.callbacks.ModelCheckpoint(str(args.out/'epoch-{epoch:02d}.h5'))])
    model.save(args.out/'development-only.h5')
    reload_model = tf.keras.models.load_model(args.out/'development-only.h5',compile=False)
    check_rows = rows[:min(32,len(rows))]
    before = model([encoded[:len(check_rows)],features[:len(check_rows)]],training=False).numpy().reshape(-1)
    adapter = KerasT2(reload_model,vocab)
    after = np.asarray(adapter([r['raw_url'] for r in check_rows],features[:len(check_rows)]))
    decision_equal = bool(np.array_equal(before < .5,after < .5))
    if not decision_equal or not np.allclose(before,after,rtol=1e-6,atol=1e-7):
        raise AssertionError('reload_mismatch')
    gpu_memory = {}
    for index in range(len(gpus)):
        try:
            gpu_memory[str(index)] = tf.config.experimental.get_memory_info('GPU:'+str(index))
        except (ValueError, RuntimeError) as exc:
            gpu_memory[str(index)] = dict(unavailable=str(exc))
    report = dict(stage=args.stage,development_only=True,source_commit=split['source_commit'],
                  split_manifest_sha256=sha256(manifest_path),
                  selected_ids=[r['input_id'] for r in rows],seed=42,n=len(rows),epochs=epochs,
                  batch_size=args.batch_size,epoch_seconds=epoch_times,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  tensorflow_gpu_memory_bytes=gpu_memory,
                  host=socket.gethostname(),runtime=runtime,job_id=os.environ.get('SLURM_JOB_ID'),
                  resume_support='Checkpoints retained; exact stochastic resume not implemented. Restart development run.',
                  python=sys.version,tensorflow=tf.__version__,build=tf.sysconfig.get_build_info(),
                  visible_gpus=[str(g) for g in gpus],gpu_matmul_probe=gpu_probe,
                  model_gpu_placement='Not established by device visibility or matmul alone; profile original GRU separately.',
                  parameter_count=model.count_params(),feature_names=FEATURE_NAMES,
                  gru=dict(dropout=.2,recurrent_dropout=.2),history=history.history,
                  reload_decisions_equal=decision_equal,max_abs_reload_diff=float(np.max(np.abs(before-after))),
                  near_boundary_count=int(np.sum(np.abs(before-.5)<1e-6)),
                  weights_sha256=sha256(args.out/'development-only.h5'),vocab_sha256=sha256(args.out/'vocab.json'))
    (args.out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('selected_ids','history')},indent=2,default=str))


if __name__ == '__main__':
    main()
