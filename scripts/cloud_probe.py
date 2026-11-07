"""W1 diagnostics; stdlib inventory first, optional synthetic ML checks."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from urlcascade.runtime import compute_context

PINS = {'tensorflow':'2.12.0','numpy':'1.23.5','pandas':'1.5.3',
        'scikit-learn':'1.2.2','tldextract':'5.4.0'}


def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return dict(argv=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(argv=argv, error=str(exc))


def inventory(runtime, work):
    versions = {}
    for package in PINS:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(runtime=runtime, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                python=sys.version, executable=sys.executable, platform=platform.platform(),
                machine=platform.machine(), cpu_count=os.cpu_count(),
                memory=Path('/proc/meminfo').read_text() if Path('/proc/meminfo').exists() else None,
                workspace_free_bytes=shutil.disk_usage(work).free, versions=versions,
                expected_versions=PINS,
                gpu=command(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv']),
                nvcc=command(['nvcc','--version']),
                account_quota='Not detectable here; inspect account UI separately.')


def check_ml(report, require_gpu):
    if sys.version_info[:2] != (3,10) or report['versions'] != PINS:
        raise RuntimeError('pinned_Python_3.10_and_package_versions_required')
    pip_check = command([sys.executable,'-m','pip','check'])
    report['pip_check'] = pip_check
    if pip_check.get('returncode') != 0:
        raise RuntimeError('pip_check_failed')
    import numpy as np
    import pandas
    import sklearn
    import tldextract
    import tensorflow as tf
    from urlcascade.features import extract_features
    from urlcascade.adapters import vectorizer, preprocess_link
    tf.keras.utils.set_random_seed(42)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    report['tensorflow_build'] = tf.sysconfig.get_build_info()
    report['visible_gpus'] = [str(gpu) for gpu in gpus]
    if require_gpu and not gpus:
        raise RuntimeError('requested_GPU_not_visible')
    if gpus:
        with tf.device('/GPU:0'):
            matrix = tf.matmul(tf.ones((4,4)), tf.ones((4,4)))
        report['gpu_matmul'] = dict(device=matrix.device, sum=float(matrix.numpy().sum()))
        if require_gpu and 'GPU' not in matrix.device:
            raise RuntimeError('matmul_not_on_GPU')
    spec = importlib.util.spec_from_file_location('pinned_model', ROOT/'upstream/Utils/Model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.create_model(50,10000,18)
    if model.count_params() != 162481:
        raise AssertionError('original_model_parameter_count_changed')
    urls = ['https://example.com/a','https://example.org/b','https://example.net/c']
    layer = vectorizer()
    text = tf.constant([preprocess_link(u) for u in urls])
    layer.adapt(text)
    output = model([layer(text),np.asarray([extract_features(u) for u in urls],dtype=np.float32)],training=False)
    values = output.numpy()
    if values.shape != (3,1) or not np.isfinite(values).all():
        raise AssertionError('invalid_synthetic_forward')
    report['synthetic_model'] = dict(parameters=model.count_params(),output_shape=list(values.shape),
                                    output_device=output.device,training_performed=False)
    report['gpu_model_speed'] = 'Not established; matmul and final output placement do not prove all GRU operations or speed.'
    report['status'] = 'ml_gpu_checks_passed' if require_gpu else 'ml_checks_passed'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime',choices=['kaggle','colab'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--check-ml',action='store_true')
    p.add_argument('--require-gpu',action='store_true')
    args = p.parse_args()
    runtime = compute_context(args.runtime,args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    report = inventory(runtime,args.out.parent)
    report['status'] = 'inventory_only'
    rc = 0
    try:
        if args.require_gpu and not args.check_ml:
            raise ValueError('require_gpu_needs_check_ml')
        if args.check_ml:
            check_ml(report,args.require_gpu)
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        rc = 1
    args.out.write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps(report,indent=2,default=str))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
