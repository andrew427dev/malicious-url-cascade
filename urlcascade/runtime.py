"""Explicit compute contexts; never permit training on the school login node."""
from pathlib import Path
import os
import platform
import socket


def runtime_facts():
    return dict(system=platform.system(), host=socket.gethostname(),
                kaggle=(Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir()),
                colab=(Path('/content').is_dir() and bool(
                    os.environ.get('COLAB_RELEASE_TAG') or os.environ.get('COLAB_BACKEND_VERSION'))),
                slurm_job=bool(os.environ.get('SLURM_JOB_ID')))


def select_runtime(requested='auto', facts=None):
    facts = runtime_facts() if facts is None else facts
    if facts['host'].split('.')[0] == 'cscloud-master':
        raise RuntimeError('compute_on_school_login_node_forbidden')
    if facts['system'] != 'Linux':
        raise RuntimeError('compute_requires_verified_Linux_runtime')
    detected = ('kaggle' if facts['kaggle'] else 'colab' if facts['colab'] else
                'school' if facts['slurm_job'] else None)
    if not detected or requested not in ('auto', detected):
        raise RuntimeError('runtime_not_verified:' + requested)
    return detected


def writable_path(path, runtime):
    path = Path(path).resolve()
    roots = {'kaggle': Path('/kaggle/working'), 'colab': Path('/content'),
             'school': Path.home()/'workspace'}
    root = roots[runtime].resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('output_must_be_inside_project_workspace:' + str(path))
    return path


def compute_context(requested, output):
    runtime = select_runtime(requested)
    writable_path(output, runtime)
    return runtime
