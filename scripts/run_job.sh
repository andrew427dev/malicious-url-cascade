#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
test -n "${SLURM_JOB_ID:-}" || { echo 'Use jobsubmit, not the login shell.' >&2; exit 2; }
test "$(hostname)" != cscloud-master || { echo 'Refusing compute on login node.' >&2; exit 2; }
mkdir -p cache tmp artifacts
export TMPDIR="$project_root/tmp"
export XDG_CACHE_HOME="$project_root/cache"
export KERAS_HOME="$project_root/cache/keras"
export PYTHONHASHSEED=42
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
case "${1:-}" in
  prepare-b) shift; exec .venv/bin/python scripts/prepare_b.py "$@" ;;
  smoke|pilot) project_stage="$1"; shift; exec .venv/bin/python scripts/smoke_pilot.py --stage "$project_stage" "$@" ;;
  *) echo 'Expected prepare-b, smoke, or pilot.' >&2; exit 2 ;;
esac
