"""Capture reproducible versions and non-sensitive host resource facts."""

import ctypes
import importlib.metadata
import platform
import shutil
import sys
import subprocess
import winreg
from pathlib import Path
from common import ROOT, sha, write_json
import tldextract


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong)
        for name in (
            "total_physical",
            "available_physical",
            "total_page",
            "available_page",
            "total_virtual",
            "available_virtual",
            "available_extended",
        )
    ]


status = MemoryStatus()
status.length = ctypes.sizeof(status)
if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
    raise OSError("memory_query_failed")
key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
psl = Path(tldextract.__file__).parent / ".tld_set_snapshot"
pip_check = subprocess.run(
    [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=True
)
record = {
    "python": platform.python_version(),
    "executable": sys.executable,
    "cpu": cpu,
    "platform": platform.platform(),
    "packages": {
        name: importlib.metadata.version(name)
        for name in (
            "tensorflow",
            "keras",
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "joblib",
            "tldextract",
            "h5py",
        )
    },
    "psl_snapshot_sha256": sha(psl),
    "psl_private_suffixes": True,
    "psl_network_refresh": False,
    "physical_memory_gib": status.total_physical / 2**30,
    "available_memory_gib_at_capture": status.available_physical / 2**30,
    "disk_free_gib_at_capture": shutil.disk_usage(ROOT).free / 2**30,
    "pip_check": "PASS; " + pip_check.stdout.strip(),
}
write_json(ROOT / "environment.json", record)
print(record)
