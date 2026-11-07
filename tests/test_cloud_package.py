import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_cloud_notebook import make_bundle,payload_files
from cloud_run import child_environment


class CloudPackageTests(unittest.TestCase):
    def test_embedded_payload_matches_current_code_and_safe_members(self):
        nb = json.loads((ROOT/'notebooks/HUFS_CompSec_Cloud.ipynb').read_text())
        cell = ''.join(nb['cells'][3]['source'])
        # Only constant assignments are inspected; cloud cells are not executed locally.
        import ast
        constants = {}
        for node in ast.parse(cell).body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
                if node.targets[0].id in ('PAYLOAD_SHA256','PAYLOAD','EXPECTED_FILES'):
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
        data = base64.b64decode(constants['PAYLOAD'],validate=True)
        self.assertEqual(hashlib.sha256(data).hexdigest(),constants['PAYLOAD_SHA256'])
        current,hashes = make_bundle()
        self.assertEqual(data,current,'Rebuild notebook after changing bundled code')
        self.assertEqual(hashes,constants['EXPECTED_FILES'])
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                self.assertFalse(PurePosixPath(name).is_absolute())
                self.assertNotIn('..',PurePosixPath(name).parts)
                self.assertNotIn(name.split('/')[0],('data','artifacts','evidence'))
                self.assertEqual(z.read(name),(ROOT/name).read_bytes())

    def test_notebook_is_unexecuted_diagnostic_by_default_and_compiles(self):
        nb = json.loads((ROOT/'notebooks/HUFS_CompSec_Cloud.ipynb').read_text())
        self.assertIn('STAGE = "diagnose"',''.join(nb['cells'][1]['source']))
        self.assertFalse(nb['metadata']['hufs']['executed'])
        for i,cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'],[])
                self.assertIsNone(cell['execution_count'])
                compile(''.join(cell['source']),f'cell-{i}','exec')

    def test_child_environment_isolated_without_changing_parent(self):
        with patch.dict(os.environ,{'PYTHONPATH':'/kernel-packages','PIP_TARGET':'/global',
                                   'UV_PYTHON_INSTALL_DIR':'/global-python'},clear=False):
            env = child_environment(Path('/kaggle/working/hufs'))
            self.assertNotIn('PYTHONPATH',env)
            self.assertNotIn('PIP_TARGET',env)
            self.assertEqual(env['UV_PYTHON_INSTALL_DIR'],'/kaggle/working/hufs/python')
            self.assertEqual(env['PYTHONHASHSEED'],'42')
            self.assertEqual(os.environ['PIP_TARGET'],'/global')
