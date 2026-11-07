import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_assets import verify


class AssetTests(unittest.TestCase):
    def test_missing_pointer_corruption_and_correct_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'asset'
            expected=dict(size=3,sha256=hashlib.sha256(b'abc').hexdigest())
            self.assertEqual(verify(p,expected)['status'],'missing')
            p.write_bytes(b'version https://git-lfs.github.com/spec/v1\n')
            self.assertEqual(verify(p,expected)['status'],'lfs_pointer_only')
            p.write_bytes(b'abd')
            self.assertEqual(verify(p,expected)['status'],'mismatch')
            p.write_bytes(b'abc')
            self.assertEqual(verify(p,expected)['status'],'verified')
