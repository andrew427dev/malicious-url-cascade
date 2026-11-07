import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cloud_run
import fetch_assets
import verify_assets


class CloudRecoveryTests(unittest.TestCase):
    def manifest(self):
        payload = b'url,type\nexample.com,1\n'
        names = ['dataset/df_final.csv','dataset/all_top_domains_merged.csv',
                 'dataset/df_another_validation.csv','Trained_Model/Model/Model.keras']
        expected = dict(size=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        return dict(commit='fixed-test-commit',files={name:expected for name in names}),payload

    def test_optional_model_network_failure_preserves_required_success(self):
        manifest,payload = self.manifest()
        urls = []
        def response(url,timeout):
            urls.append(url)
            if url.endswith('.keras'):
                raise OSError('simulated reference-model outage')
            return io.BytesIO(payload)
        with tempfile.TemporaryDirectory() as td, patch.object(fetch_assets,'urlopen',side_effect=response):
            report = fetch_assets.fetch_manifest(manifest,Path(td),include_reference_model=True)
            self.assertTrue(report['required_ok'])
            self.assertEqual(len(report['results']),3)
            self.assertTrue(all(r['status']=='verified' for r in report['results'].values()))
            optional = report['optional_results']['Trained_Model/Model/Model.keras']
            self.assertEqual(optional['status'],'failed')
            self.assertFalse(optional['required'])
            self.assertIn('outage',optional['error'])
            self.assertEqual(len(urls),4)

    def test_default_fetch_skips_reference_model_and_required_corruption_fails(self):
        manifest,payload = self.manifest()
        with tempfile.TemporaryDirectory() as td, patch.object(fetch_assets,'urlopen',side_effect=lambda *a,**k:io.BytesIO(payload)) as request:
            report = fetch_assets.fetch_manifest(manifest,Path(td))
            self.assertTrue(report['required_ok'])
            self.assertEqual(request.call_count,3)
            self.assertEqual(report['optional_results']['Trained_Model/Model/Model.keras']['status'],'not_requested')
            broken = Path(td)/'dataset/df_final.csv'
            broken.write_bytes(b'corrupt')
            report = fetch_assets.fetch_manifest(manifest,Path(td))
            self.assertFalse(report['required_ok'])
            self.assertEqual(broken.read_bytes(),b'corrupt')

    def test_required_data_verification_ignores_missing_model_but_requires_external_csv(self):
        manifest,payload = self.manifest()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'configs').mkdir()
            (root/'configs/upstream.json').write_text(json.dumps(manifest))
            for name in manifest['files']:
                if name.endswith('.csv'):
                    p=root/'assets'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
            argv=['verify','--root',str(root/'assets'),'--report',str(root/'report.json'),'--required-data-only']
            with patch.object(verify_assets,'ROOT',root), patch.object(sys,'argv',argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(verify_assets.main(),0)
                (root/'assets/dataset/df_another_validation.csv').unlink()
                self.assertEqual(verify_assets.main(),1)
            report=json.loads((root/'report.json').read_text())
            self.assertEqual(report['optional_results']['Trained_Model/Model/Model.keras']['status'],'missing')
            self.assertFalse(report['required_ok'])

    def test_unreadable_optional_model_is_a_recorded_failure(self):
        with patch.object(verify_assets,'verify',side_effect=PermissionError('not readable')):
            report=verify_assets.verify_optional(Path('/reference.keras'),{})
        self.assertEqual(report['status'],'failed')
        self.assertFalse(report['required'])

    def test_existing_python_without_ensurepip_uses_managed_fallback(self):
        calls=[]
        def run(argv,check=True):
            calls.append((argv,check))
            return 1 if not check and 'ensurepip' in ' '.join(argv) else 0
        with tempfile.TemporaryDirectory() as td, patch.object(cloud_run.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'uv '+cloud_run.UV_VERSION+'\n','')):
            root=Path(td)
            cloud_run.setup(run,root,root/'env/bin/python',{})
        self.assertTrue(any(not check and 'ensurepip' in ' '.join(argv) for argv,check in calls))
        self.assertTrue(any('--python' in argv and cloud_run.MANAGED_PYTHON in argv for argv,_ in calls))
        self.assertFalse(any('-m' in argv and 'venv' in argv for argv,_ in calls))

    def test_incomplete_environment_is_preserved_without_installing(self):
        for has_python in (False,True):
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);env=root/'env';env.mkdir()
                (env/'incomplete.txt').write_text('preserve this evidence')
                if has_python:
                    (env/'bin').mkdir();(env/'bin/python').write_text('incomplete executable')
                calls=[]
                with self.assertRaisesRegex(RuntimeError,'existing_environment_incomplete'):
                    cloud_run.setup(lambda *a,**k:calls.append(a),root,env/'bin/python',{})
                self.assertEqual(calls,[])
                self.assertEqual((env/'incomplete.txt').read_text(),'preserve this evidence')

    def test_failed_existing_env_preflight_does_not_switch_or_reinstall(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);env=root/'env';(env/'bin').mkdir(parents=True)
            (env/'pyvenv.cfg').write_text('test');(env/'bin/python').write_text('test')
            calls=[]
            def run(argv,check=True):
                calls.append(argv);return 1
            with self.assertRaisesRegex(RuntimeError,'existing_environment_failed_preflight'):
                cloud_run.setup(run,root,env/'bin/python',{})
            self.assertEqual(len(calls),1)
            self.assertFalse(any('install' in argv for argv in calls))
