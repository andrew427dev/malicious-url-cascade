"""Optional integration check against the exact input features of the source model."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from urlcascade.adapters import preprocess_link
from urlcascade.features import extract_features

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'upstream/Utils/FeatureCreation.py'
NAMES = (
    'having_ip_address', 'abnormal_url', 'count_dot', 'count_www', 'count_atrate',
    'no_of_dir', 'no_of_embed', 'count_https', 'count_http', 'count_per',
    'count_ques', 'count_hyphen', 'count_equal', 'url_length', 'hostname_length',
    'digit_count', 'letter_count', 'fd_length',
)


@unittest.skipUnless(SOURCE.exists(), 'Run scripts/fetch_reference.py for upstream parity')
class ReferenceTests(unittest.TestCase):
    def test_features_and_preprocessing_match_reference(self):
        hashes = json.loads((ROOT / 'configs/upstream.json').read_text())['source_sha256']
        # Only scalar string functions are called. Stub ML imports for this offline check.
        modules = {'pandas': types.ModuleType('pandas'),
                   'tensorflow': types.SimpleNamespace(data=types.SimpleNamespace(AUTOTUNE=0)),
                   'Utils': types.ModuleType('Utils')}
        with patch.dict(sys.modules, modules):
            for name in ('FeatureCreation', 'DataProcessing'):
                path = SOURCE.with_name(name + '.py')
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), hashes['Utils/' + path.name])
                spec = importlib.util.spec_from_file_location('Utils.' + name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
            feature_module = sys.modules['Utils.FeatureCreation']
            preprocess = sys.modules['Utils.DataProcessing'].preprocess_link

        schemes = ('https://', 'HTTP://', '', '//')
        hosts = ('example.test', 'WWW.Example.test:80', 'user@host.test', '192.0.2.1',
                 '[2001:db8::1]', '예제.test')
        paths = ('', '/', '/a//b', '/0x7f.0x0.0x0.0x1/1:2:3:4:5:6:7:8',
                 '/192.0.2.1/', '/256.2.3.4/', '/١²四?x=%20-2=', '/WWW.www?q=www')
        for scheme, host, path in itertools.product(schemes, hosts, paths):
            raw = scheme + host + path
            with self.subTest(raw=raw):
                self.assertEqual(extract_features(raw), [getattr(feature_module, n)(raw) for n in NAMES])
                self.assertEqual(preprocess_link(raw), preprocess(raw))
