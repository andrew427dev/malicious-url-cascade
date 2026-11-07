import unittest
from pathlib import Path
from urlcascade.runtime import select_runtime,writable_path


class RuntimeTests(unittest.TestCase):
    def facts(self,**kw):
        f=dict(system='Linux',host='worker',kaggle=False,colab=False,slurm_job=False)
        return dict(f,**kw)

    def test_kaggle_and_colab_do_not_need_fabricated_slurm_id(self):
        self.assertEqual(select_runtime('kaggle',self.facts(kaggle=True)),'kaggle')
        self.assertEqual(select_runtime('colab',self.facts(colab=True)),'colab')

    def test_school_login_is_always_rejected(self):
        for host in ['cscloud-master','cscloud-master.example.edu']:
            with self.assertRaisesRegex(RuntimeError,'login_node'):
                select_runtime('kaggle',self.facts(host=host,kaggle=True,slurm_job=True))

    def test_unknown_runtime_and_mac_compute_are_rejected(self):
        for f in [self.facts(),self.facts(system='Darwin',kaggle=True)]:
            with self.assertRaises(RuntimeError):
                select_runtime('auto',f)
        with self.assertRaises(RuntimeError):
            select_runtime('colab',self.facts(kaggle=True))

    def test_cloud_writes_cannot_escape_workspace(self):
        self.assertEqual(writable_path('/kaggle/working/project/run','kaggle'),Path('/kaggle/working/project/run'))
        for path in ['/kaggle/input/file','/kaggle/working/../input/file','/kaggle/working','/tmp/run']:
            with self.assertRaises(ValueError):
                writable_path(path,'kaggle')

    def test_legacy_compute_job_remains_supported(self):
        self.assertEqual(select_runtime('school',self.facts(slurm_job=True)),'school')
        with self.assertRaises(RuntimeError):
            select_runtime('kaggle',self.facts(slurm_job=True))
