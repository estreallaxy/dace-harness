from tests.support import test_temp_dir
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dace.runtime_lock import capture_runtime_lock, verify_runtime_lock


class RuntimeLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = self.root/'project'
        (self.project/'dace').mkdir(parents=True)
        (self.project/'tools').mkdir()
        self.source = self.project/'dace/implementation.py'
        self.source.write_text('original = True',encoding='utf-8')
        script = self.project/'tools/runner.py'
        script.write_text('# runner',encoding='utf-8')
        runtime = self.root/'runtime'
        runtime.mkdir()
        self.harbor = runtime/('harbor.exe' if os.name=='nt' else 'harbor')
        self.harbor.write_bytes(b'launcher')
        self.harbor.chmod(0o700)
        (runtime/('python.exe' if os.name=='nt' else 'python')).write_bytes(b'python-runtime')
        self.command = [sys.executable,str(script),'--harbor',str(self.harbor),'--max-turns','8']
        self.dataset = self.root/'dataset'
        task = self.dataset/'task'
        task.mkdir(parents=True)
        self.manifest = task/'task.toml'
        self.manifest.write_text('[environment]\ndocker_image="test:tag"',encoding='utf-8')
        self.probe = {'harbor_version':'0.25.0','packages':[['harbor','0.25.0']],
                      'harbor_sources':{'terminus.py':'hash'}}
        self.image = {'id':'sha256:original','repo_digests':[], 'os':'linux','architecture':'amd64'}
        for name,value in [('PROJECT_ROOT',self.project),('_runtime_probe',None),('_image_identity',None)]:
            p = patch('dace.runtime_lock.'+name,value) if value is not None else patch('dace.runtime_lock.'+name)
            mocked = p.start()
            self.addCleanup(p.stop)
            if name=='_runtime_probe':
                mocked.side_effect = lambda python: dict(self.probe)
            elif name=='_image_identity':
                mocked.side_effect = lambda image: dict(self.image)

    def capture(self):
        return capture_runtime_lock(self.command,self.dataset,['task'])

    def verify(self,lock):
        return verify_runtime_lock(lock,self.command,self.dataset,['task'])

    def test_unchanged_runtime_has_stable_identity(self):
        lock = self.capture()
        self.assertEqual(len(self.verify(lock)),64)
        self.assertEqual(lock['images']['test:tag']['id'],'sha256:original')

    def test_source_edit_in_place_is_detected_even_if_argv_did_not_change(self):
        lock = self.capture()
        self.source.write_text('original = False',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'FROZEN_RUNTIME_CHANGED'):
            self.verify(lock)

    def test_new_implementation_module_changes_bundle_identity(self):
        lock = self.capture()
        (self.project/'dace/new_module.py').write_text('extra = True',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'FROZEN_RUNTIME_CHANGED'):
            self.verify(lock)

    def test_removed_implementation_module_changes_bundle_identity(self):
        lock = self.capture()
        self.source.unlink()
        with self.assertRaisesRegex(ValueError,'FROZEN_RUNTIME_CHANGED'):
            self.verify(lock)

    def test_mutable_docker_tag_cannot_silently_point_at_new_image(self):
        lock = self.capture()
        self.image['id'] = 'sha256:changed'
        with self.assertRaisesRegex(ValueError,'FROZEN_RUNTIME_CHANGED'):
            self.verify(lock)

    def test_dependency_version_drift_is_detected(self):
        lock = self.capture()
        self.probe['packages'] = [['harbor','0.25.0'],['new-dependency','1.0']]
        with self.assertRaisesRegex(ValueError,'FROZEN_RUNTIME_CHANGED'):
            self.verify(lock)

    def test_harbor_source_edit_is_detected(self):
        lock = self.capture()
        self.probe['harbor_sources'] = {'terminus.py':'changed'}
        with self.assertRaisesRegex(ValueError,'FROZEN_RUNTIME_CHANGED'):
            self.verify(lock)

    def test_launcher_replacement_is_detected(self):
        lock = self.capture()
        self.harbor.write_bytes(b'replaced')
        with self.assertRaisesRegex(ValueError,'FROZEN_RUNTIME_CHANGED'):
            self.verify(lock)

    def test_execution_limit_change_is_not_the_same_protocol(self):
        lock = self.capture()
        self.command[-1] = '16'
        with self.assertRaisesRegex(ValueError,'FROZEN_RUNTIME_CHANGED'):
            self.verify(lock)

    def test_other_directory_cannot_stand_in_for_executing_source(self):
        with self.assertRaisesRegex(ValueError,'executing DACE'):
            capture_runtime_lock(self.command,self.dataset,['task'],self.root/'elsewhere')

    def test_build_only_task_requires_separate_image_preparation(self):
        self.manifest.write_text('[environment]\ncpus=1',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'BUILD_ONLY_TASK_UNPINNED'):
            self.capture()

    def test_shared_runtime_scope_can_cover_confirm_and_selection_task_subsets(self):
        other = self.dataset/'selection'
        other.mkdir()
        (other/'task.toml').write_bytes(self.manifest.read_bytes())
        lock = capture_runtime_lock(self.command,self.dataset,['task','selection'])
        verify_runtime_lock(lock,self.command,self.dataset,['task'])
        verify_runtime_lock(lock,self.command,self.dataset,['selection'])
        with self.assertRaisesRegex(ValueError,'outside the frozen runtime scope'):
            verify_runtime_lock(lock,self.command,self.dataset,['unfrozen-task'])


if __name__=='__main__':
    unittest.main()
