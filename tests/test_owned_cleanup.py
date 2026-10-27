from tests.support import test_temp_dir
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dace.benchmarks.cleanup import cleanup_owned_job


class OwnedCleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.job,self.task = self.root/'job',self.root/'task'
        self.trial = self.job/'task__ABc123'
        self.trial.mkdir(parents=True)
        self.task.mkdir()
        self.config = {'trial_name':self.trial.name,'trials_dir':str(self.job),
                       'task':{'path':str(self.task)}}
        self.path = self.trial/'config.json'
        self.write_config()
        self.request = {'task_id':'task','task_path':str(self.task)}
        self.project = self.trial.name.lower()+'__env'

    def write_config(self):
        self.path.write_text(json.dumps(self.config),encoding='utf-8')

    def docker(self,argv,**kwargs):
        args = argv[1:]
        if args[:2]==['ps','-aq']:
            text = 'owned\nforeign\n'
        elif args[:2]==['inspect','owned']:
            text = json.dumps([{'Id':'owned-full','Config':{'Labels':{'com.docker.compose.project':self.project}}}])
        elif args[:2]==['inspect','foreign']:
            text = json.dumps([{'Id':'foreign-full','Config':{'Labels':{'com.docker.compose.project':'unrelated'}}}])
        elif args[:2]==['network','ls']:
            text = 'network-owned\n'
        elif args[:3]==['network','inspect','network-owned']:
            text = json.dumps([{'Id':'network-full','Labels':{'com.docker.compose.project':self.project},'Containers':{}}])
        else:
            text = ''
        return SimpleNamespace(returncode=0,stdout=text)

    def test_only_exact_owned_project_resources_are_removed(self):
        with patch('dace.benchmarks.cleanup.subprocess.run',side_effect=self.docker) as docker:
            result = cleanup_owned_job(self.job,self.request)
        calls = [c.args[0] for c in docker.call_args_list]
        self.assertIn(['docker','rm','--force','owned-full'],calls)
        self.assertIn(['docker','network','rm','network-full'],calls)
        self.assertNotIn(['docker','rm','--force','foreign-full'],calls)
        self.assertEqual(result['removed_containers'],['owned-full'])
        self.assertEqual(result['status'],'CLEANUP_UNVERIFIED')  # A foreign resource was refused.

    def test_wrong_task_identity_prevents_all_docker_mutations(self):
        self.config['task']['path'] = str(self.root/'other-task')
        self.write_config()
        with patch('dace.benchmarks.cleanup.subprocess.run') as docker:
            result = cleanup_owned_job(self.job,self.request)
        docker.assert_not_called()
        self.assertEqual(result['status'],'CLEANUP_UNVERIFIED')

    def test_missing_trial_metadata_is_not_a_cleanup_success(self):
        self.path.unlink()
        with patch('dace.benchmarks.cleanup.subprocess.run') as docker:
            result = cleanup_owned_job(self.job,self.request)
        docker.assert_not_called()
        self.assertEqual(result['status'],'CLEANUP_UNVERIFIED')

    def test_docker_error_is_recorded_without_claiming_resources_removed(self):
        with patch('dace.benchmarks.cleanup.subprocess.run',side_effect=RuntimeError('fixture failure')):
            result = cleanup_owned_job(self.job,self.request)
        self.assertEqual(result['removed_containers'],[])
        self.assertEqual(result['status'],'CLEANUP_UNVERIFIED')
        self.assertEqual(result['errors'][0]['error_type'],'RuntimeError')


if __name__=='__main__':
    unittest.main()
