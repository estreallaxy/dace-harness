from tests.support import test_temp_dir
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from dace.benchmarks.process import run_owned_process


class OwnedProcessTests(unittest.TestCase):
    def test_short_lived_runner_keeps_its_output(self):
        result = run_owned_process([sys.executable,'-c','print("complete")'],
                                   capture_output=True,text=True,timeout=10)
        self.assertEqual((result.returncode,result.stdout.strip()),(0,'complete'))

    def test_timeout_stops_descendants_and_preserves_partial_output(self):
        with tempfile.TemporaryDirectory(dir=test_temp_dir()) as tmp:
            root = Path(tmp)
            heartbeat = root/'heartbeat.txt'
            child = root/'child.py'
            child.write_text('import sys,time\nfrom pathlib import Path\n'
                'p=Path(sys.argv[1])\ni=0\nwhile True:\n'
                '    p.write_text(str(i))\n    i+=1\n    time.sleep(.05)\n',encoding='utf-8')
            parent = root/'parent.py'
            parent.write_text('import subprocess,sys,time\n'
                'subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2]])\n'
                'print("started",flush=True)\ntime.sleep(60)\n',encoding='utf-8')
            with self.assertRaises(subprocess.TimeoutExpired) as caught:
                run_owned_process([sys.executable,str(parent),str(child),str(heartbeat)],
                                  capture_output=True,text=True,timeout=1.5)
            self.assertIn('started',caught.exception.stdout)
            self.assertTrue(heartbeat.is_file())
            before = heartbeat.read_text()
            time.sleep(.3)
            self.assertEqual(heartbeat.read_text(),before)

    def test_successful_parent_does_not_leave_background_child_running(self):
        with tempfile.TemporaryDirectory(dir=test_temp_dir()) as tmp:
            marker = Path(tmp)/'late.txt'
            child = 'import time;from pathlib import Path;time.sleep(1);Path('+repr(str(marker))+').write_text("late")'
            parent = 'import subprocess,sys;subprocess.Popen([sys.executable,"-c",sys.argv[1]],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)'
            result = run_owned_process([sys.executable,'-c',parent,child],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0)
            time.sleep(1.2)
            self.assertFalse(marker.exists())

    def test_shell_execution_is_never_enabled(self):
        with self.assertRaises(ValueError):
            run_owned_process(['echo','hello'],shell=True,timeout=1)


if __name__=='__main__':
    unittest.main()
