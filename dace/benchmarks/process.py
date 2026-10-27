"""Bound an owned runner process and its descendants, without a shell."""
import os
import signal
import subprocess


class _WindowsJob:
    def __init__(self):
        import ctypes
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),
                ('flags',w.DWORD),('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),
                ('active',w.DWORD),('affinity',ctypes.c_size_t),('priority',w.DWORD),('schedule',w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name,ctypes.c_ulonglong) for name in
                        ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
        class Limits(ctypes.Structure):
            _fields_ = [('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),
                ('job_memory',ctypes.c_size_t),('peak_process_memory',ctypes.c_size_t),
                ('peak_job_memory',ctypes.c_size_t)]
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p,w.LPCWSTR]
        kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.SetInformationJobObject.argtypes = [w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD]
        kernel.SetInformationJobObject.restype = w.BOOL
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE,w.HANDLE]
        kernel.AssignProcessToJobObject.restype = w.BOOL
        kernel.CloseHandle.argtypes = [w.HANDLE]
        kernel.CloseHandle.restype = w.BOOL
        self.kernel,self.handle = kernel,kernel.CreateJobObjectW(None,None)
        if not self.handle:
            raise RuntimeError('OWNED_PROCESS_JOB_UNAVAILABLE')
        limits = Limits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            self.close()
            raise RuntimeError('OWNED_PROCESS_JOB_UNAVAILABLE')

    def assign_resume(self,process):
        import ctypes
        from ctypes import wintypes as w
        if not self.kernel.AssignProcessToJobObject(self.handle,w.HANDLE(int(process._handle))):
            raise RuntimeError('OWNED_PROCESS_JOB_ASSIGNMENT_FAILED')
        # Spawn suspended so even a short-lived launcher cannot create a child
        # or exit before its job ownership is established.
        native = ctypes.WinDLL('ntdll')
        native.NtResumeProcess.argtypes = [w.HANDLE]
        native.NtResumeProcess.restype = ctypes.c_long
        if native.NtResumeProcess(w.HANDLE(int(process._handle))) < 0:
            raise RuntimeError('OWNED_PROCESS_RESUME_FAILED')

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _kill_tree(process,job):
    if os.name=='nt':
        # Terminate the live root tree, then close its independently owned job.
        if process.poll() is None:
            subprocess.run([os.path.join(os.environ.get('SystemRoot','C:/Windows'),
                'System32','taskkill.exe'),'/PID',str(process.pid),'/T','/F'],
                capture_output=True,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW)
        if job:
            job.close()
    else:
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)


def run_owned_process(args,*,timeout,capture_output=False,**kwargs):
    """A run()-compatible result, with owned-tree cleanup on timeout/interruption.

    Windows uses an OS job plus tree termination; POSIX uses a new process
    group. This owns helper descendants, not arbitrary Docker resources.
    """
    if not isinstance(args,list) or not args or kwargs.get('shell',False):
        raise ValueError('Owned runners require an argv list and shell=False')
    if capture_output:
        if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
            raise ValueError('capture_output conflicts with output streams')
        kwargs.update(stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    job = None
    if os.name=='nt':
        job = _WindowsJob()
        kwargs['creationflags'] = (kwargs.get('creationflags',0)
            | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW | 0x4)  # CREATE_SUSPENDED
    else:
        kwargs['start_new_session'] = True
    try:
        process = subprocess.Popen(args,**kwargs)
    except BaseException:
        if job:
            job.close()
        raise
    try:
        if os.name=='nt':
            job.assign_resume(process)
        try:
            stdout,stderr = process.communicate(timeout=timeout)
        except BaseException as exc:
            _kill_tree(process,job)
            stdout,stderr = process.communicate(timeout=5)
            if isinstance(exc,subprocess.TimeoutExpired):
                exc.output,exc.stderr = stdout,stderr
            raise
        return subprocess.CompletedProcess(args,process.returncode,stdout,stderr)
    except BaseException:
        if process.poll() is None:
            _kill_tree(process,job)
        raise
    finally:
        if job:
            job.close()
        elif os.name!='nt':
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
        for stream in (process.stdin,process.stdout,process.stderr):
            if stream is not None:
                stream.close()
