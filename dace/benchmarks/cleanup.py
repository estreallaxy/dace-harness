"""Remove only Docker resources tied to this exact Harbor job and task."""
import json
import subprocess
from pathlib import Path


def cleanup_owned_job(job,request):
    job = Path(job).resolve()
    task = Path(request['task_path']).resolve()
    report = {'schema':'dace-owned-cleanup/v1','checked_projects':[],
              'removed_containers':[],'removed_networks':[],'refused':[],'errors':[]}
    def docker(*args):
        result = subprocess.run(['docker',*args],capture_output=True,text=True,timeout=30,shell=False)
        if result.returncode:
            raise RuntimeError('DOCKER_CLEANUP_COMMAND_FAILED')
        return result.stdout
    for path in job.glob('*/config.json'):
        try:
            config = json.loads(path.read_text(encoding='utf-8'))
            name = config.get('trial_name','')
            if (name!=path.parent.name or not name.startswith(request['task_id']+'__')
                    or Path(config.get('trials_dir','')).resolve()!=job
                    or Path((config.get('task') or {}).get('path','')).resolve()!=task):
                report['refused'].append({'config':str(path),'reason':'FOREIGN_TRIAL_IDENTITY'})
                continue
            project = name.lower()+'__env'  # Pinned Harbor 0.25.0 Docker backend.
            report['checked_projects'].append(project)
            for container in docker('ps','-aq','--filter',f'label=com.docker.compose.project={project}').split():
                data = json.loads(docker('inspect',container))[0]
                if (data.get('Config',{}).get('Labels') or {}).get('com.docker.compose.project')!=project:
                    report['refused'].append({'container':container,'reason':'FOREIGN_PROJECT_LABEL'})
                    continue
                identity = data['Id']
                docker('rm','--force',identity)
                report['removed_containers'].append(identity)
            for network in docker('network','ls','-q','--filter',
                                  f'label=com.docker.compose.project={project}').split():
                data = json.loads(docker('network','inspect',network))[0]
                if ((data.get('Labels') or {}).get('com.docker.compose.project')!=project
                        or data.get('Containers')):
                    report['refused'].append({'network':network,'reason':'FOREIGN_OR_IN_USE_NETWORK'})
                    continue
                docker('network','rm',data['Id'])
                report['removed_networks'].append(data['Id'])
        except Exception as exc:
            report['errors'].append({'config':str(path),'error_type':type(exc).__name__})
    report['status'] = ('CLEANUP_COMPLETED' if report['checked_projects']
        and not report['errors'] and not report['refused'] else 'CLEANUP_UNVERIFIED')
    return report
