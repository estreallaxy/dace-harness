"""Transactional task roles and disclosure exclusion across source sessions.

A study fixes the role partition. A lineage is one optimizer's information
history, including all its rounds and descendants. Independent runs may share
the same task partition, but cannot silently relabel exposed tasks as fresh.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .benchmarks.fingerprint import task_fingerprint
from .core import digest


ROLES = frozenset({'DIAG','CONFIRM','SELECT','EVAL'})


class TaskRegistry:
    def __init__(self,path):
        path = Path(path).resolve()
        path.parent.mkdir(parents=True,exist_ok=True)
        self.db = sqlite3.connect(str(path),timeout=10,isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS studies (
                id TEXT PRIMARY KEY, protocol TEXT NOT NULL, hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS lineages (
                study TEXT NOT NULL, id TEXT NOT NULL, parent TEXT,
                PRIMARY KEY(study,id), FOREIGN KEY(study) REFERENCES studies(id),
                FOREIGN KEY(study,parent) REFERENCES lineages(study,id));
            CREATE TABLE IF NOT EXISTS sessions (
                study TEXT NOT NULL, lineage TEXT NOT NULL, owner TEXT NOT NULL,
                role TEXT NOT NULL, output TEXT NOT NULL UNIQUE,
                plan_hash TEXT NOT NULL, tasks TEXT NOT NULL,
                PRIMARY KEY(study,lineage,owner),
                FOREIGN KEY(study,lineage) REFERENCES lineages(study,id));
            CREATE TABLE IF NOT EXISTS claims (
                study TEXT NOT NULL, lineage TEXT NOT NULL, task TEXT NOT NULL,
                owner TEXT NOT NULL, PRIMARY KEY(study,lineage,task,owner),
                FOREIGN KEY(study,lineage,owner) REFERENCES sessions(study,lineage,owner));
            CREATE TABLE IF NOT EXISTS disclosures (
                study TEXT NOT NULL, lineage TEXT NOT NULL, task TEXT NOT NULL,
                PRIMARY KEY(study,lineage,task),
                FOREIGN KEY(study,lineage) REFERENCES lineages(study,id));
            CREATE TABLE IF NOT EXISTS executions (
                study TEXT NOT NULL, lineage TEXT NOT NULL, owner TEXT NOT NULL,
                task TEXT NOT NULL, PRIMARY KEY(study,lineage,owner,task),
                FOREIGN KEY(study,lineage,owner) REFERENCES sessions(study,lineage,owner));
            CREATE TABLE IF NOT EXISTS events (
                idx INTEGER PRIMARY KEY, previous TEXT, kind TEXT NOT NULL,
                payload TEXT NOT NULL, hash TEXT NOT NULL);
        ''')

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.db.close()

    @contextmanager
    def _write(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def _event(self,kind,**payload):
        last = self.db.execute('SELECT idx,hash FROM events ORDER BY idx DESC LIMIT 1').fetchone()
        event = {'index':last['idx']+1 if last else 0,
                 'previous':last['hash'] if last else None,'kind':kind,'payload':payload}
        self.db.execute('INSERT INTO events VALUES (?,?,?,?,?)',
            (event['index'],event['previous'],kind,json.dumps(payload,sort_keys=True),digest(event)))

    def create_study(self,study,dataset,allocation,benchmark='terminalbench2'):
        """Freeze every task in the supplied dataset into exactly one role."""
        root = Path(dataset).resolve()
        if not isinstance(study,str) or not study:
            raise ValueError('Study must have an explicit ID')
        inventory = {p.name for p in root.iterdir() if p.is_dir() and (p/'task.toml').is_file()}
        if not inventory or set(allocation) - ROLES:
            raise ValueError('Expected dataset and DIAG/CONFIRM/SELECT/EVAL roles')
        tasks, seen = {}, set()
        for role,ids in allocation.items():
            if not isinstance(ids,(list,tuple)):
                raise ValueError('Role allocation must list exact task IDs')
            for task in ids:
                if task not in inventory or task in tasks:
                    raise ValueError('Task role overlap or unknown task')
                fingerprint = task_fingerprint(root/task)
                if fingerprint in seen:
                    raise ValueError('Duplicate task content cannot be independent units')
                seen.add(fingerprint)
                tasks[task] = {'role':role,'fingerprint':fingerprint}
        if set(tasks) != inventory:
            raise ValueError('All dataset tasks must have an explicit frozen role')
        protocol = {'schema':'dace-study/v1','benchmark':benchmark,
                    'dataset':str(root),'tasks':tasks}
        identity = digest(protocol)
        with self._write():
            previous = self.db.execute('SELECT hash FROM studies WHERE id=?',(study,)).fetchone()
            if previous:
                if previous['hash'] != identity:
                    raise ValueError('Study role partition and dataset are immutable')
                return identity
            self.db.execute('INSERT INTO studies VALUES (?,?,?)',
                            (study,json.dumps(protocol,sort_keys=True),identity))
            self._event('STUDY_FROZEN',study=study,protocol_hash=identity,protocol=protocol)
        return identity

    def protocol(self,study):
        row = self.db.execute('SELECT protocol,hash FROM studies WHERE id=?',(study,)).fetchone()
        if row is None:
            raise ValueError('Study is not frozen')
        protocol = json.loads(row['protocol'])
        if digest(protocol) != row['hash']:
            raise ValueError('Study protocol integrity failure')
        return protocol

    def create_lineage(self,study,lineage,parent=None):
        self.protocol(study)
        if not isinstance(lineage,str) or not lineage or parent == lineage:
            raise ValueError('Expected a distinct, explicit optimizer lineage ID')
        with self._write():
            old = self.db.execute('SELECT parent FROM lineages WHERE study=? AND id=?',
                                  (study,lineage)).fetchone()
            if old:
                if old['parent'] != parent:
                    raise ValueError('Lineage ancestry is immutable')
                return
            if parent is not None and not self.db.execute(
                    'SELECT 1 FROM lineages WHERE study=? AND id=?',(study,parent)).fetchone():
                raise ValueError('Parent optimizer lineage is unknown')
            self.db.execute('INSERT INTO lineages VALUES (?,?,?)',(study,lineage,parent))
            self._event('LINEAGE_CREATED',study=study,lineage=lineage,parent=parent)

    def _ancestry(self,study,lineage):
        result = []
        while lineage is not None:
            if lineage in result:
                raise ValueError('Lineage ancestry cycle')
            result.append(lineage)
            row = self.db.execute('SELECT parent FROM lineages WHERE study=? AND id=?',
                                  (study,lineage)).fetchone()
            if row is None:
                raise ValueError('Optimizer lineage must be registered before execution')
            lineage = row['parent']
        return result

    def claim(self,study,lineage,owner,role,tasks,fingerprints,output,plan_hash):
        """Reserve the entire session before executing; failures never release units."""
        protocol = self.protocol(study)
        tasks = list(tasks)
        output = os.path.normcase(str(Path(output).resolve()))
        if (role not in ROLES or not tasks or len(set(tasks)) != len(tasks)
                or set(fingerprints) != set(tasks) or not owner or not plan_hash):
            raise ValueError('Invalid frozen session task plan')
        for task in tasks:
            expected = protocol['tasks'].get(task)
            if (expected is None or expected['role'] != role
                    or expected['fingerprint'] != fingerprints[task]):
                raise ValueError('Task role or execution fingerprint violates frozen study')
        with self._write():
            ancestry = self._ancestry(study,lineage)
            old = self.db.execute('SELECT * FROM sessions WHERE study=? AND lineage=? AND owner=?',
                                  (study,lineage,owner)).fetchone()
            if old:
                if (old['role'] != role or old['output'] != output
                        or old['plan_hash'] != plan_hash or json.loads(old['tasks']) != tasks):
                    raise ValueError('Session cannot be cloned or changed on resume')
                return
            for task in tasks:
                for ancestor in ancestry:
                    if self.db.execute('SELECT 1 FROM disclosures WHERE study=? AND lineage=? AND task=?',
                                       (study,ancestor,task)).fetchone():
                        raise ValueError('Task or derived evidence already disclosed to this lineage')
                    used = self.db.execute('SELECT 1 FROM claims WHERE study=? AND lineage=? AND task=?',
                                          (study,ancestor,task)).fetchone()
                    if used and role != 'EVAL':
                        raise ValueError('Task already reserved or used in this optimizer lineage')
            self.db.execute('INSERT INTO sessions VALUES (?,?,?,?,?,?,?)',
                (study,lineage,owner,role,output,plan_hash,json.dumps(tasks)))
            self.db.executemany('INSERT INTO claims VALUES (?,?,?,?)',
                               [(study,lineage,task,owner) for task in tasks])
            self._event('TASKS_RESERVED',study=study,lineage=lineage,owner=owner,role=role,
                        tasks=tasks,output=output,plan_hash=plan_hash)

    def mark_started(self,study,lineage,owner,task):
        """Persist before the first model call; a missing local ledger is no retry license."""
        with self._write():
            if not self.db.execute('SELECT 1 FROM claims WHERE study=? AND lineage=? AND owner=? AND task=?',
                                   (study,lineage,owner,task)).fetchone():
                raise ValueError('Execution lacks an exact session reservation')
            if self.db.execute('SELECT 1 FROM executions WHERE study=? AND lineage=? AND owner=? AND task=?',
                               (study,lineage,owner,task)).fetchone():
                raise ValueError('Task already started; local evidence cannot be discarded to retry')
            role = self.db.execute('SELECT role FROM sessions WHERE study=? AND lineage=? AND owner=?',
                                   (study,lineage,owner)).fetchone()['role']
            # A parent's history may grow after this child's plan was reserved.
            # Recheck immediately before execution rather than trust a stale lease.
            for ancestor in self._ancestry(study,lineage):
                if self.db.execute('SELECT 1 FROM disclosures WHERE study=? AND lineage=? AND task=?',
                                   (study,ancestor,task)).fetchone():
                    raise ValueError('Task evidence was disclosed after session reservation')
                if ancestor != lineage and role != 'EVAL' and self.db.execute(
                        'SELECT 1 FROM claims WHERE study=? AND lineage=? AND task=?',
                        (study,ancestor,task)).fetchone():
                    raise ValueError('Ancestor reserved this unit after session reservation')
            self.db.execute('INSERT INTO executions VALUES (?,?,?,?)',(study,lineage,owner,task))
            self._event('EXECUTION_STARTED',study=study,lineage=lineage,owner=owner,task=task)

    def disclose(self,study,lineage,tasks,fields):
        """Register actual source units of optimizer feedback, including aggregates."""
        tasks = list(tasks)
        fields = list(fields)
        protocol = self.protocol(study)
        if not tasks or not fields or any(not isinstance(x,str) or not x for x in fields):
            raise ValueError('Disclosure requires source tasks and field names')
        with self._write():
            ancestry = self._ancestry(study,lineage)
            for task in tasks:
                expected = protocol['tasks'].get(task)
                if expected is None or expected['role']=='EVAL':
                    raise ValueError('Held-out evaluation evidence cannot reach the optimizer')
                if not any(self.db.execute(
                        'SELECT 1 FROM executions WHERE study=? AND lineage=? AND task=?',
                        (study,a,task)).fetchone() for a in ancestry):
                    raise ValueError('Disclosure source task has no registered execution')
            self.db.executemany('INSERT OR IGNORE INTO disclosures VALUES (?,?,?)',
                               [(study,lineage,task) for task in tasks])
            self._event('AGENT_DISCLOSED',study=study,lineage=lineage,tasks=tasks,fields=fields)

    def audit_events(self):
        events = []
        for row in self.db.execute('SELECT * FROM events ORDER BY idx'):
            event = {'index':row['idx'],'previous':row['previous'],'kind':row['kind'],
                     'payload':json.loads(row['payload'])}
            if (event['index'] != len(events) or event['previous'] != (events[-1]['hash'] if events else None)
                    or digest(event) != row['hash']):
                raise ValueError('Registry audit chain integrity failure')
            events.append(dict(event,hash=row['hash']))
        return events
