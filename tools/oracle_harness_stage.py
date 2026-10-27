"""CONTROLLED ORACLE ONLY: executes official solutions with injected faults.

Runs INSIDE the task's docker image (invoked by tools/tb2_live_runner.py).
It implements the same four-stage planner -> adapter -> tool -> verifier
pipeline and the same fault semantics as the simulated environment
(dace/harness.py), but every stage does REAL work:

  planner   emits the solution artifact (the task's own solution/solve.sh);
            under PLANNER_TYPE_ERROR it emits a type-invalid script (an
            immediate abort) unless planner_cast repairs the emission;
            under STALE_PLANNER_CACHE it emits a stale partial copy unless
            planner_refresh recomputes it
  adapter   forwards the planned artifact; with adapter_cast it re-derives
            the artifact from the RAW task input instead (the compensation
            channel: the downstream outcome recovers while the planner's
            own contract stays violated)
  tool      really executes the artifact with bash in the task workdir
            (under TOOL_OFF_BY_ONE it runs in the wrong workdir unless
            tool_strict verifies the path)
  verifier  really runs the task's own pytest suite and writes the binary
            reward (under VERIFIER_DRIFT the verifier disagrees unless
            verifier_recheck)

Boundary contracts and activation events are execution facts reported in
the trace, never inferred from the configuration diff. Usage:

    python3 /harness/stage.py <config_json> <faults_json> <payload_json> \
                               [<intervention_json>]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TASK = Path('/task')
OUT = Path('/out')
WORK = Path('/dace_work')


def applies(fault, payload) -> bool:
    """Mirror of harness.Fault.applies (same task-payload semantics)."""
    magnitude = payload['magnitude']
    if fault['kind'] == 'STALE_PLANNER_CACHE' and not payload['cacheable']:
        return False
    return (magnitude % 1000) / 1000.0 < fault['scope']


def corrupt(raw: bytes) -> bytes:
    """PLANNER_TYPE_ERROR: the planner emits a type-invalid script -- the
    emitted artifact aborts immediately (an 'int as string' analog: the
    output violates the downstream interface contract)."""
    text = raw.decode('utf-8', errors='replace')
    lines = text.splitlines(keepends=True)
    if lines and lines[0].startswith('#!'):
        lines.insert(1, 'exit 1\n')
    else:
        lines.insert(0, 'exit 1\n')
    return ''.join(lines).encode('utf-8')


def stale_copy(raw: bytes) -> bytes:
    """STALE_PLANNER_CACHE: the planner serves a stale partial copy of the
    solution (an earlier, incomplete cached entry)."""
    lines = raw.decode('utf-8', errors='replace').splitlines(keepends=True)
    keep = max(1, int(len(lines) * 0.4))
    return ''.join(lines[:keep]).encode('utf-8')


def main() -> int:
    if os.environ.get('DACE_CONTROLLED_ORACLE') != '1':
        raise RuntimeError('CONTROLLED_ORACLE_ONLY: explicit opt-in required')
    config = json.loads(sys.argv[1])
    faults = json.loads(sys.argv[2])
    payload = json.loads(sys.argv[3])
    intervention = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None

    WORK.mkdir(parents=True, exist_ok=True)
    (OUT / 'verifier').mkdir(parents=True, exist_ok=True)

    raw = (TASK / 'solution' / 'solve.sh').read_bytes()

    exercised = set()
    planner_valid, cache_fresh = True, True
    planned = raw
    for f in faults:
        if not applies(f, payload):
            continue
        if f['kind'] == 'PLANNER_TYPE_ERROR':
            planner_valid = bool(config.get('planner_cast'))
            exercised.add('planner_cast')
            if not planner_valid:
                planned = corrupt(raw)
        elif f['kind'] == 'STALE_PLANNER_CACHE':
            cache_fresh = bool(config.get('planner_refresh'))
            exercised.add('planner_refresh')
            if not cache_fresh:
                planned = stale_copy(raw)
    upstream_ok = planner_valid and cache_fresh

    own_garble = any(f['kind'] == 'ADAPTER_GARBLING' and applies(f, payload)
                     for f in faults)
    if own_garble or not upstream_ok:
        adapter_cast = bool(config.get('adapter_cast'))
        adapter_valid = adapter_cast
        exercised.add('adapter_cast')
        # the compensation channel: re-derive from the RAW task input,
        # ignoring the (broken) upstream artifact
        artifact = raw if adapter_cast else planned
    else:
        adapter_valid = True
        artifact = planned

    tool_fault = any(f['kind'] == 'TOOL_OFF_BY_ONE' and applies(f, payload)
                     for f in faults)
    if tool_fault:
        exercised.add('tool_strict')

    # ---- tool stage: REAL execution of the artifact ----
    exec_path = WORK / 'exec.sh'
    exec_path.write_bytes(artifact)
    os.chmod(exec_path, 0o755)
    run_cwd = os.getcwd() if not tool_fault or config.get('tool_strict') else '/tmp'
    proc = subprocess.run(['bash', str(exec_path)], cwd=run_cwd,
                          capture_output=True, text=True, timeout=600)
    (OUT / 'solve_stdout.txt').write_text(proc.stdout, encoding='utf-8', errors='replace')
    (OUT / 'solve_stderr.txt').write_text(proc.stderr, encoding='utf-8', errors='replace')
    tool_success = adapter_valid and proc.returncode == 0 \
        and (not tool_fault or bool(config.get('tool_strict')))

    # ---- verifier stage: REAL pytest run of the task's own tests ----
    verifier_drift = any(f['kind'] == 'VERIFIER_DRIFT' and applies(f, payload)
                         for f in faults)
    if verifier_drift:
        exercised.add('verifier_recheck')
    try:
        vproc = subprocess.run(
            [sys.executable, '-m', 'pytest', '/tests/test_outputs.py', '-rA',
             '--tb=short', '-q'],
            cwd=os.getcwd(), capture_output=True, text=True, timeout=900)
        tests_passed = vproc.returncode == 0
        (OUT / 'verifier_stdout.txt').write_text(
            vproc.stdout, encoding='utf-8', errors='replace')
        (OUT / 'verifier_stderr.txt').write_text(
            vproc.stderr, encoding='utf-8', errors='replace')
        verifier_ran = True
    except Exception:
        tests_passed, verifier_ran = False, False
    verifier_agrees = verifier_ran and (not verifier_drift
                                        or bool(config.get('verifier_recheck')))

    # terminal task outcome: the verifier's real reward, gated by the
    # pipeline contracts exactly like the simulated environment
    utility = 1 if (tool_success and verifier_agrees and tests_passed) else 0

    # ---- boundary contracts + activation events (execution facts) ----
    boundary = {
        'planner_valid': bool(planner_valid),
        'cache_fresh': bool(cache_fresh),
        'adapter_valid': bool(adapter_valid),
        'tool_success': bool(tool_success),
        'verifier_agrees': bool(verifier_agrees),
    }
    activation_events = {}
    if intervention is not None:
        for key in intervention.get('patch_keys', []):
            activation_events[key] = bool(key in exercised)

    (OUT / 'verifier' / 'reward.txt').write_text(str(utility) + '\n', encoding='utf-8')
    trace = {
        'task_id': os.environ.get('DACE_TASK_ID', ''),
        'boundary': boundary,
        'tests_passed': bool(tests_passed),
        'verifier_ran': bool(verifier_ran),
        'solve_exit': proc.returncode,
        'activation_events': activation_events,
        'intervention': intervention,
    }
    (OUT / 'trace.json').write_text(
        json.dumps(trace, indent=2, sort_keys=True), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
