"""Durable background orchestration for the high-frequency research pipeline."""
from __future__ import annotations

import json
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from alphagym import storage_io
from alphagym.config import resolve_root
from alphagym.hf_book import build_samples
from alphagym.hf_research import load_hf_spec, run_research


def _path(job_id):
    return f'factor_library/hf_jobs/{job_id}.json'


def _write(root, row):
    row = {**row, 'updated_at': datetime.now(UTC).isoformat()}
    storage_io.store_for(root, initialize=True).write_blob(
        _path(row['job_id']), json.dumps(row, ensure_ascii=False).encode())
    return row


def read_job(root, job_id):
    root = resolve_root(root)
    return json.loads(storage_io.store_for(root).read_blob(_path(job_id)))


def execute_pipeline(root, spec_path, job_id):
    root = resolve_root(root)
    spec_path = Path(spec_path).expanduser().resolve()
    row = read_job(root, job_id)
    try:
        spec = load_hf_spec(spec_path)
        row = _write(root, {**row, 'status': 'running', 'phase': 'reconstructing'})
        build = build_samples(root, start=spec['dates'][0], end=spec['dates'][-1])
        row = _write(root, {**row, 'phase': 'researching', 'build': build})
        report = run_research(root, spec_path)
        return _write(root, {**row, 'status': 'succeeded', 'phase': 'complete',
                             'report': report, 'error': None})
    except Exception as error:
        _write(root, {**row, 'status': 'failed', 'phase': 'failed',
                      'error': {'code': type(error).__name__, 'message': str(error)}})
        raise


def start_pipeline(root, spec_path, *, spawn=True):
    root = resolve_root(root)
    spec_path = Path(spec_path).expanduser().resolve()
    load_hf_spec(spec_path)
    job_id = 'hfjob-'+uuid.uuid4().hex[:16]
    row = _write(root, {'job_id': job_id, 'status': 'queued', 'phase': 'queued',
                        'spec_path': str(spec_path), 'root': str(root), 'error': None})
    if not spawn:
        return row
    process = subprocess.Popen(
        [sys.executable, '-m', 'alphagym.cli', 'hf', 'execute-pipeline',
         '--root', str(root), '--spec', str(spec_path), '--job-id', job_id, '--json'],
        cwd=str(Path(__file__).resolve().parents[2]), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    return {**row, 'pid': process.pid}


def execute_ml(root, spec_path, job_id):
    from alphagym.hf_ml import load_ml_spec, run_ml_research

    root = resolve_root(root)
    spec_path = Path(spec_path).expanduser().resolve()
    row = read_job(root, job_id)
    try:
        load_ml_spec(spec_path)
        row = _write(root, {**row, 'status': 'running', 'phase': 'combining'})
        report = run_ml_research(root, spec_path)
        return _write(root, {**row, 'status': 'succeeded', 'phase': 'complete',
                             'report': report, 'error': None})
    except Exception as error:
        _write(root, {**row, 'status': 'failed', 'phase': 'failed',
                      'error': {'code': type(error).__name__, 'message': str(error)}})
        raise


def start_ml(root, spec_path, *, spawn=True):
    from alphagym.hf_ml import load_ml_spec

    root = resolve_root(root)
    spec_path = Path(spec_path).expanduser().resolve()
    load_ml_spec(spec_path)
    job_id = 'hfmljob-'+uuid.uuid4().hex[:16]
    row = _write(root, {'job_id': job_id, 'job_type': 'hf_ml', 'status': 'queued',
                        'phase': 'queued', 'spec_path': str(spec_path), 'root': str(root),
                        'error': None})
    if not spawn:
        return row
    process = subprocess.Popen(
        [sys.executable, '-m', 'alphagym.cli', 'hf', 'execute-ml',
         '--root', str(root), '--spec', str(spec_path), '--job-id', job_id, '--json'],
        cwd=str(Path(__file__).resolve().parents[2]), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    return {**row, 'pid': process.pid}
