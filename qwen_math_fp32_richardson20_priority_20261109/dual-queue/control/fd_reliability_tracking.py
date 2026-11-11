"""Small stdlib-only experiment-board writers; no service or GPU side effects."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile


STATES = {"planned", "queued", "starting", "running", "restarting", "failed", "succeeded", "cancelled"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    """Replace one JSON snapshot atomically; reject NaN/Infinity."""
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def _run_dir(experiment_root, run_id):
    root = Path(experiment_root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run_id must be one safe directory component")
    run = (root / "runs" / run_id).resolve()
    if not run.is_relative_to(root):
        raise ValueError("Run path escapes experiment root")
    return root, run


def _relative(root, path):
    path = Path(path)
    if path.is_absolute():
        raise ValueError("Manifest source/artifact paths must be relative to experiment root")
    if not (root / path).resolve().is_relative_to(root):
        raise ValueError("Manifest source/artifact path escapes experiment root")
    return path.as_posix()


def register_run(experiment_root, run_id, title, kind="analysis", parameters=None,
                 job=None, sources=None, artifacts=None, tags=None, notes="", replace=False):
    """Register before submission. Existing records require explicit replace=True.

    Replacement preserves prior status, creation time and any unknown metadata.
    Sources and artifacts are always relative to the experiment directory.
    """
    root, run = _run_dir(experiment_root, run_id)
    if not (root / "experiment.json").is_file():
        raise ValueError("Register experiment.json before registering a run")
    prior = _read(run / "run.json")
    if prior and not replace:
        raise FileExistsError(run / "run.json")
    source_paths = {"status": f"runs/{run_id}/state.json",
                    "progress": f"runs/{run_id}/progress.json",
                    "results": f"runs/{run_id}/result.json"}
    source_paths.update(prior.get("sources", {}))
    source_paths.update(sources or {})
    for key, value in source_paths.items():
        if isinstance(value, list):
            source_paths[key] = [_relative(root, x) for x in value]
        else:
            source_paths[key] = _relative(root, value)
    linked = prior.get("artifacts", []) if artifacts is None else artifacts
    linked = [{**item, "path": _relative(root, item["path"])} for item in linked]
    manifest = {**prior, "schema_version": 1, "id": run_id, "title": title,
                "kind": kind, "status": prior.get("status", "planned"),
                "created_at": prior.get("created_at", utc_now()), "updated_at": utc_now(),
                "parameters": parameters if parameters is not None else prior.get("parameters", {}),
                "sources": source_paths, "artifacts": linked,
                "tags": tags if tags is not None else prior.get("tags", []),
                "notes": notes or prior.get("notes", "")}
    if job is not None:
        if not job.get("name") or not job.get("namespace"):
            raise ValueError("A submitted/pre-registered job requires name and namespace")
        manifest["job"] = job
    atomic_json(run / "run.json", manifest)
    if not (run / "state.json").exists():
        update_state(root, run_id, "planned", stage="planning", message="Registered before submission")
    return manifest


def update_state(experiment_root, run_id, state, stage=None, message=None, **fields):
    """Update process status, preserving prior attempts and failure evidence.

    Pass attempt_id when beginning a new attempt. A changed attempt archives
    the full previous state in attempts/<old-attempt>/state.json. Live job names
    belong in run.json; update them via register_run(..., replace=True).
    """
    if state not in STATES:
        raise ValueError(f"Unknown experiment state: {state}")
    _, run = _run_dir(experiment_root, run_id)
    if not (run / "run.json").is_file():
        raise ValueError("Register run before updating state")
    prior = _read(run / "state.json")
    now = utc_now()
    old_attempt = prior.get("attempt_id")
    new_attempt = fields.get("attempt_id", old_attempt)
    for attempt in (old_attempt, new_attempt):
        if attempt is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(attempt)):
            raise ValueError("attempt_id must be a safe directory component")
    changed_attempt = new_attempt is not None and new_attempt != old_attempt
    if changed_attempt and prior:
        label = str(old_attempt) if old_attempt is not None else "before-first-attempt"
        archive = run / "attempts" / label / "state.json"
        if archive.exists():
            archive = archive.with_name("state-" + now.replace(":", "-") + ".json")
        atomic_json(archive, prior)
    history = list(prior.get("state_history", []))
    if prior and (state != prior.get("state") or changed_attempt):
        history.append({key: value for key, value in prior.items() if key != "state_history"})
    record = {} if changed_attempt else dict(prior)
    if state not in {"succeeded", "failed", "cancelled"}:
        for key in ("finished_at", "exit_code", "error"):
            record.pop(key, None)
    record.update(fields)
    record.update(state=state, updated_at=now)
    if history:
        record["state_history"] = history
    if stage is not None:
        record["stage"] = stage
    if message is not None:
        record["message"] = message
    if state in {"starting", "running"}:
        record.setdefault("started_at", now)
    if state in {"succeeded", "failed", "cancelled"}:
        record.setdefault("finished_at", now)
    atomic_json(run / "state.json", record)
    return record


def update_progress(experiment_root, run_id, done=None, total=None, stage=None, **fields):
    _, run = _run_dir(experiment_root, run_id)
    if not (run / "run.json").is_file():
        raise ValueError("Register run before updating progress")
    if (done is None) != (total is None):
        raise ValueError("Supply both done and total, or neither")
    if done is not None and (not isinstance(done, int) or not isinstance(total, int)
                             or total <= 0 or done < 0 or done > total):
        raise ValueError("Progress requires integer 0 <= done <= positive total")
    value = {**fields, "updated_at": utc_now()}
    if stage is not None:
        value["stage"] = stage
    if done is not None:
        value.update(done=done, total=total)
    atomic_json(run / "progress.json", value)
    return value
