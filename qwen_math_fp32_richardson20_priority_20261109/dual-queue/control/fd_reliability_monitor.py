#!/usr/bin/env python3
"""Lightweight monitor for one registered FD experiment; never submits work.

Default cluster actions are reads (pods and terminal pod logs). Cleanup is
opt-in and restricted to current run.json mappings with archived terminal logs
and verified terminal state. Credentials are only consumed, never refreshed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from fd_reliability_tracking import atomic_json, utc_now
from fd_reliability_banks import file_hash, fingerprint, validate_chunk


KUBECONFIG = "/mnt/cvlab/scratch/cvlab/home/shuli/.codex/secrets/native-runai-kubeconfig.json"
NAMESPACE = "runai-cvlab-shuli"
DEFAULT_EXPERIMENT = Path("/mnt/cvlab/scratch/cvlab/home/shuli/data-rsi/qwen_math_fd_reliability_20261108")
TERMINAL = {"succeeded", "failed", "cancelled"}
POD_TERMINAL = {"Succeeded", "Failed"}
JOB_LABELS = ("workloadName", "release", "runai-job", "runai/job-name", "runai.org/job-name", "job-name", "batch.kubernetes.io/job-name")


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def inside(root, value):
    root = Path(root).resolve()
    value = Path(value)
    if value.is_absolute():
        raise ValueError("Expected an experiment-relative path")
    path = (root/value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path escapes the registered experiment")
    return path


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value):
        raise ValueError("Invalid Kubernetes/run identifier")
    return value


def kubectl(arguments):
    """Read Kubernetes directly; explicit cleanup uses the RunAI service API."""
    env = os.environ.copy()
    env["KUBECONFIG"] = KUBECONFIG
    if arguments and arguments[0] == "runai":
        if (len(arguments) != 6 or arguments[:3] != ["runai", "delete", "job"]
                or arguments[4:] != ["-p", "cvlab-shuli"]):
            raise ValueError("Only the explicit RunAI job cleanup transport is supported")
        safe_name(arguments[3])
        command = list(arguments)
    else:
        command = ["kubectl", "--request-timeout=15s", "-n", NAMESPACE, *arguments]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError(f"{command[0]} exited {result.returncode}: {(result.stderr or result.stdout).strip()[:1500]}")
    return result.stdout


def matches_job(pod, name):
    metadata = pod.get("metadata", {})
    if metadata.get("namespace", NAMESPACE) != NAMESPACE:
        return False
    labels = metadata.get("labels", {})
    return (any(labels.get(key) == name for key in JOB_LABELS)
            or any(owner.get("kind", "").lower() == "runaijob" and owner.get("name") == name
                   for owner in metadata.get("ownerReferences", [])))


def latest_replicas(pods, job_name):
    """Use the established RunAI replica-attempt label; retain uncertain pods."""
    latest = {}
    for pod in pods:
        metadata = pod["metadata"]
        index = metadata.get("labels", {}).get("runai/pod-index")
        if index is None:
            match = re.fullmatch(re.escape(job_name) + r"-(\d+)-(\d+)", metadata["name"])
            index = "-".join(match.groups()) if match else None
        if index and re.fullmatch(r"\d+-\d+", index):
            replica, attempt = map(int, index.split("-"))
        else:
            replica, attempt = metadata.get("uid", metadata["name"]), 0
        order = (attempt, metadata.get("creationTimestamp", ""))
        if replica not in latest or order > latest[replica][0]:
            latest[replica] = (order, pod)
    return [value[1] for value in latest.values()]


def pod_outcome(pods, job_name):
    latest = latest_replicas(pods, job_name)
    phases = [pod.get("status", {}).get("phase") for pod in latest]
    if not phases:
        return "missing"
    if any(phase == "Running" for phase in phases):
        return "running"
    if any(phase == "Pending" for phase in phases):
        return "starting" if any(p.get("spec", {}).get("nodeName") for p in latest) else "queued"
    if all(phase == "Succeeded" for phase in phases):
        return "succeeded"
    if all(phase in POD_TERMINAL for phase in phases) and any(phase == "Failed" for phase in phases):
        return "failed"
    return "unknown"


def job_terminal_outcome(job, name):
    """A terminal pod alone does not prove RunAI has exhausted its retries."""
    metadata, status = job.get("metadata", {}), job.get("status", {})
    if metadata.get("name") != name or metadata.get("namespace", NAMESPACE) != NAMESPACE:
        return None
    phase = str(status.get("phase", status.get("status", ""))).lower()
    if phase in {"succeeded", "completed", "complete"}:
        return "succeeded"
    if phase in {"failed", "error"}:
        return "failed"
    if phase:
        return None  # Unknown/active phases never authorize cleanup.
    terminal = set()
    for condition in status.get("conditions", []):
        if str(condition.get("status", "")).lower() != "true":
            continue
        label = str(condition.get("type", "")).lower()
        if label in {"complete", "completed", "succeeded"}: terminal.add("succeeded")
        if label in {"failed", "error"}: terminal.add("failed")
    return next(iter(terminal)) if len(terminal) == 1 else None


def termination_metadata(pod):
    metadata, status = pod.get("metadata", {}), pod.get("status", {})
    containers = []
    for key in ("initContainerStatuses", "containerStatuses"):
        for value in status.get(key, []):
            containers.append({"name": value.get("name"), "kind": key,
                               "restart_count": value.get("restartCount", 0),
                               "state": value.get("state", {}), "last_state": value.get("lastState", {})})
    return {"name": metadata.get("name"), "uid": metadata.get("uid"), "namespace": metadata.get("namespace", NAMESPACE),
            "created_at": metadata.get("creationTimestamp"), "labels": metadata.get("labels", {}),
            "phase": status.get("phase"), "reason": status.get("reason"), "message": status.get("message"),
            "node": pod.get("spec", {}).get("nodeName"), "containers": containers, "conditions": status.get("conditions", [])}


def archive_terminal_pod(run, pod, execute):
    metadata = pod["metadata"]
    name, uid = safe_name(metadata["name"]), safe_name(metadata.get("uid", metadata["name"]))
    folder = inside(run, f"cluster/pods/{uid}")
    folder.mkdir(parents=True, exist_ok=True)
    termination = folder/"termination.json"
    if not termination.exists():
        atomic_json(termination, {**termination_metadata(pod), "archived_at": utc_now()})
    marker = folder/"logs-archive.json"
    previous = read_json(marker, {})
    if previous.get("complete") and (folder/"pod.log").is_file():
        return previous
    try:
        log = execute(["logs", name, "--all-containers=true", "--timestamps=true"])
        temporary = folder/"pod.log.tmp"
        temporary.write_text(log)
        temporary.replace(folder/"pod.log")
        result = {"complete": True, "pod": name, "uid": uid, "archived_at": utc_now(),
                  "bytes": len(log.encode()), "sha256": hashlib.sha256(log.encode()).hexdigest()}
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        result = {"complete": False, "pod": name, "uid": uid, "attempted_at": utc_now(), "error": str(exc)}
    atomic_json(marker, result)
    return result


def bank_progress(run):
    bank = inside(run, "bank")
    manifest = read_json(bank/"manifest.json")
    if manifest is None:
        return None
    config = manifest["configuration"]
    rows = config["rows"]
    counts = {"probe": config["samples"], "target": config["targets"]}
    if (not rows or len(set(rows)) != len(rows) or any(not isinstance(n, int) or n <= 0 for n in counts.values())):
        raise ValueError("Invalid frozen bank counts")
    total = len(rows)*sum(counts.values())
    done, completed_rows, issues = 0, 0, []
    for row in rows:
        row_dir = inside(bank, f"row_{row:08d}")
        row_valid = True
        all_seeds = set()
        for kind, expected in counts.items():
            indices = set()
            for path in sorted((row_dir/kind).glob("chunk_*.json")):
                try:
                    if not path.resolve().is_relative_to(bank):
                        raise ValueError("chunk path escapes bank")
                    chunk = read_json(path)
                    records = chunk["records"]
                    new_indices = [record["sample_index"] for record in records]
                    new_seeds = [record["seed"] for record in records]
                    if (chunk["config_fingerprint"] != manifest["config_fingerprint"] or chunk["bank"] != kind
                            or len(records) != chunk["stop"]-chunk["start"]
                            or new_indices != list(range(chunk["start"],chunk["stop"]))
                            or new_seeds != chunk["seeds"] or len(set(new_seeds)) != len(new_seeds)
                            or set(new_seeds) & all_seeds or set(new_indices) & indices
                            or any(index < 0 or index >= expected for index in new_indices)):
                        raise ValueError("chunk identity/count/order/seed mismatch")
                    indices.update(new_indices); all_seeds.update(new_seeds)
                    done += len(records)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    issues.append({"path": str(path.relative_to(run)), "error": str(exc)})
                    row_valid = False
            row_valid &= len(indices) == expected
        completion = read_json(row_dir/"complete.json", {})
        if (row_valid and completion.get("config_fingerprint") == manifest["config_fingerprint"]
                and all(completion.get("banks", {}).get(kind, {}).get("records") == n for kind,n in counts.items())):
            completed_rows += 1
    return {"done": done, "total": total, "stage": "generation", "updated_at": utc_now(),
            "completed_rows": completed_rows, "expected_rows": len(rows), "issues": issues,
            "counter_scope": "actual records in validated published chunks; probe plus target"}


def enrich_bank_results(root, run, manifest, progress=None):
    """Expose already-recorded bank results and one unchanged real sample.

    Full content fingerprints are checked once per immutable source inventory.
    Subsequent polls reuse that verification only when manifest/complete/prompt
    hashes and every chunk's path, size and mtime are unchanged.
    """
    root,run=Path(root).resolve(),Path(run).resolve()
    progress=bank_progress(run) if progress is None else progress
    if (not progress or progress["done"]!=progress["total"] or progress["issues"]
            or progress["completed_rows"]!=progress["expected_rows"]):
        return {"enriched":False,"reason":"bank incomplete"}
    bank=inside(run,"bank")
    bank_manifest=read_json(bank/"manifest.json")
    config=bank_manifest["configuration"]
    config_hash=bank_manifest["config_fingerprint"]
    if fingerprint(config)!=config_hash:
        raise ValueError("Bank manifest configuration fingerprint mismatch")
    result_path=inside(root,manifest.get("sources",{}).get("results",f"runs/{run.name}/result.json"))
    existing=read_json(result_path)
    if not isinstance(existing,dict) or not existing:
        return {"enriched":False,"reason":"worker result artifact not available"}
    inventory={"manifest_sha256":file_hash(bank/"manifest.json"),"rows":{},"chunks":{}}
    for row in config["rows"]:
        row_dir=inside(bank,f"row_{row:08d}")
        inventory["rows"][str(row)]={name:file_hash(inside(row_dir,name)) for name in ("prompt.json","complete.json")}
        for kind in ("probe","target"):
            for path in sorted((row_dir/kind).glob("chunk_*.json")):
                path=inside(run,path.relative_to(run))
                stat=path.stat()
                inventory["chunks"][str(path.relative_to(run))]={"size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}
    sample_path=inside(run,"first_probe_sample.json")
    sample_relative=str(sample_path.relative_to(root))
    previous=existing.get("bank_recording",{})
    cached=(previous.get("fingerprints_verified") and previous.get("config_fingerprint")==config_hash
            and previous.get("source_inventory")==inventory and sample_path.is_file()
            and previous.get("sample_sha256")==file_hash(sample_path))
    if not cached:
        totals={kind:{"records":0,"correct":0,"truncated":0,"tokens":0} for kind in ("probe","target")}
        first_sample=None
        seen_seeds=set()
        for row in config["rows"]:
            row_dir=inside(bank,f"row_{row:08d}")
            prompt_payload=read_json(row_dir/"prompt.json")
            prompt=prompt_payload["prompt"]
            prompt_hash=fingerprint(prompt)
            completion=read_json(row_dir/"complete.json")
            if (prompt_payload.get("config_fingerprint")!=config_hash
                    or prompt_payload.get("prompt_fingerprint")!=prompt_hash
                    or completion.get("config_fingerprint")!=config_hash
                    or completion.get("prompt_fingerprint")!=prompt_hash
                    or prompt.get("row_index")!=row
                    or prompt.get("prompt_ids_sha256")!=fingerprint(prompt["prompt_ids"])):
                raise ValueError(f"Bank row {row} prompt/completion fingerprint mismatch")
            for kind,count_key in (("probe","samples"),("target","targets")):
                records=[]
                for path in sorted((row_dir/kind).glob("chunk_*.json")):
                    chunk=read_json(path)
                    metadata={key:chunk[key] for key in ("protocol","config_fingerprint","prompt_fingerprint",
                        "bank","start","stop","seeds","shard","shards")}
                    if (metadata["config_fingerprint"]!=config_hash or metadata["prompt_fingerprint"]!=prompt_hash
                            or metadata["bank"]!=kind or metadata["start"]!=len(records)):
                        raise ValueError(f"Bank chunk provenance/order mismatch: {path}")
                    loaded=validate_chunk(path,metadata,prompt["prompt_ids_sha256"])
                    for record in loaded:
                        if record["seed"] in seen_seeds:
                            raise ValueError("Duplicate seed across complete bank records")
                        seen_seeds.add(record["seed"])
                        if kind=="probe" and (first_sample is None or record["seed"]<first_sample["record"]["seed"]):
                            first_sample={"artifact_kind":"unchanged real probe sample","config_fingerprint":config_hash,
                                "row_index":row,"prompt":prompt,"record":record,
                                "source_chunk":str(path.relative_to(root)),"source_chunk_sha256":file_hash(path),
                                "source_prompt":str((row_dir/"prompt.json").relative_to(root)),
                                "source_prompt_sha256":inventory["rows"][str(row)]["prompt.json"],
                                "selection":"minimum actual probe seed; no correctness or length filtering"}
                    records.extend(loaded)
                saved=completion["banks"][kind]
                count=len(records)
                correct=sum(record["correct"] for record in records)
                truncated=sum(record["truncated"] for record in records)
                tokens=sum(record["response_tokens"] for record in records)
                if (count!=config[count_key] or saved["records"]!=count
                        or saved["records_sha256"]!=fingerprint(records)
                        or saved["correct"]!=correct or saved["truncated"]!=truncated
                        or not math.isclose(saved["accuracy"],correct/count,rel_tol=1e-12,abs_tol=1e-12)
                        or not math.isclose(saved["mean_response_tokens"],tokens/count,rel_tol=1e-12,abs_tol=1e-12)):
                    raise ValueError(f"Bank row {row} {kind} saved summary differs from fingerprint-verified records")
                for key,value in (("records",count),("correct",correct),("truncated",truncated),("tokens",tokens)):
                    totals[kind][key]+=value
        if first_sample is None:
            raise ValueError("Complete bank has no real probe sample")
        metrics={}
        for kind,values in totals.items():
            metrics.update({f"{kind}_records":values["records"],f"{kind}_correct":values["correct"],
                f"{kind}_accuracy_percent":100*values["correct"]/values["records"],
                f"{kind}_truncations":values["truncated"],f"{kind}_mean_tokens":values["tokens"]/values["records"]})
        atomic_json(sample_path,first_sample)
        enriched={**existing,"metrics":{**existing.get("metrics",{}),**metrics},
            "bank_recording":{"fingerprints_verified":True,"verified_at":utc_now(),"config_fingerprint":config_hash,
                "source_inventory":inventory,"metrics":metrics,"sample_artifact":sample_relative,
                "sample_sha256":file_hash(sample_path),
                "scope":"Existing generated records and unchanged project-verifier outcomes; no new generation or relabeling"}}
        # Preserve any wrapper fields added since the initial read.
        latest=read_json(result_path,{})
        enriched={**latest,**{k:v for k,v in enriched.items() if k in ("metrics","bank_recording")},
                  "metrics":{**latest.get("metrics",{}),**metrics}}
        atomic_json(result_path,enriched)
    else:
        metrics=previous["metrics"]
    current_manifest=read_json(run/"run.json")
    artifacts=current_manifest.setdefault("artifacts",[])
    if not any(item.get("path")==sample_relative for item in artifacts):
        artifacts.append({"label":"首个 seed 的真实 probe 样本（原始评分）","path":sample_relative})
        atomic_json(run/"run.json",current_manifest)
    return {"enriched":True,"cached_verification":bool(cached),"metrics":metrics,"sample_artifact":sample_relative}


def validate_results(root, run, manifest, progress=None):
    """Check actual saved results and required artifact files, never just exit 0."""
    issues = []
    result_path = inside(root, manifest.get("sources", {}).get("results", f"runs/{run.name}/result.json"))
    try:
        result = read_json(result_path)
        if not isinstance(result, dict) or not result:
            issues.append("missing or empty structured result artifact")
    except (OSError, ValueError) as exc:
        issues.append(f"result artifact unreadable: {exc}")
    spec = read_json(run/"spec.json", {})
    for name in spec.get("required_artifacts", []):
        try:
            artifact = inside(run, name)
            if not artifact.is_file() or artifact.stat().st_size == 0:
                issues.append(f"missing/empty required artifact: {name}")
            elif artifact.suffix == ".json":
                read_json(artifact)
        except (OSError, ValueError) as exc:
            issues.append(f"invalid required artifact {name}: {exc}")
    if run.name.startswith("bank"):
        if (not progress or progress["done"] != progress["total"]
                or progress["completed_rows"] != progress["expected_rows"] or progress["issues"]):
            issues.append("bank records or row completion artifacts are incomplete")
    return {"verified": not issues, "issues": issues, "checked_at": utc_now(), "result_path": str(result_path)}


def write_state_if_allowed(path, outcome, cluster, verification):
    """Worker stages and terminal facts take priority over monitor observations."""
    local = read_json(path, {})
    current = local.get("state", "planned")
    if current in TERMINAL:
        return local
    if outcome in ("queued", "starting") and current not in ("planned", "queued", "starting"):
        return local
    if outcome not in ("queued", "starting", "failed", "succeeded"):
        return local
    if outcome == "succeeded" and not verification["verified"]:
        return local  # Preserve worker state, expose artifact failure in observation.
    record = {**local, "state": outcome, "updated_at": utc_now(),
              "monitor_job_name": cluster["job_name"], "monitor_source": "current mapped RunAI pods"}
    if outcome in ("queued", "starting"):
        record.update(stage="waiting for resources" if outcome == "queued" else "container startup")
    else:
        record.setdefault("finished_at", utc_now())
        record["cluster_termination"] = cluster["current_pods"]
        codes = [item["state"]["terminated"].get("exitCode") for pod in cluster["current_pods"]
                 for item in pod["containers"] if "terminated" in item["state"]]
        if codes:
            record["exit_code"] = next((code for code in codes if code), codes[-1])
        if outcome == "failed":
            record["error"] = "Mapped RunAI pod terminated unsuccessfully; see cluster/pods termination and logs"
        else:
            record["message"] = "Mapped pods succeeded and saved result/required artifacts verified"
    # Recheck immediately before replacement; do not clobber a worker transition.
    if read_json(path, {}) == local:
        atomic_json(path, record)
        return record
    return read_json(path, {})


def monitor_once(experiment, cleanup_completed=False, execute=kubectl):
    root = Path(experiment).resolve()
    if not (root/"experiment.json").is_file():
        raise ValueError("Expected a registered experiment root")
    status = {"observed_at": utc_now(), "experiment": str(root), "cluster_ok": False,
              "cleanup_enabled": cleanup_completed, "runs": [], "errors": []}
    mappings = []
    job_counts = defaultdict(int)
    for path in sorted((root/"runs").glob("*/run.json")):
        if not path.resolve().is_relative_to(root):
            continue
        manifest = read_json(path)
        job = manifest.get("job", {})
        if job.get("provider", "runai") != "runai" or not job.get("name") or job.get("namespace", NAMESPACE) != NAMESPACE:
            continue
        name = safe_name(job["name"])
        mappings.append((path.parent, manifest, name)); job_counts[name] += 1
    try:
        pods = json.loads(execute(["get", "pods", "-o", "json"]))["items"]
        status["cluster_ok"] = True
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
        status["errors"].append(str(exc))
        atomic_json(root/"monitoring/status.json", status)
        return status  # Credential/API failure is not an experiment failure.
    for run, manifest, name in mappings:
        try:
            if job_counts[name] != 1:
                raise ValueError("Job maps to multiple runs; refusing ambiguous monitoring")
            matched = [pod for pod in pods if matches_job(pod, name)]
            outcome = pod_outcome(matched, name)
            observation = {"job_name": name, "namespace": NAMESPACE, "outcome": outcome,
                           "observed_at": utc_now(), "pods": [termination_metadata(pod) for pod in matched],
                           "current_pods": [termination_metadata(pod) for pod in latest_replicas(matched,name)]}
            archives = {}
            for pod in matched:
                if pod.get("status", {}).get("phase") in POD_TERMINAL:
                    archives[pod["metadata"].get("uid", pod["metadata"]["name"])] = archive_terminal_pod(run,pod,execute)
            progress = bank_progress(run) if run.name.startswith("bank") else None
            if progress is not None:
                progress_path = inside(root, manifest.get("sources", {}).get("progress", f"runs/{run.name}/progress.json"))
                atomic_json(progress_path, progress)
            verification = validate_results(root,run,manifest,progress) if outcome == "succeeded" else {"verified":False,"issues":[]}
            current_mapping = read_json(run/"run.json").get("job",{})
            if (current_mapping.get("name") != name or current_mapping.get("namespace",NAMESPACE) != NAMESPACE):
                status["runs"].append({"run_id":run.name,"job_name":name,"skipped":"run.json job mapping changed during observation"})
                continue
            state_path = inside(root, manifest.get("sources", {}).get("status", f"runs/{run.name}/state.json"))
            local = write_state_if_allowed(state_path,outcome,observation,verification)
            if local.get("state") == "succeeded" and outcome != "succeeded":
                verification = validate_results(root,run,manifest,progress)
            if (run.name.startswith("bank") and local.get("state")=="succeeded" and verification["verified"]):
                observation["bank_recording"]=enrich_bank_results(root,run,manifest,progress)
            observation.update(worker_state=local.get("state"), result_verification=verification, archives=archives)
            if (local.get("state") == "succeeded" and verification["verified"] and manifest.get("kind") == "validation"
                    and not run.name.startswith(("audit","bank"))):
                progress_path = inside(root,manifest.get("sources",{}).get("progress",f"runs/{run.name}/progress.json"))
                atomic_json(progress_path,{"done":1,"total":1,"stage":"saved validation artifact verified","updated_at":utc_now()})
            atomic_json(run/"cluster/observation.json",observation)
            if (cleanup_completed and matched and outcome in ("failed","succeeded")
                    and local.get("state") == outcome
                    and (outcome == "failed" or verification["verified"])
                    and all(p.get("status",{}).get("phase") in POD_TERMINAL for p in matched)
                    and all(archives.get(p["metadata"].get("uid",p["metadata"]["name"]),{}).get("complete") for p in matched)):
                current = read_json(run/"run.json")
                job_resource = json.loads(execute(["get","runaijob",name,"-o","json"]))
                job_terminal = job_terminal_outcome(job_resource,name)
                # Re-list immediately before deletion to catch a newly active retry.
                refreshed = json.loads(execute(["get","pods","-o","json"]))["items"]
                now_matched = [pod for pod in refreshed if matches_job(pod,name)]
                expected_uids = {p["metadata"].get("uid",p["metadata"]["name"]) for p in matched}
                current_uids = {p["metadata"].get("uid",p["metadata"]["name"]) for p in now_matched}
                still_mapped = current.get("job",{}).get("name") == name and current.get("job",{}).get("namespace",NAMESPACE) == NAMESPACE
                if (still_mapped and job_terminal == outcome and current_uids == expected_uids
                        and all(p.get("status",{}).get("phase") in POD_TERMINAL for p in now_matched)
                        and pod_outcome(now_matched,name) == outcome
                        and read_json(state_path,{}).get("state") == outcome):
                    response = execute(["runai","delete","job",name,"-p","cvlab-shuli"])
                    atomic_json(run/"cluster/cleanup.json",{"job_name":name,"namespace":NAMESPACE,
                                "deleted_at":utc_now(),"pod_uids":sorted(current_uids),"response":response.strip(),
                                "logs_archived":True,"verified_state":outcome,
                                "job_uid":job_resource.get("metadata",{}).get("uid"),"job_status":job_resource.get("status",{})})
                    observation["cleanup"] = "deleted terminal mapped job"
                else:
                    observation["cleanup"] = "skipped: job terminal status, mapping, state, or pod set not verified"
            status["runs"].append({"run_id":run.name,**observation})
        except (OSError,ValueError,KeyError,TypeError,RuntimeError,subprocess.TimeoutExpired) as exc:
            status["errors"].append({"run_id":run.name,"job_name":name,"error":str(exc)})
    atomic_json(root/"monitoring/status.json",status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,default=DEFAULT_EXPERIMENT)
    parser.add_argument("--once",action="store_true")
    parser.add_argument("--interval",type=float,default=30)
    parser.add_argument("--max-seconds",type=float,help="Required for a bounded monitor loop")
    parser.add_argument("--cleanup-completed",action="store_true")
    args=parser.parse_args()
    if args.interval < 1 or (not args.once and (args.max_seconds is None or args.max_seconds <= 0)):
        parser.error("Use --once, or a positive --max-seconds for a controlled loop; interval >=1")
    # One monitor process owns this experiment; a second loop must not race it.
    import fcntl
    lock_path=args.experiment/"monitoring/monitor.lock"
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open("a+") as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: parser.error("Another monitor already owns this experiment")
        start=time.monotonic()
        while True:
            result=monitor_once(args.experiment,args.cleanup_completed)
            print(json.dumps({"observed_at":result["observed_at"],"cluster_ok":result["cluster_ok"],
                              "runs":len(result["runs"]),"errors":result["errors"]}),flush=True)
            if args.once or time.monotonic()-start >= args.max_seconds:
                break
            time.sleep(min(args.interval, max(0,args.max_seconds-(time.monotonic()-start))))


if __name__ == "__main__":
    main()
