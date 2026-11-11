#!/usr/bin/env python3
"""Read-only FACET environment/public-release audit. No CUDA or task execution."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]


def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {"command": argv, "returncode": p.returncode,
                "stdout": p.stdout[-12000:], "stderr": p.stderr[-4000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": argv, "error": str(exc)}


def public_json(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return {"url": url, "data": json.load(response)}
    except Exception as exc:
        return {"url": url, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Host CLI, local build recipes, cluster access, public HF metadata; no GPU validation",
        "tools": {x: shutil.which(x) for x in ("runai", "kubectl", "docker", "uv", "python3")},
        "recipes": {},
        "cluster": command(["runai", "list"]),
    }
    for relative in ("docker/Dockerfile", "docker/archer-requirements.txt",
                     "docker/Dockerfile.asft", "docker/asft-extra-requirements.txt",
                     "LINEAGE-final/docker/Dockerfile"):
        path = WORKSPACE / relative
        if path.is_file():
            content = path.read_bytes()
            report["recipes"][relative] = {"sha256": hashlib.sha256(content).hexdigest(),
                                             "text": content.decode()}
    report["hf_datasets"] = public_json("https://huggingface.co/api/datasets?author=FACET-Terminal")
    report["hf_task_files"] = public_json(
        "https://huggingface.co/api/datasets/FACET-Terminal/FACET-Terminal-Tasks-6k/tree/main?recursive=true")
    result = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
        print(args.output)
    else:
        print(result)
    return 0 if report["cluster"].get("returncode") == 0 and all(
        "data" in report[key] for key in ("hf_datasets", "hf_task_files")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
