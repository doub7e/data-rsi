#!/usr/bin/env python3
"""Freeze offline probe/target banks for FD reliability audits.

Rows are zero-based physical rows of the *resolved parquet file*. They are
never indices in an ability-filtered table or a teacher-screened pool. A file
argument is used exactly; a directory prefers DeepMath's original
train_filtered_level6.parquet, then math_and_code/train.parquet. Use inspect
before freezing --rows; row IDs from different source files are not portable.

Outputs: manifest.json; row_<id>/prompt.json; probe/chunk_<start>_<stop>.json;
target/chunk_<start>_<stop>.json; and complete.json per row. Chunk stop indices
are exclusive. Each record refers to prompt.json by its prompt_ids_sha256.
No correctness, length, truncation, or homogeneous-outcome filtering occurs.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import tempfile


PROTOCOL = "fd_reliability_banks_v1"
ROW_SEMANTICS = "zero-based physical parquet row; no filtering or reordering"
GRADER_DESCRIPTION = "verbatim project math_verify.compute_score, same DeepMath default dispatch"


def validate_prompt_token_ids(value):
    """Require a nonempty flat integer sequence; never iterate mapping keys."""
    if (not isinstance(value, (list, tuple)) or not value
            or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in value)):
        raise ValueError("prompt token IDs must be a nonempty flat list of nonnegative integers, not bools or mapping keys")
    return list(value)


def render_prompt_ids(tokenizer, messages):
    """Render the unchanged chat text, then explicitly encode without extra special tokens."""
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    if not isinstance(rendered, str):
        raise TypeError("tokenize=False must render one chat prompt as a string")
    return validate_prompt_token_ids(tokenizer.encode(rendered, add_special_tokens=False))


def flatten_tokenized_prompt(value):
    """Normalize tokenize=True output only for equivalence checks, including BatchEncoding."""
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("tokenized chat mapping has no input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("expected exactly one tokenized chat prompt")
        value = value[0]
    return validate_prompt_token_ids(value)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return jsonable(value.item())
    raise TypeError(f"Unsupported artifact value: {type(value).__name__}")


def write_once(path, value):
    """Publish a complete JSON atomically, refusing any different existing value."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if canonical(json.loads(path.read_text())) != canonical(value):
            raise ValueError(f"Refusing to overwrite different artifact: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if canonical(json.loads(path.read_text())) != canonical(value):
                raise ValueError(f"Concurrent different artifact: {path}")
    finally:
        Path(temporary).unlink(missing_ok=True)


def parse_rows(text):
    if text is None:
        return None
    try:
        rows = [int(part.strip()) for part in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--rows must be comma-separated integers") from exc
    if not rows or min(rows) < 0 or len(set(rows)) != len(rows):
        raise argparse.ArgumentTypeError("--rows must contain unique nonnegative row IDs")
    return rows


def resolve_dataset(path):
    path = Path(path).expanduser().resolve(strict=True)
    if path.is_file():
        return path, "explicit file; no substitution"
    candidates = [
        path / "DeepMath-103K/train_filtered_level6.parquet",
        path / "train_filtered_level6.parquet",
        path / "math_and_code/train.parquet",
        path / "train.parquet",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), "directory preference: original DeepMath, then combined training parquet"
    raise FileNotFoundError(f"No supported training parquet under {path}; pass its exact file path")


def dataset_identity(path, parquet):
    # Footer + size/mtime avoids hashing a 1.6GB dataset independently on every
    # generation worker. This is explicitly a metadata identity, not content SHA.
    stat = path.stat()
    with path.open("rb") as handle:
        handle.seek(-8, 2)
        tail = handle.read(8)
        if tail[4:] != b"PAR1":
            raise ValueError(f"Not an ordinary parquet file: {path}")
        footer_size = struct.unpack("<I", tail[:4])[0]
        if footer_size > stat.st_size - 8:
            raise ValueError("Invalid parquet footer size")
        handle.seek(-8 - footer_size, 2)
        footer_hash = hashlib.sha256(handle.read(footer_size + 8)).hexdigest()
    return {
        "path": str(path), "identity_kind": "path_size_mtime_parquet_footer",
        "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "parquet_footer_sha256": footer_hash, "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "schema": str(parquet.schema_arrow), "row_semantics": ROW_SEMANTICS,
    }


def read_rows(parquet, rows):
    total = parquet.metadata.num_rows
    if any(index >= total for index in rows):
        raise ValueError(f"Row IDs exceed parquet length {total}: {rows}")
    wanted = set(rows)
    found = {}
    offset = 0
    for group in range(parquet.metadata.num_row_groups):
        count = parquet.metadata.row_group(group).num_rows
        selected = sorted(index for index in wanted if offset <= index < offset + count)
        if selected:
            table = parquet.read_row_group(group)
            values = table.take([index - offset for index in selected]).to_pylist()
            found.update(zip(selected, values, strict=True))
        offset += count
    return [found[index] for index in rows]


def as_prompt(row, row_index):
    # Refuse anonymous/raw-question adapters: changing template or grader here
    # would silently change the audited problem and its row semantics.
    if not isinstance(row.get("prompt"), (list, tuple)) or not isinstance(row.get("reward_model"), dict):
        raise ValueError("Expected project parquet columns prompt and reward_model; inspect schema before adapting raw data")
    if row.get("ability") not in (None, "math"):
        raise ValueError(f"Row {row_index} is not Math: ability={row.get('ability')!r}")
    source = row.get("data_source")
    if not isinstance(source, str) or "DeepMath" not in source:
        raise ValueError(f"Row {row_index} has unsupported data_source={source!r}; no grader fallback")
    if "ground_truth" not in row["reward_model"]:
        raise ValueError(f"Row {row_index} has no reward_model.ground_truth")
    messages = jsonable(row["prompt"])
    if not messages or any(not isinstance(m, dict) or not isinstance(m.get("content"), str)
                           or not isinstance(m.get("role"), str) for m in messages):
        raise ValueError(f"Row {row_index} has invalid chat messages")
    return {"row_index": row_index, "row_semantics": ROW_SEMANTICS,
            "messages": messages, "answer": str(row["reward_model"]["ground_truth"]),
            "data_source": source, "extra_info": jsonable(row.get("extra_info") or {})}


def model_identity(path, revision):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError("--model must be a cached local model directory")
    small = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
             "merges.txt", "vocab.json", "model.safetensors.index.json", "pytorch_model.bin.index.json")
    assets = {name: file_hash(path / name) for name in small if (path / name).is_file()}
    weights = []
    for item in sorted(list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))):
        stat = item.stat()
        weights.append({"name": item.name, "resolved_path": str(item.resolve()),
                        "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if "config.json" not in assets or not weights or any(w["size_bytes"] <= 0 for w in weights):
        raise ValueError(f"Model config or nonempty weights missing under {path}")
    snapshot = path.name if path.parent.name == "snapshots" else None
    if revision and snapshot and revision != snapshot:
        raise ValueError("--revision disagrees with local snapshot directory")
    return {"path": str(path), "revision": revision or snapshot,
            "metadata_sha256": assets, "weight_identity_kind": "resolved_path_size_mtime",
            "weights": weights}


def package_versions():
    result = {}
    for name in ("vllm", "torch", "transformers", "pyarrow", "math-verify"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def load_project_math_grader(root=None):
    """Load DeepMath's exact project grader without importing verl's runtime.

    The project file catches missing Math-Verify imports and can then return
    zero for every answer. Import the dependency explicitly and exercise both
    known-positive and known-negative cases before allocating a vLLM engine.
    No answer extraction, parsing, verification, or reward semantics change.
    """
    import math_verify  # Fail before model loading if the dependency is absent.

    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    source = (root / "verl/verl/utils/reward_score/math_verify.py").resolve(strict=True)
    dispatch = (root / "verl/verl/utils/reward_score/__init__.py").resolve(strict=True)
    source_sha256 = file_hash(source)
    module_name = "_opd_fd_verbatim_math_verify_" + source_sha256[:16]
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the exact project math grader: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != source or file_hash(source) != source_sha256:
        raise RuntimeError("Project grader source changed while loading")
    compute_score = module.compute_score
    positive = compute_score(r"\boxed{2}", "2")
    negative = compute_score(r"\boxed{3}", "2")
    if positive != 1.0 or negative != 0.0:
        raise RuntimeError(f"Project Math-Verify sanity failed: boxed2/2={positive!r}, boxed3/2={negative!r}")
    dependency_file = getattr(math_verify, "__file__", None)
    metadata = {
        "description": GRADER_DESCRIPTION,
        "project_source": str(source), "project_source_sha256": source_sha256,
        "dispatch_source": str(dispatch), "dispatch_source_sha256": file_hash(dispatch),
        "module_name": module.__name__, "module_file": module.__file__,
        "module_spec_origin": module.__spec__.origin,
        "dependency_module_file": dependency_file,
        "dependency_module_sha256": file_hash(dependency_file) if dependency_file and Path(dependency_file).is_file() else None,
        "sanity": {"boxed2_ground_truth2": float(positive), "boxed3_ground_truth2": float(negative), "passed": True},
    }
    return compute_score, metadata


def seed_for(args, row_position, bank, sample_index):
    return args.seed_start + row_position * (args.samples + args.targets) + (
        args.samples if bank == "target" else 0) + sample_index


def chunks_for(args, row_position, row_root, config_hash, prompt_hash):
    chunks = []
    for bank, count in (("probe", args.samples), ("target", args.targets)):
        for start in range(0, count, args.chunk_size):
            stop = min(start + args.chunk_size, count)
            metadata = {"protocol": PROTOCOL, "config_fingerprint": config_hash,
                        "prompt_fingerprint": prompt_hash, "bank": bank, "start": start, "stop": stop,
                        "seeds": [seed_for(args, row_position, bank, i) for i in range(start, stop)],
                        "shard": row_position % args.shards, "shards": args.shards}
            chunks.append((row_root / bank / f"chunk_{start:06d}_{stop:06d}.json", metadata))
    return chunks


def validate_chunk(path, metadata, prompt_ids_hash):
    payload = json.loads(path.read_text())
    for key, value in metadata.items():
        if payload.get(key) != value:
            raise ValueError(f"Existing chunk differs at {key}: {path}")
    records = payload.get("records", [])
    if [r.get("seed") for r in records] != metadata["seeds"]:
        raise ValueError(f"Existing chunk has missing/reordered seeds: {path}")
    for index, record in enumerate(records, start=metadata["start"]):
        if (record.get("sample_index") != index or record.get("bank") != metadata["bank"]
                or record.get("prompt_ids_sha256") != prompt_ids_hash
                or record.get("response_tokens") != len(record.get("token_ids", []))
                or not isinstance(record.get("text"), str)
                or not isinstance(record.get("correct"), bool)
                or record["correct"] != (float(record["reward"]) > 0)):
            raise ValueError(f"Invalid existing record in {path}")
    if payload.get("records_sha256") != fingerprint(records):
        raise ValueError(f"Record checksum mismatch: {path}")
    return records


def inspect(args, parquet, identity, selection):
    rows = args.rows if args.rows is not None else list(range(min(3, identity["rows"])))
    report = {"protocol": PROTOCOL, "mode": "inspect", "dataset": identity,
              "dataset_resolution": selection, "selected_rows": rows, "previews": []}
    for index, row in zip(rows, read_rows(parquet, rows), strict=True):
        preview = {"row_index": index, "columns": list(row), "ability": row.get("ability"),
                   "data_source": row.get("data_source")}
        try:
            prompt = as_prompt(row, index)
            preview.update(prompt_fingerprint=fingerprint(prompt), answer=prompt["answer"],
                           prompt_preview=" ".join(m["content"] for m in prompt["messages"])[:400],
                           message_roles=[m["role"] for m in prompt["messages"]])
        except ValueError as exc:
            preview["schema_error"] = str(exc)
        report["previews"].append(preview)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def sample(args, parquet, identity, selection):
    # Fail fast on a missing/broken grader, before tokenizer or GPU setup.
    compute_score, grader_runtime = load_project_math_grader()
    model = model_identity(args.model, args.revision)
    output = args.output.expanduser().resolve()
    root = Path(__file__).resolve().parents[1]
    grader_dir = root / "verl/verl/utils/reward_score"
    grader_sources = {str(p.relative_to(root)): file_hash(p) for p in sorted(grader_dir.glob("*.py"))}
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model["path"], local_files_only=True)
    prompts = []
    for index, row in zip(args.rows, read_rows(parquet, args.rows), strict=True):
        prompt = as_prompt(row, index)
        ids = render_prompt_ids(tokenizer, prompt["messages"])
        prompt.update(prompt_ids=ids, prompt_ids_sha256=fingerprint(ids))
        prompts.append(prompt)
    max_model_len = args.max_model_len or max(len(p["prompt_ids"]) for p in prompts) + args.max_tokens
    if any(len(p["prompt_ids"]) + args.max_tokens > max_model_len for p in prompts):
        raise ValueError("--max-model-len cannot truncate the declared prompt plus response cap")
    config = {"protocol": PROTOCOL, "output": str(output), "model": model,
              "prompt_encoding": "apply_chat_template(tokenize=False, enable_thinking=False, add_generation_prompt=True); encode(add_special_tokens=False)",
              "dataset": identity, "dataset_resolution": selection, "rows": args.rows,
              "prompt_fingerprints": [fingerprint(p) for p in prompts],
              "samples": args.samples, "targets": args.targets, "seed_start": args.seed_start,
              "seed_rule": "seed_start + row_list_position*(samples+targets) + bank_offset + sample_index; target offset=samples",
              "chunk_size": args.chunk_size, "shards": args.shards,
              "shard_rule": "row_list_position modulo shards; probe and target stay on same worker",
              "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                           "max_tokens": args.max_tokens, "n": 1, "enable_thinking": False,
                           "skip_special_tokens": True},
              "engine": {"dtype": args.dtype, "tensor_parallel_size": 1,
                         "max_num_seqs": args.max_num_seqs, "max_model_len": max_model_len,
                         "max_num_batched_tokens": args.max_num_batched_tokens,
                         "gpu_memory_utilization": args.gpu_memory_utilization,
                         "enforce_eager": True, "enable_prefix_caching": False,
                         "enable_chunked_prefill": True, "seed": args.seed_start},
              "grader": GRADER_DESCRIPTION,
              "grader_reward_interpretation": "correct=score>0; unchanged text",
              "grader_runtime": grader_runtime,
              "grader_source_sha256": grader_sources, "script_sha256": file_hash(__file__),
              "packages": package_versions()}
    config_hash = fingerprint(config)
    write_once(output / "manifest.json", {"config_fingerprint": config_hash, "configuration": config})
    pending_rows = []
    for position, prompt in enumerate(prompts):
        if position % args.shards != args.shard:
            continue
        row_root = output / f"row_{prompt['row_index']:08d}"
        prompt_hash = fingerprint(prompt)
        write_once(row_root / "prompt.json", {"config_fingerprint": config_hash,
                   "prompt_fingerprint": prompt_hash, "prompt": prompt})
        chunks = chunks_for(args, position, row_root, config_hash, prompt_hash)
        for path, metadata in chunks:
            if path.exists():
                validate_chunk(path, metadata, prompt["prompt_ids_sha256"])
        pending_rows.append((prompt, row_root, chunks))
    needs_gpu = any(not path.exists() for _, _, chunks in pending_rows for path, _ in chunks)
    llm = None
    if needs_gpu:
        from vllm import LLM, SamplingParams
        llm = LLM(model=model["path"], tokenizer=model["path"], **config["engine"])
    for prompt, row_root, chunks in pending_rows:
        for path, metadata in chunks:
            if path.exists():
                continue
            params = [SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, n=1,
                      max_tokens=args.max_tokens, skip_special_tokens=True, seed=seed)
                      for seed in metadata["seeds"]]
            outputs = llm.generate([{"prompt_token_ids": prompt["prompt_ids"]} for _ in params],
                                   sampling_params=params, use_tqdm=False)
            records = []
            for offset, (seed, request) in enumerate(zip(metadata["seeds"], outputs, strict=True)):
                if list(request.prompt_token_ids) != prompt["prompt_ids"] or len(request.outputs) != 1:
                    raise ValueError("vLLM output does not match the frozen request")
                completion = request.outputs[0]
                result = compute_score(completion.text, prompt["answer"])
                details = jsonable(result if isinstance(result, dict) else {"score": float(result)})
                reward = float(details["score"])
                if not math.isfinite(reward):
                    raise ValueError("Project grader returned a nonfinite score")
                ids = list(completion.token_ids)
                records.append({"row_index": prompt["row_index"], "bank": metadata["bank"],
                    "sample_index": metadata["start"] + offset, "seed": seed,
                    "prompt_ids_sha256": prompt["prompt_ids_sha256"], "token_ids": ids,
                    "text": completion.text, "reward": reward, "correct": reward > 0,
                    "score_details": details, "finish_reason": completion.finish_reason,
                    "stop_reason": jsonable(completion.stop_reason), "response_tokens": len(ids),
                    "truncated": completion.finish_reason == "length"})
            write_once(path, {**metadata, "records": records, "records_sha256": fingerprint(records)})
            print(json.dumps({"written": str(path), "records": len(records),
                  "correct": sum(r["correct"] for r in records),
                  "truncated": sum(r["truncated"] for r in records)}), flush=True)
        banks = {"probe": [], "target": []}
        for path, metadata in chunks:
            banks[metadata["bank"]].extend(validate_chunk(path, metadata, prompt["prompt_ids_sha256"]))
        summaries = {}
        for bank, count in (("probe", args.samples), ("target", args.targets)):
            records = banks[bank]
            if len(records) != count or len({r["seed"] for r in records}) != count:
                raise ValueError(f"Incomplete {bank} bank at {row_root}")
            correct = sum(r["correct"] for r in records)
            summaries[bank] = {"records": count, "correct": correct, "accuracy": correct / count,
                "homogeneous": correct in (0, count), "truncated": sum(r["truncated"] for r in records),
                "mean_response_tokens": sum(r["response_tokens"] for r in records) / count,
                "records_sha256": fingerprint(records)}
        if set(r["seed"] for r in banks["probe"]) & set(r["seed"] for r in banks["target"]):
            raise ValueError("Probe and target seeds overlap")
        write_once(row_root / "complete.json", {"config_fingerprint": config_hash,
                   "prompt_fingerprint": fingerprint(prompt), "row_index": prompt["row_index"],
                   "shard": args.shard, "banks": summaries})
        print(json.dumps({"completed_row": prompt["row_index"], "banks": summaries}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "sample"), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--revision", help="Optional explicit revision; must match a local HF snapshot ID")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rows", type=parse_rows)
    parser.add_argument("--samples", type=int, default=1024, help="Independent probe responses per prompt")
    parser.add_argument("--targets", type=int, default=16, help="Separate target responses per prompt")
    parser.add_argument("--seed-start", type=int, default=271150000)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--dtype", "--model-dtype", dest="dtype", default="bfloat16",
                        choices=("bfloat16", "float32", "float16"))
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if any(getattr(args, key) <= 0 for key in ("samples", "targets", "max_tokens", "max_num_seqs",
                                              "chunk_size", "shards", "max_num_batched_tokens")):
        parser.error("Sample counts, chunk size, caps, and shards must be positive")
    if not 0 <= args.shard < args.shards or not 0 < args.gpu_memory_utilization < 1:
        parser.error("Require 0 <= shard < shards and 0 < GPU memory utilization < 1")
    if args.max_model_len is not None and args.max_model_len <= 0:
        parser.error("--max-model-len must be positive")
    if args.seed_start < 0 or args.seed_start + len(args.rows or []) * (args.samples + args.targets) >= 2**63:
        parser.error("Seed range must fit nonnegative signed int64")
    if args.mode == "sample" and (args.model is None or args.output is None or args.rows is None):
        parser.error("sample requires --model, --output, and a frozen --rows list")
    import pyarrow.parquet as pq
    path, selection = resolve_dataset(args.dataset)
    parquet = pq.ParquetFile(path)
    identity = dataset_identity(path, parquet)
    if args.mode == "inspect":
        inspect(args, parquet, identity, selection)
    else:
        sample(args, parquet, identity, selection)


if __name__ == "__main__":
    main()
