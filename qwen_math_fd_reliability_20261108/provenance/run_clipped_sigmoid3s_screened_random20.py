"""Train clipped-residual sigmoid(3*s) OPD on screened Math or Code data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from run_unweighted30_forks import ROOT, PARENT, CRITICAL as BASE_CRITICAL, environment


STUDENT = (
    ROOT.parent
    / ".cache/huggingface/hub/models--Qwen--Qwen3-4B"
    / "snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)
DOMAIN = {
    "math": {
        "data": ROOT / "prompt_screens/math_screened_random1600_seed27115/train.parquet",
        "teacher": (
            ROOT.parent
            / ".cache/huggingface/hub/models--Keven16--Qwen3-4B-Non-Thinking-RL-Math-Step500"
            / "snapshots/05d82d02780d4a6f8295b2909dbbd89e8a8b5aaa"
        ),
        "method": (
            "opd_math_math_n64_clipscoreloss_sigmoid3s_balanced_richardson_"
            "frombase_h1em5_fallback0_screenedrandom1600_seed27115_step20"
        ),
        "eval_tool": "tools/eval_math_checkpoint_local.py",
    },
    "code": {
        "data": ROOT / "prompt_screens/code_screened_random1600_seed27115_n64_v1/train.parquet",
        "teacher": (
            ROOT.parent
            / ".cache/huggingface/hub/models--Keven16--Qwen3-4B-Non-Thinking-RL-Code-Step300"
            / "snapshots/a3735900473592c2d33fc27706ff6fda30abda67"
        ),
        "method": (
            "opd_code_code_n64_clipscoreloss_sigmoid3s_balanced_richardson_"
            "frombase_h1em5_fallback0_screenedrandom1600_seed27115_step20"
        ),
        "eval_tool": "tools/eval_code_checkpoint_local.py",
    },
}
COMMON_CRITICAL = BASE_CRITICAL + (
    "verl/verl/utils/crossfit_fallback.py",
    "verl/verl/utils/fd_comparison.py",
    "tools/test_bf16_cachefix_training_path.py",
    "tools/test_balanced_crossfit_modes.py",
    "tools/run_clipped_sigmoid3s_screened_random20.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def configuration(domain: str) -> dict[str, str]:
    item = DOMAIN[domain]
    baseline = json.loads(PARENT.read_text())
    config = environment("sigmoid", baseline)
    for key in ("RESUME_FROM_PATH", "RESUME_PARENT_MANIFEST", "ALLOW_IN_PLACE_RESUME"):
        config.pop(key, None)
    config.update(
        LAMBDA_VALS="1.0",
        REF_TEACHER=str(item["teacher"]),
        TRAIN_FILE=str(item["data"]),
        CURRICULUM_FILE=str(item["data"]),
        DATA_ABILITY_FILTER=domain,
        DATA_SHUFFLE="False",
        TOTAL_STEPS="20",
        TOTAL_EPOCHS="1",
        SAVE_FREQ="20",
        ROLLOUT_N="64",
        TRAIN_BATCH_SIZE="16",
        PPO_MINI_BATCH_SIZE="16",
        OUTCOME_CROSSFIT_EXPECTED_ROLLOUTS="64",
        TEACHER_SELECTION_MODE="outcome-crossfit-token-weight",
        MAX_RESPONSE_LENGTH="16384",
        ACTOR_LR="1e-5",
        OPD_TOP_K="0",
        OUTCOME_CROSSFIT_FD_MODE="richardson",
        OUTCOME_CROSSFIT_FD_RELATIVE_STEP="1e-5",
        OUTCOME_CROSSFIT_BALANCE_WORK="True",
        OUTCOME_CROSSFIT_OPTIMIZER_GEOMETRY="identity",
        OUTCOME_CROSSFIT_SCORE_AUDIT="True",
        OUTCOME_CROSSFIT_SCORE_USE_CLIPPED_RESIDUAL="True",
        OPD_RESIDUAL_CLIP_THRESHOLD="1",
        OUTCOME_CROSSFIT_MIN_WEIGHT="0",
        OUTCOME_CROSSFIT_MAX_WEIGHT="1",
        OUTCOME_CROSSFIT_FALLBACK_WEIGHT="0",
        OUTCOME_CROSSFIT_POSITIVE_WEIGHT_MODE="sigmoid",
        OUTCOME_CROSSFIT_SIGMOID_TAU="3",
        OUTCOME_CROSSFIT_SIGMOID_SHIFT="0",
        OUTCOME_CROSSFIT_RAW_SCORE_THRESHOLD="-1",
        OUTCOME_CROSSFIT_RAW_POSITIVE_SCORE_THRESHOLD="-1",
        OUTCOME_CROSSFIT_FD_LEGACY_CACHE="False",
        OUTCOME_CROSSFIT_FD_FULL_PRECISION="False",
        OUTCOME_CROSSFIT_PROBE_FULL_PRECISION="False",
        OUTCOME_CROSSFIT_FD_CACHE_WEIGHTS="False",
        BF16_CACHEFIX_VARIANT="clipped",
        METHOD_OVERRIDE=str(item["method"]),
        BF16_CACHEFIX_STUDY_NAME=f"clipped_sigmoid3s_screened_random20_v1/{domain}",
        PROMPT_ORDER=(
            f"fresh Base; first 320 prompts of frozen seed27115 screened-random {domain} pool; "
            "no difficulty sorting"
        ),
        EVAL_CHECKPOINT_STEPS="20",
        EVAL_FINAL_STEP="20",
        RUN_EVAL="0",
        TRAIN_STOP_AFTER_CHECKPOINT_STEP="0",
        SEED="27115",
    )
    return config


def validate(domain: str, config: dict[str, str]) -> dict:
    item = DOMAIN[domain]
    expected = {
        "TOTAL_STEPS": "20",
        "SAVE_FREQ": "20",
        "ROLLOUT_N": "64",
        "TRAIN_BATCH_SIZE": "16",
        "PPO_MINI_BATCH_SIZE": "16",
        "LAMBDA_VALS": "1.0",
        "TRAIN_FILE": str(item["data"]),
        "REF_TEACHER": str(item["teacher"]),
        "DATA_ABILITY_FILTER": domain,
        "DATA_SHUFFLE": "False",
        "TEACHER_SELECTION_MODE": "outcome-crossfit-token-weight",
        "MAX_RESPONSE_LENGTH": "16384",
        "ACTOR_LR": "1e-5",
        "OPD_TOP_K": "0",
        "OUTCOME_CROSSFIT_EXPECTED_ROLLOUTS": "64",
        "OUTCOME_CROSSFIT_FD_MODE": "richardson",
        "OUTCOME_CROSSFIT_FD_RELATIVE_STEP": "1e-5",
        "OUTCOME_CROSSFIT_BALANCE_WORK": "True",
        "OUTCOME_CROSSFIT_OPTIMIZER_GEOMETRY": "identity",
        "OUTCOME_CROSSFIT_SCORE_USE_CLIPPED_RESIDUAL": "True",
        "OPD_RESIDUAL_CLIP_THRESHOLD": "1",
        "OUTCOME_CROSSFIT_MIN_WEIGHT": "0",
        "OUTCOME_CROSSFIT_MAX_WEIGHT": "1",
        "OUTCOME_CROSSFIT_FALLBACK_WEIGHT": "0",
        "OUTCOME_CROSSFIT_POSITIVE_WEIGHT_MODE": "sigmoid",
        "OUTCOME_CROSSFIT_SIGMOID_TAU": "3",
        "OUTCOME_CROSSFIT_SIGMOID_SHIFT": "0",
        "OUTCOME_CROSSFIT_RAW_SCORE_THRESHOLD": "-1",
        "OUTCOME_CROSSFIT_RAW_POSITIVE_SCORE_THRESHOLD": "-1",
        "OUTCOME_CROSSFIT_FD_LEGACY_CACHE": "False",
        "OUTCOME_CROSSFIT_FD_FULL_PRECISION": "False",
        "OUTCOME_CROSSFIT_PROBE_FULL_PRECISION": "False",
        "OUTCOME_CROSSFIT_FD_CACHE_WEIGHTS": "False",
        "BF16_CACHEFIX_VARIANT": "clipped",
    }
    for key, value in expected.items():
        assert config[key] == value, (key, config[key], value)
    assert not any(key.startswith("RESUME_") for key in config)
    data = Path(item["data"])
    manifest_path = data.with_suffix(".json")
    assert data.is_file() and manifest_path.is_file()
    assert STUDENT.is_dir() and Path(item["teacher"]).is_dir()
    manifest = json.loads(manifest_path.read_text())
    assert digest(data) == manifest["sha256"]
    assert manifest["seed"] == 27115 and manifest["num_prompts"] == 1600
    assert manifest["teacher_screening"] and not manifest["difficulty_sorting"]
    assert len(manifest["prompts"]) == 1600
    assert len({entry["prompt_hash"] for entry in manifest["prompts"][:320]}) == 320
    if domain == "math":
        assert manifest["accuracy_range"] == [0.125, 0.875]
    else:
        assert manifest["screen_correct_count_interval"] == [1, 7]
    critical = COMMON_CRITICAL + (str(item["eval_tool"]),)
    for relative in critical:
        assert (ROOT / relative).is_file(), relative
    teacher_config = json.loads((Path(item["teacher"]) / "config.json").read_text())
    assert teacher_config["model_type"] == "qwen3"
    assert teacher_config["vocab_size"] == 151936
    return {
        "data_manifest": manifest,
        "critical": critical,
        "teacher_config_sha256": digest(Path(item["teacher"]) / "config.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=tuple(DOMAIN))
    args = parser.parse_args()
    domain = args.domain
    item = DOMAIN[domain]
    config = configuration(domain)
    checked = validate(domain, config)
    method = str(item["method"])
    output = ROOT / "outputs" / ("singleteacher_" + method)
    study = ROOT / "eval_outputs" / config["BF16_CACHEFIX_STUDY_NAME"]
    assert not output.exists(), output
    assert not (study / "run.json").exists(), study

    temp_root = ROOT.parent / "tmp"
    temp_root.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"tr-{domain}-", dir=temp_root))
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("OUTCOME_CROSSFIT_", "OPD_", "ADAMV_", "FD_", "RAW_DIAG_", "PROBE_SPEED_", "RESUME_")
        )
    }
    for key in ("EXPERIMENT_STORAGE_ROOT", "TEACHER_SELECTION_MODE_OVERRIDE", "ALLOW_IN_PLACE_RESUME"):
        env.pop(key, None)
    env.update(
        config,
        METHOD=method,
        PYTHON_BIN=sys.executable,
        HF_HOME=str(ROOT.parent / ".cache/huggingface"),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        WANDB_MODE="offline",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        TMPDIR=str(temporary),
        TEMP=str(temporary),
        TMP=str(temporary),
        RAY_TMPDIR=str(temp_root),
        TORCHINDUCTOR_CACHE_DIR=str(temporary / "torchinductor"),
        TRITON_CACHE_DIR=str(temporary / "triton"),
        PYTHONPATH=f"{ROOT}/verl:{ROOT}/tools",
        TEST_BALANCE_WORK="True",
        TEST_WEIGHT_MODE="sigmoid",
        TEST_EXPECTED_ROLLOUTS="64",
        TEST_SCORE_CLIP="True",
        TEST_LOSS_CLIP="1",
        TEST_OUTPUT=str(study / "smoke"),
    )
    record = {
        "state": "preflight",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "environment": config,
        "output": str(output),
        "initialization": "Qwen3-4B original Base; fresh Adam state, scheduler, RNG and data iterator",
        "teacher": str(item["teacher"]),
        "teacher_config_sha256": checked["teacher_config_sha256"],
        "data": str(item["data"]),
        "data_sha256": digest(Path(item["data"])),
        "teacher_screened": True,
        "difficulty_sorted": False,
        "available_screened_prompts": 1600,
        "consumed_prompts": 320,
        "responses_per_prompt": 64,
        "prompts_per_step": 16,
        "total_rollouts_per_step": 1024,
        "weight_function": "sigmoid(3 * s), range (0, 1)",
        "fallback_weight": 0,
        "score_residual_clipping": [-1, 1],
        "loss_residual_clipping": [-1, 1],
        "score_geometry": "identity (non-Adam-aware)",
        "fd": {"precision": "bfloat16", "mode": "Richardson", "relative_step": 1e-5},
        "eval_steps": [20],
        "saved_steps": [20],
        "temporary_storage": str(temporary),
        "source_sha256": {relative: digest(ROOT / relative) for relative in checked["critical"]},
    }

    def state(value: str) -> None:
        record["state"] = value
        write_json(study / "run.json", record)

    def run(command: list[str]) -> None:
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    state("preflight")
    try:
        for test in ("tools/test_bf16_cachefix_training_path.py", "tools/test_balanced_crossfit_modes.py"):
            run([
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                test,
            ])
        print(
            f"PREFLIGHT passed: {domain}, RL teacher, clipped score/loss residual, sigmoid(3*s)",
            flush=True,
        )
        state("training")
        run(["bash", "scripts/run_singleteacher_train20.sh"])
        assert (output / "latest_checkpointed_iteration.txt").read_text().strip() == "20"
        checkpoint = output / "global_step_20"
        assert (checkpoint / "data.pt").is_file()
        for rank in range(8):
            for kind in ("model", "optim", "extra_state"):
                path = checkpoint / "actor" / f"{kind}_world_size_8_rank_{rank}.pt"
                assert path.is_file() and path.stat().st_size > 0
        state("evaluating_checkpoint20")
        run([sys.executable, str(item["eval_tool"]), method, "--step", "20"])
        state("complete")
    except BaseException as error:
        record["error"] = repr(error)
        state("exited_before_completion")
        raise


if __name__ == "__main__":
    main()
