"""BFCL model alias, scoped scorer output, and versioned history handling.

The alias reuses Qwen3-8B-FC's handler configuration for Qwen3-4B. The
history_protocol module installs the strict-v2 formatter overlay by default;
legacy-v1 remains available for reproducing archived runs. Scorers are unchanged.
"""
from dataclasses import replace
import json
from pathlib import Path

def scoped_leaderboard(table,score_dir):
 # The pinned V3 protocol supplies only its selected categories. The wheel's
 # global V4 CSV writer tries to load every unrelated benchmark to fill zeros.
 # Preserve the actual per-category scorer outputs and write a scoped audit.
 path=Path(score_dir)/'scoped-leaderboard.json'
 path.write_text(json.dumps(table,indent=2,default=str)+'\n')
 print('Saved scoped leaderboard; per-category BFCL scoring is unchanged.')

def register():
 from bfcl_eval.constants import model_config as mc
 from history_protocol import install
 install()
 key='Qwen/Qwen3-4B-FC'
 if key not in mc.MODEL_CONFIG_MAPPING:
  config=replace(mc.MODEL_CONFIG_MAPPING['Qwen/Qwen3-8B-FC'],model_name='Qwen/Qwen3-4B',display_name='Qwen3-4B (FC)',url='https://huggingface.co/Qwen/Qwen3-4B')
  mc.MODEL_CONFIG_MAPPING[key]=config
  mc.local_inference_model_map[key]=config
 from bfcl_eval.eval_checker import eval_runner
 eval_runner.generate_leaderboard_csv=scoped_leaderboard
 return mc.MODEL_CONFIG_MAPPING[key]
