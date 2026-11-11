# Data Self-Improvement experiments

This repository tracks the experiment code, configurations, documentation, run manifests, and lightweight results for the Data Self-Improvement workspace. Each immediate experiment directory is an experiment node.

**Model checkpoints are not managed by Git or Git LFS.** Model weights, optimizer states, full gradient tensors, raw datasets and rollout traces, caches, downloaded dependencies, sandbox/build directories, logs, and credentials stay on shared storage. This repository is not a backup of those artifacts.

## Experiment index

| Directory | Experiment |
| --- | --- |
| `001-facet-reproduction` | FACET reproduction preparation |
| `002-openthoughts-qwen3-8b` | OpenThoughts-Agent Qwen3-8B 10K reproduction |
| `003-openthoughts-qwen3-8b-1k` | Qwen3-8B 1K reproduction and evaluation |
| `004-runai-native-eval` | Native RunAI evaluation sandboxes |
| `005-datasense-bfcl` | DataSense/BFCL baseline |
| `006-bfcl-probe-gradient` | BFCL probe-gradient selection experiments |
| `007-openthoughts-probe-gradient` | OpenThoughts-Agent probe-gradient diagnostics |
| `008-openthoughts-easier-probes` | Easier-task probe diagnostics |
| `009-ota-environment-resampling` | Initial environment-resampling experiment |
| `010-ota-full70-r4` | Full70 x4 probe and uncapped resampling |
| `011-ota-cap3-resampling` | Resampling with a hard cap of3 per environment |
| `qwen_math_*` | Separate math gradient/reliability and training experiments; see their protocols |

## Latest completed OTA comparison

All conditions use70 benchmark tasks with4 attempts per task. Missing terminal verifier scores count as0. All70 tasks supplied probe feedback, so these are not independent held-out results. One training seed was used.

| Condition | Mean score | Full passes | Full-pass rate |
| --- | ---: | ---: | ---: |
| First epoch | 0.1021 | 23/280 | 8.21% |
| Uniform continuation | 0.1468 | 36/280 | 12.86% |
| Uncapped Guided continuation | 0.0931 | 18/280 | 6.43% |
| Cap3 Guided continuation | 0.1374 | 31/280 | 11.07% |

Cap3 improves the point estimate relative to uncapped Guided, but the paired task-bootstrap95% interval for the score difference includes0. See the reports in `010-ota-full70-r4/runs/comparison/` and `011-ota-cap3-resampling/runs/comparison/` for the complete comparisons and limitations.

## Running and recording experiments

- Read `AGENTS.md`, the relevant experiment README, and its frozen protocol before running anything.
- GPU and substantial CPU work run on RunAI. Host-to-container paths map `/mnt/cvlab/scratch` to `/scratch`.
- Current scripts retain environment-specific shared-storage paths, PVC/project names, and immutable image references. Adapt them for another cluster; cloning alone does not restore the external datasets or models.
- Run manifests and lightweight results are tracked. Large artifact paths referenced by those manifests may be unavailable in a fresh clone until the original shared storage is mounted or the data is regenerated.
- The experiment-board application lives in the sibling `../experiment-board` directory and is intentionally not part of this repository. Experiment results remain here, not in the board application.
- The nested `005-datasense-bfcl/upstream` reference checkout, downloaded runtime dependencies, and generated `qwen_math_*/controller*/source` copies from other projects are excluded. Their provenance/configuration records remain tracked; restore the pinned external source/runtime before rerunning these experiments.

## Git hygiene

After cloning, enable the checked-in commit guard:

```bash
git config core.hooksPath .githooks
```

The guard checks staged content for model/optimizer/gradient artifacts, files over5MiB, common credential formats, and unintended nested Git repositories. `.gitignore` keeps large artifact trees out of normal `git add` operations. Do not force-add checkpoints or credentials. Git LFS is not used for model artifacts.

The guard is also runnable directly with `python3 tools/check_staged.py`. It inspects the staged snapshot, not unstaged working files.
