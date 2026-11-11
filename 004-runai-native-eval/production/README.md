# Full native migration

Prepared configuration: 1K and 10K, each with 4×H200 inference, 32 native CPU sandboxes, 289 tasks × 3 repeats, JSON parser. Existing results are retained separately.

`migration-plan.json` describes the reviewed scope. `scripts/prepare_all.py` builds at most two images concurrently and has an explicit timing gate. Automatic approval review initially required an explicit update to the earlier post-training-only condition. The user has now explicitly authorized immediate preparation of 288 images with at most 2 CPU-only builds; the gate is open.

The preparation and cutover services are now enabled. Deployment sequence: install/start `prepare-all.service`; install `native-eval-1k.service` and `native-eval-10k.service`; install/start `native-cutover.service`. Cutover waits until all 100 SWE task images are ready; later benchmarks wait for their own complete task image sets, stops the old evaluation producers, verifies scoped Daytona cleanup, reuses the 1K serving job, and enables native services. It does not stop training. Native workers execute long-context inference probes and a real Terminus smoke before scoring.

All static native migration tests pass. This file does not claim the full migration has already run.
