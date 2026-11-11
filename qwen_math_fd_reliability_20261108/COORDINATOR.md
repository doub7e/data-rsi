# Overnight FD reliability experiment — coordinator notes

User authorized approximately 12 hours starting 2026-11-07 23:59:52 UTC; deadline 2026-11-08 11:59:52 UTC (13:00 Zurich). Continue autonomously, adapt based on evidence, and use sleep while jobs run. All CUDA work goes through RunAI; H200/H100/B300 explicitly authorized. No new training is currently planned.

Code worktree: `/mnt/cvlab/scratch/cvlab/home/shuli/OPD-experiments-inf-token-reweight`, branch `inf-token-reweight`, original HEAD `8634274`. Original `OPD-experiments` remains dirty qs-dev and untouched. New code is uncommitted. JSON/JSONL must not be pushed.

Data/board node: this directory. Board API `http://127.0.0.1:18765`. Read experiment-board-recording skill and runai-gpu-job skill. Always use native KUBECONFIG `/mnt/cvlab/scratch/cvlab/home/shuli/.codex/secrets/native-runai-kubeconfig.json` for automation. Map host prefix `/mnt/cvlab/scratch` to worker `/scratch`.

## Scope

- Best reported Math step20 config: N64, clipped score/loss, sigmoid(3s), identity geometry, balanced BF16 Richardson h_rel=1e-5, fallback0; historical Math Avg47.27. Exact trained checkpoint and frozen screened1600 data are NOT local. An optional async user question requested accessible path; continue without waiting.
- Base Qwen3-4B cache and Math teacher/dataset are available. Base initialization matches production step0 parameterization, but all stored weights are on the BF16 grid.
- Auxiliary actual trained FP32 master reconstructed from old MOPD matchedN8 step20 actor shards. It is NOT bestN64. Output `runs/auxiliary-master-merge/model`, roundtrip exact. Only0.0135% of stratified saved-tensor samples lie on BF16 grid. Tied embedding/head are duplicated in serialized keys but remain tied; no vocab expansion.
- Primary numerical comparison fixes checkpoint, target tokens, residual and unit probe direction across exactFP32/FP32FD/BF16FD. Separate BF16-vsFP32 probe gradients and independent finite-sample reference noise. Counts16/32/128/1024 refer to responses per probe, not total two-fold training rollouts.
- Bank2048 probes: A=first1024 reference, B=next1024. SmallN use nested firstN in3 disjoint128-record B blocks. N1024 uses B. Targets16 separate seeds. Keep all probe lengths/rewards/truncations and homogeneous cases. Audit first2048 tokens of first complete independent target>=32tokens, no correctness filter. Positions16uniform+8largestabsrawresidual, strata recorded. See protocol.json and per-run frozen sources/specs.

## Runtime and jobs

- Default archer image digest45d99... has torch2.5cu124, vLLM0.7.3; unsuitable for B300 generation.
- Numerics image: `registry.rcp.epfl.ch/shuli/data-rsi-sft@sha256:d15905bbc491f0f5a7c65e36524b2b83e345e7b7031153824e64667804081d1a`, torch2.8cu128, transformers4.57.6; B300 smoke succeeded.
- Generation image: `vllm/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9` (vLLM0.32 CUDA13). Requires `--run-as-user`, Python executable is python3. Shared pure-Python grader overlay and PyArrow23 cp312 overlay staged under runtime/. Never override image torch/sympy.
- Generic `scripts/run_fd_reliability_job.sh RUN_ID` executes preregistered run/spec.json, logs per pod. Supports immutable run/source snapshot, deadline, fail-state preservation. `tools/fd_reliability_freeze.py` freezes needed tools/tests/production utilities. Upcoming main audits MUST execute snapshots. Early numerical smoke had source changes during execution; treat it as plumbing validation, not the main immutable scientific result.
- Completed/cleaned: environment-preflight-b300, environment-preflight-cpu, environment-preflight-h100, source-inventory, numerical-smoke-b300. Cluster evidence archived under respective run/cluster. H200 initial preflight `opd-fd-preflight-1108a` still queued, preserve while full.
- Auxiliary merge `opd-fd-merge-fp32-1108a` completed, not yet cluster-cleaned.
- First bank `bank-base-homotopy`: row20589,2048+16,seed681150000,16K BF16. Attempts a=UIDpermission, b=missing python alias, c=missing pyarrow; all before generation. Fixes staged. `opd-fd-bank-base-p1-1108c` failed, preserve log then cleanup. Wait for CPU generation-runtime-check before retry d.
- `generation-runtime-check` / `opd-fd-gencheck-1108a`: CPU-only modern vLLM/pyarrow/grader and full tokenizer ID compatibility check, currently creating container.
- `fsdp-cache-smoke` / `opd-fd-fsdp2-1108a`:2GPU tinyQwen actualFULL_SHARD with unchanged production cache invalidation/fullprecision context, currently creating container. This is cache-contract validation, not4B score quality.
- `audit-base-homotopy` planned and source frozen, job predeclared `opd-fd-audit-base-p1-1108a`, NOT submitted. Depends on first bank complete.json. Submit1B300 (128G host), pinned numerics image. It executes9 torch/CPU tests then main fullmatrix, source snapshot includes finite guards and token strata.

## Next steps

1. Check gencheck and FSDP logs; fix real errors while preserving attempts. Resume first bank with corrected dependencies after gencheck passes.
2. Once first64 generated responses validate real outcomes/runtime, consider parallel second historical prompt39906 and new source-only numeric prompt46675 (coin pattern). Candidate inventory has8 deterministic short numeric prompts. Do not choose by score/utility effects.
3. Start full audit as soon as bank complete. Measure first probe cost/peakmemory before scaling. Main loops retain all actual16K probe lengths. PreferB300 for long FP32 probes; H10080GB mayOOM.
4. Generate separate on-policy bank for auxiliary FP32 trained master before its audit; never reuseBase bank as on-policy trained-bank silently. Same homotopy prompt offers matched numerical-center comparison.
5. Report random/extreme strata separately, remove largestscore sensitivity, exact-vsFD error/sign/weight/calibration, and N/reference noise. N1024 is noisy reference, not truth. No intervention utility or training improvement claim from these diagnostics.
6. Keep board statuses/progress/results/artifacts verified through actual API. Preserve terminal logs and clean own completed jobs. User’s unrelated ota-rs jobs MUST remain untouched.


## Latest coordinator update (authoritative status in supervisor-status.json / monitoring/status.json)

- One controlled supervisor is running, PID in supervisor-process.json (initial PID2858022). It polls monitor+dispatcher every30s until11:59:52UTC. Read supervisor-status.json to confirm exit_code0; first iteration had a benign lock collision with a simultaneous manual monitor, subsequent cycles healthy. Do not start a second supervisor. It submits only dispatch-plan.json entries, max4 allocated/reserved own GPUs. No automatic scientific retries.
- H200 old-image environment preflight was cancelled as superseded by completed runtime validations, not left queued.
- Canonical tokenizer fix is now mandatory: render_prompt_ids does tokenize=False then encode(add_special_tokens=False), rejects strings/bools/empty. Transformers5 returned BatchEncoding; attemptd was stopped and invalid bank/source archived, 0valid generations. Valid first bank attempt is opd-fd-bank-base-p1-1108e.
- Base homotopy bank COMPLETE:2048 probes,954correct(46.582%),0truncated,mean496tokens;16targets. Basezeros COMPLETE:2048,1502correct(73.34%),0truncated,mean782tokens. Basecoin bank COMPLETE shortly after; auxhomotopy still generating (~4900mean tokens,~90%correct). All4banks registered separately and use fresh independent seeds.
- audit-base-homotopy and audit-base-zeros were manually submitted at00:58UTC and are running frozen sources. Dispatcher recognizes existingjobs. audit-base-coin auto-submitted oncebankcomplete. audit-aux-homotopy waitsitsbank. Primary4Nmatrix20directioncases/run,4128response backwards/run (referenceA1024 + smallN3repeats*2precisions + B1024*2precisions).
- First homotopyN16r0 fixedFP32probe results: F32Richardsonh1e-5 relativeL2=.01635, h1e-6=.000326; BF16h1e-5 relativeL2=.6182, Pearson=.989,weightMAE=.0390. This is partialsinglecase, NOT globalconclusion. BF16probegradientN16 vsindependentA cosine~.364, FP32probe~.366.
- Initialshape/probe math passed9worker CPU/Torch tests; allmainrunningwithsource snapshots,finite guards,random/extreme selection flags. Main config still bestN64clip+sigmoid3s, Baseinitialization/auxclearlydistinguished.
- Bank monitor now preservesactual2064/2064progress,enrichesresultmetricsandexportsfirst_probe_sample.json. ActualboardAPI bank-base-homotopy verified=true; preview content was parsed and exactlymatchedactualsample(seed681150000,597tokens).
- Monitor cleanup uses RunAI CLI afterstrictguards (directkubectldelete wasRBACforbidden); latest monitor tested15/18 mockedcases; actualmonitor/cleanup nowworks. Preserve recordedearlierfailures.
- dispatch-plan now has4primaryaudits plus4boundedfollowups, inpriorityorder: full-prefix-base-homotopy (all515targettokens,eagerexactJVP,ownbackwardvalidationtolrtol.002/atol.0005), weighted-base-homotopy-n32, weighted-base-homotopy-n1024 (selectedsubsetgradientsonly), replay-base-homotopy-hopper (samecachedA1024direction,H200/H100). These waitforcapacity/dependencies, no newprobes.
- Agent /root/intervention_implementation is implementing tools/fd_reliability_fsdp_real.py: optionalactual4B2-rankFSDP numericalcase on sameN32records, recomputegR distributed, cachefixedBF16/FP32FDvsexact. Do NOT run untilreview/register/snapshot. Root intendsinsertbeforeHopperentry ifready, resource2B300aftercapacityfree. It onlyaddsanewfile,nochangesfrozenmain.
- Optionaltools alreadyready: fd_reliability_full_jvp.py, replay_direction.py, weighted_update.py, allunexecuteduntildispatcherstarts. FullJVP eageroracleisnotproductionFSDP replacement.
- Free scratch13TB; expectedreferencecaches~64GBplusauxFP32model17.6GBareacceptable.
- Read liveworkerlogs/progress, analyze random/extremestrata separately and savedcentral2 estimates forfree asfollowup. Need comparativefigures/reports/actualAPIverification+resourcecleanupbeforefinal. Do not treat highPearsonaloneasaccuracy; inspectcalibration/weights/probenoise/actualtrainedoffgridmaster.
