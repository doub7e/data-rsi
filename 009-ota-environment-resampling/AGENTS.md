# OTA environment resampling

- User explicitly authorizes the complete teacher rollout →1epoch SFT→two1epoch continuation branches→TB-dev comparison. They are asleep; choose routine details autonomously and persist with bounded sleep/monitoring.
- Use exact requested resources, pinned revisions. No replacement teacher/student/dataset without evidence.
- All stages and failures recorded in board; results remain here. Read data-rsi/AGENTS.md.
- Heavy CPU/GPU and image builds run on RunAI, not jump host. Use existing short-lived token-only kubeconfig. Never touch sibling experiment jobs.
- Paired branches start from same first-epoch checkpoint and same optimizer treatment/training recipe. Equal number of new trajectories, not necessarily equal tokens; report both.
- Probe: fixed random20% TB-dev questions,32rollouts/question at first-epoch student checkpoint. Require all32 original verifier scores; exclude incomplete questions and constant-score questions explicitly, without adaptive replacement. Preserve all raw outcomes.
- Dot product: signed full-parameter gradient dot in same trainable parameter space and SFT loss; no cosine or preconditioning. Each probe question equal weight.
- Treat the remaining80% as held-out for final comparison, also report full benchmark and probe subset. Do not call fullbenchmark independent validation.
- Keep model failures/partial outcomes; infrastructure retries preserve terminal artifacts. One requested rollout per selected environment occurrence, not success-only rejection sampling.

- Numerical contract: actual DeepSpeed ZeRO3 training uses BF16model forward. Production probe/candidate gradients MUST use runtime(precision="training_bf16") and FP32 external gradient accumulation via gradient_ops.py. Legacy FP32-parameter autocast is only a named diagnostic comparator; source-loader dtype alone is not evidence of training runtime precision.
- Gate formal attribution on runs/gradient-bf16-contract-check/complete.json. Real checkpoint comparison: loss identical, gradientcosine0.9999606. Old FP32-autocast comparison on same sample hadcosine0.78597 and is not interchangeable.

- Latest explicit user instruction: 按原verifier来. Only actual original verifier rewards are numeric scores. Missing rewards stay null; do not impute0 for resource/context/cwd failures. Probe questions require all32 original verifier scores; exclude incomplete questions explicitly without replacement. Report coverage and compare jointly scored tasks with missing-outcome bounds. Raw Harbor results and prior derived statistics must remain preserved.
