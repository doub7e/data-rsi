# Completed experiment handoff

The full user objective for009-ota-environment-resampling is complete. Do not relaunch jobs for a status request.

- Study completed2026-11-08T07:49:57Z (08:49Europe/Zurich).
- All3 teacher batches728/728, first SFT and both continuations1epoch/23steps each, both final evals70/70 complete.
- Latest explicit user scoring instruction is 按原verifier来. Missing original rewards remainnull; no inferred zeros. Audit preserves previous derived records and unchanged raw Harbor outputs.
- Probe443/448 original scores,4 incomplete questions excluded and8 constant questions excluded;2 independent question gradients used. Third cached question gradient retains historical imputed-score provenance and is explicitly not used.
- Candidate728 full-parameter dots completed; guided728 draws cover334 environments,ESS55.44,max repeat70. task_9288 yielded0/70 full passes; task_3480 yielded44/55 after its initial0.
- Uniform69/70 scored,7 confirmed fullpasses. Guided69/70 scored,5 confirmed fullpasses. Missing scores occur on different tasks.
- Primary paired heldout54/56 means:0.117759 vs0.131506;delta0.013747,95%CI[-0.090759,0.114815]. No reliable benefit demonstrated. Full70 joint68 delta0.002339. Probe14 mean0.119043->0.077379; both actually used probe questions declined in their single final rollout.
- All original verifier outputs preserved; raw unscored outcomes never rerolled to replace failures. Report coverage and paired complete-case comparison, not imputed means.
- All owned jobs cleaned. Controller/guard exited after completion. Shared native-auth publisher and experiment board stay untouched.
- Actual board verification passed for32 completedruns. Reports,PNG/PDF plots,teacher/student trajectories,checkpoints and audits are in this experiment.

Primary artifacts: runs/comparison/REPORT.md, result.json, comparison.png, sampling.png, training-curves.png, regeneration-audit.json; runs/study/current-status-summary.json and all-board-verifications.json. README leads with final outcome. Historical operational handoff is preserved in runs/study/revisions/completed-handoff/HANDOFF-before.md.

Limitations: one training seed,one eval rollout/task/model,two effective probe questions,heterogeneous score coverage,334 vs728 environment coverage,teacher RL dataset entirelyNL2Bash while TB-dev is broader. Do not claim the observed small heldout increase proves the method works or that the observed probe decrease proves stable success-rate decline.
