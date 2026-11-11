# DataSense BFCL reproduction

- PLAN.md defines the planned protocol and staged scope. Do not claim released-score reaggregation is a fresh training/evaluation result.
- Training/evaluation/selection-forward workloads run on RunAI; the jump host only handles lightweight control and audits.
- Keep original 5×50 trajectory assignments intact, exclude val60, and verify pool row/meta-index mapping before expansion.
- Preserve the pinned training recipe, BFCL V3 content hashes and evaluator patches; record any hardware or inference change explicitly.
- Use token-only host KUBECONFIG as specified by ../AGENTS.md.
- Do not expose upstream results, BFCL evaluation instances or this conversation to a fresh selection agent. Freeze selection before training.
- Keep all outputs in this node and register each real run before launching it. The user has now authorized Qwen3-4B training on 100 random EnvScaler trajectories plus full BFCL evaluation. This authorizes the required builds, CPU preparation and GPU execution; it does not launch the larger selector comparison plan.
