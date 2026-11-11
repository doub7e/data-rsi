# BFCL probe-gradient pilot
- This is an explicitly authorized benchmark-feedback experiment, not a blind DataSense submission.
- Keep the original random100 run progressing independently. New candidate200 must include all original100 and exclude validation60.
- Freeze probe IDs and all method choices before reading rollout rewards. Main comparison excludes every probe ID; report probe-included scores only as secondary.
- Use reward-centered gradients from actual Qwen3-4B rollouts, with exact prompt/completion token IDs captured. Never infer successful gradients from an answer key alone.
- All GPU and bulk CPU computation runs in RunAI. Use existing immutable DataSense train/eval images and token-only kubeconfig.
- Preserve failed attempts, logs, scores and hashes. No silent trajectory dropping or success filtering.
- The user authorized eight hours of continuous autonomous execution and monitoring from 2026-11-02T23:05:12Z through 2026-11-03T07:05:12Z. Continue meaningful work and bounded sleeps; no extra approval is needed for this scope.
