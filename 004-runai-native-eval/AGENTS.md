# Native RunAI sandbox experiment

- 2026-10-31: the user superseded the earlier keep-Daytona-running instruction and authorized migrating ALL evaluation to native RunAI. Preserve old records; stop and clean Daytona at cutover. Keep training running.
- CPU work and Harbor execution run on allocated RunAI workers, not the jump host.
- Oracle/capacity checks request no GPU. Production evaluation uses 4×H200 inference per model. On 2026-11-01 the user authorized running 1K and 10K evaluations concurrently on separate 4×H200 allocations; each keeps its own model endpoint and up to 32 CPU sandboxes. Readiness/load checks must target only that evaluation's dedicated endpoint.
- Sandbox pods have no workspace PVC and no copied user credentials. Only the trusted controller gets a short-lived access-token kubeconfig, never a refresh token.
- Build images with RunAI/Kaniko, no more than 2 concurrent builds. The full migration now covers all 289 task images, with the same maximum of 2 concurrent builds.
- Keep all trial logs, gold-patch checks, resource manifests and timings here. Oracle rewards are environment validation, not model evaluation.
