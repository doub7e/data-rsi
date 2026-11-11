# 1K reproduction

- The user explicitly authorized 4×H100 if memory fits, otherwise 4×H200. This overrides the workspace's generic H200 preference for this experiment.
- Keep 32K context, 7 epochs and global batch 96. Do not reduce context to fit H100.
- Queue H100 when occupied; only a measured memory failure triggers H200 fallback.
- Use the official OpenThoughts-Agent-SFT-1K dataset at the pinned revision, not an arbitrary 10K subset. All results stay in this node and are registered in the experiment board.
