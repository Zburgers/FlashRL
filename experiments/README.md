# FlashRL experiment configurations

Experiment files use JSON syntax saved with a `.yaml` extension. JSON is a
strict, portable subset of YAML, so FlashRL can parse the files without adding
another runtime dependency.

Each configuration declares a common DQN setup, independent training seeds, a
held-out evaluation protocol, and named one-factor variants. Inspect the exact
matrix without starting compute:

```bash
flashrl experiment experiments/pilot.yaml --dry-run
```

Run it sequentially and safely continue completed or interrupted jobs:

```bash
flashrl experiment experiments/pilot.yaml --resume
```

Every expanded job receives a descriptive run ID, immutable manifest, separate
`best.pt` and `last.pt` checkpoints, training metrics, and per-episode held-out
results. `--workers N` enables bounded process-level parallelism; use one worker
for the simplest deterministic resource profile.

