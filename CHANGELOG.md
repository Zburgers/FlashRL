# Changelog

All notable changes are documented here.

## 2.0.0 - 2026-07-28

### Added

- installable `flashrl` package and canonical CLI;
- versioned simulator, observation, action, reward, result, manifest, and
  checkpoint contracts;
- deterministic state, simulator-pixel vision, and hybrid observations;
- independently selectable Double, Dueling, PER, and N-step DQN components;
- atomic best/last checkpoints, deterministic model selection, and continuation;
- exact-frame-budget, multi-seed experiment matrices with safe resume;
- traceable held-out evaluation and run-first aggregation;
- seeded bootstrap analysis, CSV publication data, and SVG figures;
- local live policy laboratory with Q-values and deterministic controls;
- portable instrument-panel GIF recording;
- Python 3.10 through 3.13 CI and installed-wheel qualification;
- MIT license, contribution guide, and structured GitHub templates.

### Changed

- V2 is explicitly simulator-only; misleading browser-labelled paths are gone;
- TD targets bootstrap through time limits but not crashes;
- state observations include bird altitude and use feature-specific bounds;
- state-only simulation lazily skips pixel rendering;
- the recommended agent is evidence-selected Dueling Double DQN with three-step
  returns and uniform replay.

### Removed

- obsolete V1 scripts, duplicate entry points, PPO stub, ChromeDriver, browser
  vendor assets, committed checkpoints, logs, caches, and schema-v1 results.

### Research result

- five independent 120,000-frame policies score 301.90 mean over 500 held-out
  episodes, versus 126.18 random and 284.17 rule.

Historical V1 context remains in [docs/history/v1.md](docs/history/v1.md).

