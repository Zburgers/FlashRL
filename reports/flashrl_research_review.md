# FlashRL Research Review

Date: 2026-06-10

## Sources Reviewed

- Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015: https://www.nature.com/articles/nature14236
- Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", AAAI/arXiv 2015/2016: https://arxiv.org/abs/1509.06461
- Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML/arXiv 2016: https://arxiv.org/abs/1511.06581
- Schaul et al., "Prioritized Experience Replay", ICLR/arXiv 2015/2016: https://arxiv.org/abs/1511.05952
- Bellemare et al., "A Distributional Perspective on Reinforcement Learning", ICML/arXiv 2017: https://arxiv.org/abs/1707.06887
- Fortunato et al., "Noisy Networks for Exploration", ICLR/arXiv 2017/2018: https://arxiv.org/abs/1706.10295
- Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning", AAAI/arXiv 2017/2018: https://arxiv.org/abs/1710.02298
- Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017: https://arxiv.org/abs/1707.06347
- Marwah et al., "Chrome Dino Run using Reinforcement Learning", arXiv 2020: https://arxiv.org/abs/2008.06799
- Towers et al., "Gymnasium: A Standard Interface for Reinforcement Learning Environments", arXiv 2024/2025: https://arxiv.org/abs/2407.17032
- Gymnasium Env API docs: https://gymnasium.farama.org/api/env/
- Gymnasium migration guide: https://gymnasium.farama.org/introduction/migration_guide/

## Research Takeaways For FlashRL

FlashRL should not try to become "Rainbow" immediately. The research literature assumes a stable environment, correct observation encoding, fixed evaluation protocol, and reliable result logging. FlashRL currently lacks those prerequisites.

The correct upgrade path is:

1. Build a reproducible Gymnasium environment and benchmark protocol.
2. Implement random and rule-based baselines.
3. Make vanilla DQN correct for the selected observation mode.
4. Add Double DQN, then Dueling DQN, then PER and N-step.
5. Combine only after ablation infrastructure exists.
6. Use PPO/recurrent PPO as comparison baselines, not as a replacement for fixing DQN.

## Paper-by-Paper Review

| Reference | Relevant idea for FlashRL | Worth implementing? | Expected benefit for Dino | Difficulty | Phase |
|---|---|---:|---|---|---|
| Mnih et al. 2015 DQN | Pixel-input DQN with frame stacking, replay, target network, Atari-style preprocessing. | Yes, but only after real pixel observations exist. | Establishes the baseline for vision mode and makes the project research-comparable. | Medium | Phase 2 |
| Van Hasselt et al. Double DQN | Separate online action selection from target-network value evaluation to reduce overestimation. | Yes. | Dino has sparse terminal penalties and similar actions; reducing overoptimistic jump/noop estimates should improve stability. | Low | Phase 3 |
| Wang et al. Dueling DQN | Separate state value and action advantage heads. | Yes. | Many Dino states have similar action values except near obstacles; dueling can learn state value more efficiently. | Low-medium | Phase 3 |
| Schaul et al. PER | Replay high-TD-error transitions more often with importance correction. | Yes, after uniform replay baseline. | Crashes and near-obstacle decisions are rare but important; PER should improve sample efficiency. | Medium | Phase 3 |
| Bellemare et al. C51 | Learn a return distribution rather than only expected Q. | Maybe. | Could improve robustness under stochastic obstacle spacing, but overkill before stable baselines. | Medium-high | Phase 3/4 |
| Fortunato et al. NoisyNet | Learned parametric exploration instead of epsilon-only exploration. | Maybe. | Useful if epsilon-greedy causes too many suicidal jumps or insufficient rare-state exploration. | Medium | Phase 3/4 |
| Hessel et al. Rainbow | Combine Double, Dueling, PER, N-step, distributional RL, NoisyNet. | Yes eventually, not first. | Strong final value-based baseline once components are separately validated. | High | Phase 3/4 |
| Schulman et al. PPO | Stable clipped policy-gradient method with multiple epochs over sampled trajectories. | Yes as comparison. | PPO may handle hybrid/state observations cleanly and provide an on-policy contrast to DQN. | Medium | Phase 3 |
| Chrome Dino RL paper | Compares DQN, Expected SARSA, and Double DQN for Dino with CNN-based setup. | Use as domain reference. | Provides Dino-specific baseline framing and confirms Double DQN is a natural next algorithm. | Low to read, medium to reproduce | Phase 2 |
| Gymnasium paper/docs | Standardized API, reproducibility tools, reset/step semantics, wrappers, seeding. | Mandatory. | Enables benchmark interoperability with SB3/CleanRL-style tooling. | Low-medium | Phase 1 |

## Algorithm Roadmap

### Random Baseline

Implement first. It catches environment bugs and gives a floor. Use fixed seeds, 100 eval episodes, no training, and log score/survival/death type.

### Rule-Based Baseline

Implement before DQN. A simple heuristic using obstacle distance, current speed, obstacle type, and dino y-state should be hard for weak DQN to beat. If DQN cannot beat this, the RL setup is not useful.

### Vanilla DQN Cleanup

Maintain separate networks:

- `StateDQN`: MLP over normalized structured features.
- `VisionDQN`: CNN over stacked grayscale frames.
- `HybridDQN`: CNN encoder plus MLP features, concatenated before Q-head.

Add frame-based epsilon schedule, proper replay warmup, target update config, deterministic seeds, gradient norm logging, loss logging, CSV/JSONL metrics, and checkpoint metadata.

### Double DQN

Change target from:

```text
max_a Q_target(s', a)
```

to:

```text
a* = argmax_a Q_online(s', a)
target = Q_target(s', a*)
```

This is a low-risk improvement and should be the first algorithm upgrade.

### Dueling DQN

Replace the single Q head with value and advantage streams:

```text
Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
```

Useful when many frames have "wait" and "jump" nearly tied.

### Prioritized Experience Replay

Use proportional prioritization first. Log beta schedule and effective sample weights. Keep uniform replay as a switch for ablations.

### N-Step Returns

Add after replay is reliable. N-step targets help propagate crash/score information backward from important events.

### NoisyNet

Use after epsilon DQN is stable. It may improve exploration without hand-tuned epsilon schedules.

### C51 or QR-DQN

Use C51 if implementation simplicity matters; use QR-DQN if you want modern distributional baselines. Do not implement until regular DQN variants are passing benchmarks.

### Rainbow DQN

Implement only after component ablations exist. Rainbow without ablations is not a benchmark; it is a black box.

### PPO / Recurrent PPO

Use PPO as a comparison baseline after Gymnasium migration. Use recurrent PPO for partially observed modes or if vision-only without frame stacking underperforms. PPO is likely easier with Stable-Baselines3 if the environment follows Gymnasium/SB3 compatibility.

## Research-Backed Design Rules

- If claiming vision, use real image observations and report preprocessing exactly.
- If using structured state, say so and use MLP baselines.
- Do not compare algorithms on different observation modes unless clearly labeled.
- Every result must include seed, commit hash, config, checkpoint path, episode count, environment mode, action set, and eval protocol.
- Report sample efficiency and wall-clock time, not only best score.
- Do ablations before combining improvements.

