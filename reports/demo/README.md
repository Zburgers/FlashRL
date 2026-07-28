# Learned-policy demo evidence

The portable recording uses the median final training run by held-out mean:

- training run: `v2-benchmark-recommended-seed89`
- training seed: `89`
- held-out replay seed: `200058`
- held-out episode score: `286.2517`
- checkpoint role: `best`
- checkpoint SHA-256:
  `758f47a4ce588bc504d02876940d7ca0ca1c5f81d42c42fbaca17b71c12b6600`
- recording: `median.gif`, 960 by 540, 93 deterministic frames

Reproduce the portable recording after obtaining the V2 release checkpoint:

```bash
flashrl demo \
  --policy dqn \
  --checkpoint best.pt \
  --seed 200058 \
  --record reports/demo/median.gif
```

Launch the interactive live policy laboratory instead:

```bash
flashrl demo \
  --policy dqn \
  --checkpoint best.pt \
  --seed 200058
```

The browser interface renders the simulator live and exposes score, action,
reward, game speed, Q-values, checkpoint identity, pause, playback speed, and
deterministic seed reset. The GIF is a server-side instrument-panel rendering
of the same checkpoint and simulator contract; it is portable evidence, not a
substitute for the interactive console.
