# Real SONIC H20 training smoke test

This fixture is a 120-frame (4-second), paired robot/SOMA slice of
`neutral_walk_180_R_002__A075` from the current H20 test data. It is committed
as checksummed, gzip/base64 text so the public container project has a small,
self-contained motion input without publishing the complete training dataset.

Run it on a one-GPU machine:

```bash
scripts/sonic-training-smoke.sh \
  --source /srv/agile-sonic/source \
  --runs /srv/agile-sonic/runs \
  --cache /srv/agile-sonic/cache \
  --gpu 0
```

This is not an import test. It creates 24 Isaac simulation environments, loads
the real retargeted H20 and paired SOMA motion, collects two 8-step rollouts,
runs two PPO optimizer updates with the configured auxiliary losses, and saves
checkpoints at both updates. The verifier requires actual finite weight changes
in all of these groups:

- H20, teleop and SOMA encoders
- H20 dynamic and kinematic decoders
- value/critic network
- optimizer state

The kinematic decoder is trained through `h20_recon`; the SOMA and teleop
encoders are trained through `h20_soma_latent` and `h20_teleop_latent`.
Consequently their verified changes exercise the auxiliary trainer, while the
dynamic decoder and critic changes exercise rollout collection and PPO.

The output is written under
`<runs>/sonic-h20-smoke-<UTC timestamp>/training`. A passing test ends with:

```text
SONIC H20 REAL TRAINING SMOKE: PASS
```

This test validates one GPU and all model/training paths. Run
`scripts/nccl-smoke.sh` separately for NCCL, followed by the production-scale
multi-GPU launch test.
