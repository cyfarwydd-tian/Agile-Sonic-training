# Agile SONIC Training — Partner Build Handoff

## Project

- Repository: [cyfarwydd-tian/Agile-Sonic-training](https://github.com/cyfarwydd-tian/Agile-Sonic-training)
- Build commit: [`4655f45`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/commit/4655f4545db5b249349bed43a90cb7d7ba8f88ee)
- GitHub Actions run: [30813887535](https://github.com/cyfarwydd-tian/Agile-Sonic-training/actions/runs/30813887535)
- Workflow: [`container.yml`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/blob/4655f4545db5b249349bed43a90cb7d7ba8f88ee/.github/workflows/container.yml#L192)
- Image target: `training`

## Verified container image

Use the digest for an exact, immutable deployment:

```bash
docker pull ghcr.io/cyfarwydd-tian/agile-sonic-training@sha256:e0e0a1b7f70983ce76cd65e3b0493f641fef445ea512aefcdf64ee6123076800
```

The commit-based convenience tag points to the same build:

```bash
docker pull ghcr.io/cyfarwydd-tian/agile-sonic-training:sha-4655f45-training
```

The GHCR manifest was verified through anonymous access, so no registry login is
currently required for pulling this public image.

## Build result

This was a successful cache-assisted rebuild after the Blackwell NCCL and
non-root activation fixes.

- Build and publish job: 15 minutes 9 seconds
- Sampled peak disk consumption: 54,868,590,592 bytes
- Minimum remaining runner space: 57,332,219,904 bytes
- Largest compressed layer: 4,420,997,816 bytes
- GHCR layer-size check: passed
- Release-tag promotion: passed
- Provenance attestation: [38569052](https://github.com/cyfarwydd-tian/Agile-Sonic-training/attestations/38569052)

## Verified two-GPU runtime

This exact digest passed a clean two-GPU RTX PRO 6000 Blackwell audit: CUDA P2P,
automatic NCCL 2.26.5 activation, a 1 GiB/50-iteration collective, real
two-rank BF16 SONIC/H20 PPO updates, checkpoint verification and a same-topology
resume gate. The host has no NVLink and uses PCIe P2P. Production deployment
must still complete host-specific acceptance:

1. Install a driver compatible with CUDA 12.8, Docker Engine, and NVIDIA
   Container Toolkit. Driver `595.58.03` is runtime-verified on RTX PRO 6000
   Blackwell; validate a different production driver with the same gates.
2. Mount the private `sonic-training` checkout, datasets, run output, and cache
   directories. These are intentionally not included in the public image.
3. Start the container through this repository's `scripts/run.sh` or
   `compose.yaml`, rather than a minimal `docker run --gpus all` command.
4. Run the built-in gates inside the container:

   ```bash
   agile-sonic-preflight --strict
   agile-sonic-nccl-smoke --gpu-count <GPU_COUNT>
   ```

5. Repeat the two-update SONIC smoke and checkpoint resume on the production
   server, then run a sustained all-GPU stability/load test.

For full setup and credential-mounting instructions, see the project
[`README.md`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/blob/4655f4545db5b249349bed43a90cb7d7ba8f88ee/README.md).
