# Agile SONIC Training — Partner Build Handoff

## Project

- Repository: [cyfarwydd-tian/Agile-Sonic-training](https://github.com/cyfarwydd-tian/Agile-Sonic-training)
- Build commit: [`efd9c83`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/commit/efd9c83465a0448dec855ad4c9088ac19a80cac5)
- GitHub Actions run: [30804335746](https://github.com/cyfarwydd-tian/Agile-Sonic-training/actions/runs/30804335746)
- Workflow: [`container.yml`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/blob/efd9c83465a0448dec855ad4c9088ac19a80cac5/.github/workflows/container.yml#L192)
- Image target: `training`

## Verified container image

Use the digest for an exact, immutable deployment:

```bash
docker pull ghcr.io/cyfarwydd-tian/agile-sonic-training@sha256:6ab2d3aac46c74bb97cae40611262c16163ef1b3aae9cf503ce7e97e2f6b7b59
```

The commit-based convenience tag points to the same build:

```bash
docker pull ghcr.io/cyfarwydd-tian/agile-sonic-training:sha-efd9c83-training
```

The GHCR manifest was verified through anonymous access, so no registry login is
currently required for pulling this public image.

## Build result

This was a successful cold build with no previous registry cache available.

- Build and publish job: 22 minutes 57 seconds
- Sampled peak disk consumption: 56,978,497,536 bytes
- Minimum remaining runner space: 55,222,448,128 bytes
- Largest compressed layer: 4,102,248,768 bytes
- GHCR layer-size check: passed
- Release-tag promotion: passed
- Provenance attestation: [38546486](https://github.com/cyfarwydd-tian/Agile-Sonic-training/attestations/38546486)

## Production multi-GPU validation still required

The image has passed a clean one-GPU RTX PRO 6000 Blackwell runtime audit,
including a real two-update H20 PPO smoke. That does not verify multi-GPU NCCL,
DDP or P2P behavior. Before production training on the target multi-GPU server:

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

5. Complete a two-update SONIC smoke run, checkpoint save/restore test, and a
   sustained all-GPU test before starting a production job.

For full setup and credential-mounting instructions, see the project
[`README.md`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/blob/efd9c83465a0448dec855ad4c9088ac19a80cac5/README.md).
