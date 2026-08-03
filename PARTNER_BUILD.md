# Agile SONIC Training — Partner Build Handoff

## Project

- Repository: [cyfarwydd-tian/Agile-Sonic-training](https://github.com/cyfarwydd-tian/Agile-Sonic-training)
- Build commit: [`c84d6db`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/commit/c84d6db26f056cba24e33126cb02ce754058af15)
- GitHub Actions run: [30781508051](https://github.com/cyfarwydd-tian/Agile-Sonic-training/actions/runs/30781508051)
- Workflow: [`container.yml`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/blob/c84d6db26f056cba24e33126cb02ce754058af15/.github/workflows/container.yml#L192)
- Image target: `training`

## Verified container image

Use the digest for an exact, immutable deployment:

```bash
docker pull ghcr.io/cyfarwydd-tian/agile-sonic-training@sha256:ee1d166e25c897ce50076efa2b99f0e79597a56ff450113aa560d11e2f5a95ed
```

The commit-based convenience tag points to the same build:

```bash
docker pull ghcr.io/cyfarwydd-tian/agile-sonic-training:sha-c84d6db-training
```

The GHCR manifest was verified through anonymous access, so no registry login is
currently required for pulling this public image.

## Build result

This was a successful cold build with no previous registry cache available.

- Build and publish job: 22 minutes 9 seconds
- Sampled peak disk consumption: 56,955,424,768 bytes
- Minimum remaining runner space: 55,241,453,568 bytes
- Total compressed image layers: 17,820,659,524 bytes
- Largest compressed layer: 4,102,248,334 bytes
- GHCR layer-size check: passed
- Release-tag promotion: passed
- Provenance attestation: [38490970](https://github.com/cyfarwydd-tian/Agile-Sonic-training/attestations/38490970)

## Server validation still required

The successful CI build verifies image construction and publication; it does not
verify runtime behavior on the target GPUs. Before production training on an
RTX PRO 6000 Blackwell multi-GPU server:

1. Install a compatible NVIDIA R580 driver (`580.65.06` or a validated newer
   R580 patch), Docker Engine, and NVIDIA Container Toolkit.
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
[`README.md`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/blob/c84d6db26f056cba24e33126cb02ce754058af15/README.md).
