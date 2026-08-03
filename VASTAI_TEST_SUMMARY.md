# Vast.ai three-round training validation summary

Three isolated Vast.ai deployments were used to validate the public container
against the private SONIC/H20 training source. Each round used an RTX PRO 6000
Blackwell host with driver `595.58.03`, a fresh container deployment and an
explicitly authorized, filtered private source/asset bundle. H20 is the robot
model, not the GPU model.

## Test rounds

| Round | Test conditions | Result and purpose |
| --- | --- | --- |
| [1 — discovery, `46687432`](VASTAI_AUDIT_46687432.md) | 1 GPU, 400 GB disk, original image digest `sha256:ee1d166e25c897ce50076efa2b99f0e79597a56ff450113aa560d11e2f5a95ed` | CUDA 12.8, PyTorch, cuDNN, an `sm_120` extension, Isaac/PhysX and a real two-update H20 PPO smoke passed. The audit exposed missing `h5py`, unwritable Kit state, Vast entrypoint replacement, generated package metadata in source uploads and weak runtime limits. |
| [2 — clean rebuild, `46693892`](VASTAI_CLEAN_REBUILD_46693892.md) | 1 GPU, new 400 GB disk, no reused package overlay, cache, source or checkpoint | Verified that `h5py==3.13.0`, portable Kit paths and persistent non-root cache links were baked into the image. A fresh 24-environment, two-update H20 PPO run and stop/start persistence test passed without server-side dependency installation. |
| [3 — multi-GPU, `46697258`](VASTAI_2GPU_AUDIT_46697258.md) | 2 GPUs over PCIe P2P, no NVLink, 400 GB disk, clean final-image recycle | Found an NCCL 2.26.2 Blackwell P2P/SHM failure, validated NCCL 2.26.5, fixed automatic non-root activation, then passed a 1 GiB/50-iteration collective, two-rank BF16 H20 PPO, all-module checkpoint verification and resume. |

All three paid test instances and their attached disks have now been destroyed.
No uploaded private source, cache, log or checkpoint remains on Vast.ai.

## Bugs fixed

- Added and pinned `h5py==3.13.0` in the immutable training environment.
- Redirected Isaac/Kit cache, data and logs to writable persistent paths, even
  when Vast direct-SSH mode bypasses the normal container entrypoint.
- Excluded generated `*.egg-info`/`*.dist-info` metadata from private source
  transfer to avoid a false NumPy dependency conflict.
- Added an NCCL 2.26.5 compatibility runtime for RTX PRO 6000 Blackwell P2P and
  SHM collectives while retaining PyTorch 2.7.0+cu128.
- Fixed the NCCL helper directory permissions and added a build gate so the
  non-root user `fangzhengtian` automatically loads the validated library.

Vast's replacement of the image entrypoint is platform behavior, not an image
dependency bug. The documented Vast on-start hook restores key-only non-root
SSH. Production launch must still use `scripts/run.sh` or `compose.yaml` to set
host IPC, `memlock=-1`, file-descriptor and stack limits.

## Final verified artifact

- Build commit: [`4655f45`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/commit/4655f4545db5b249349bed43a90cb7d7ba8f88ee)
- GitHub Actions run: [`30813887535`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/actions/runs/30813887535)
- Provenance attestation: [`38569052`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/attestations/38569052)
- Convenience tag: `ghcr.io/cyfarwydd-tian/agile-sonic-training:sha-4655f45-training`
- Immutable image:
  `ghcr.io/cyfarwydd-tian/agile-sonic-training@sha256:e0e0a1b7f70983ce76cd65e3b0493f641fef445ea512aefcdf64ee6123076800`

Use the digest for partner/server deployment. Driver `595.58.03` is the tested
combination, not an exact requirement; any production driver must support CUDA
12.8 and pass the same preflight, NCCL and real-training gates.

## Detailed evidence

- [Round 1 discovery audit](VASTAI_AUDIT_46687432.md)
- [Round 2 clean-rebuild audit](VASTAI_CLEAN_REBUILD_46693892.md)
- [Round 3 two-GPU audit](VASTAI_2GPU_AUDIT_46697258.md)
- [Partner deployment handoff](PARTNER_BUILD.md)
