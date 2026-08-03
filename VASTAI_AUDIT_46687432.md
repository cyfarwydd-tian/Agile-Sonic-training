# Vast.ai training-image audit — instance 46687432

This is the chronological audit record for the immutable Agile SONIC training
image on Vast.ai. The goal is to identify Docker/environment blockers for real
SONIC training, not merely to obtain a successful process exit code.

## Scope

- Vast instance: `46687432`
- Machine / host: `105856` / `289275`
- Location: United Kingdom
- GPU allocation: 1× RTX PRO 6000 WS, 97,887 MB reported VRAM
- CPU allocation: 128 effective vCPUs
- Disk allocation: 400 GB
- Image:
  `ghcr.io/cyfarwydd-tian/agile-sonic-training@sha256:ee1d166e25c897ce50076efa2b99f0e79597a56ff450113aa560d11e2f5a95ed`
- Vast launch mode: direct SSH
- Test limitation: this allocation can validate one-GPU training paths but
  cannot validate DDP, NCCL collectives or inter-GPU P2P.

## Acceptance gates

1. Host driver exposes the GPU correctly to the container.
2. CUDA 12.8 user space loads against the host driver without symbol or PTX
   compatibility errors.
3. PyTorch sees the expected Blackwell device and can execute real kernels.
4. The Vast SSH launch-mode replacement of the image entrypoint is understood
   and either repaired or documented.
5. Isaac Sim and Isaac Lab start headlessly and create a real H20 environment.
6. The checked-in paired robot/SOMA fixture loads through SONIC's motion code.
7. Real rollout, PPO, auxiliary losses, backward, optimizer and checkpoint
   paths execute.
8. Checkpoints prove finite, non-zero changes in every required model group.
9. Resource, mount, cache, permissions and restart behavior are recorded.

## Chronological log

### 2026-08-03 — Instance creation and image pull

Status: **complete**

- Vast authentication succeeded.
- The requested immutable digest is attached to the instance.
- Vast reported status `loading`; the image was still downloading and
  verifying layers.
- After the base layers completed, Vast began building its SSH wrapper layer
  inside the instance image (`apt`/`dbus` trigger output was visible). This is
  expected for Vast SSH launch mode, but means the runtime is not byte-for-byte
  identical to the published digest: Vast replaces the original entrypoint and
  injects SSH setup. The published filesystem layers remain the base, while
  startup behavior must be audited separately.
- Marketplace metadata reports host driver `595.58.03` and maximum supported
  CUDA `13.2`.
- The image uses CUDA `12.8.1`; subsequent runtime tests verified `nvidia-smi`,
  `libcuda`, PyTorch, cuDNN and a minimal Isaac/PhysX simulation.
- Current all-in rate with 400 GB disk: approximately `$1.385185/hour`, before
  traffic charges.

### 2026-08-03 — First SSH startup attempt

Status: **failed before container login**

- The instance reached `running` and received a direct endpoint at public port
  `15869` plus a Vast proxy fallback.
- The account has one registered SSH key, so missing client authorization was
  ruled out.
- Vast startup logs show `sshd: no hostkeys available -- exiting` followed by
  `connect_to localhost port 22: failed`.
- Root cause: the image deliberately generates SSH host keys in its own
  entrypoint only when its key-managed sshd is enabled. Vast SSH mode replaces
  that entrypoint, then tries to start the already-installed OpenSSH daemon
  without first running `ssh-keygen -A`.
- This failure occurs before CUDA or the NVIDIA driver is exercised.

Planned temporary remediation for this paid test instance: install a minimal
Vast on-start hook that generates missing host keys and restarts sshd. Permanent
image remediation is discussed in F-002 below.

### 2026-08-03 — Non-root SSH recovery and untouched runtime baseline

Status: **connected; baseline captured**

- Generating host keys alone allowed SSH key exchange, but authentication then
  failed because Vast installs its key for `root` while the image enforces
  `PermitRootLogin no` and `AllowUsers fangzhengtian`.
- The on-start hook was refined to copy Vast's injected public authorization to
  `/home/fangzhengtian/.ssh/authorized_keys` with strict ownership/mode. Root
  login remained disabled.
- Login then succeeded as UID/GID 1000 (`fangzhengtian`) with the `video` group.
- `/run/agile-sonic/runtime.env` was missing, confirming the native entrypoint
  did not execute. However Vast plus `/etc/profile.d/agile-sonic.sh` left the
  SSH session on Python 3.11.15 in `/opt/miniforge3/envs/agile-sonic`.
- `/workspace/sonic-training`, `/datasets`, `/runs` and `/cache` exist and are
  writable by the non-root user.
- `/dev/shm` is 125 GB, which is adequate for this single-GPU audit.
- Shell limits are weaker than the normal server launcher: `nofile=1024`,
  `memlock=8192 KiB`, stack `8192 KiB`. These are recorded in F-003.

### 2026-08-03 — Driver, CUDA, PyTorch and cuDNN execution

Status: **passed**

- Actual GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition.
- Actual driver/kernel module: `595.58.03` (NVIDIA open kernel module).
- GPU compute capability: 12.0; reported VRAM: 97,887 MiB.
- Image compiler/runtime: CUDA 12.8 (`nvcc 12.8.93`).
- PyTorch: `2.7.0+cu128`; its compiled architecture list contains `sm_120` and
  `compute_120`.
- cuDNN: 9.7.1; NCCL: 2.26.2.
- `libcuda.so.1` resolves to the injected 595.58.03 host library and a usable
  `libcuda.so` linker symlink is present.
- Real 4096×4096 FP32 and BF16 CUDA matmuls completed with finite results.
- A BF16 cuDNN convolution completed forward and backward with finite loss and
  gradients.

Driver conclusion: **the feared 595/12.8 incompatibility is not present at the
PyTorch/cuDNN layer**. Isaac and extension compilation still remain as stronger
compatibility gates.

### 2026-08-03 — Native CUDA extension compile and execution

Status: **passed**

- PyTorch's C++ extension loader invoked the image compiler and Ninja from the
  active `agile-sonic` environment.
- `nvcc 12.8.93` compiled a custom CUDA kernel explicitly for
  `compute_120`/`sm_120`; the host linker produced a loadable Python extension.
- The extension executed on the RTX PRO 6000 over 1,048,576 elements and
  returned the exact expected result after CUDA synchronization.
- The persistent extension cache at `/cache/torch_extensions` is writable by
  the non-root runtime user.

Driver conclusion after this stronger gate: **driver 595.58.03, CUDA 12.8,
PyTorch 2.7 and Blackwell `sm_120` are mutually compatible on this host**.
Driver replacement is not indicated by current evidence.

### 2026-08-03 — Built-in preflight before source deployment

Status: **expected failure, useful partial results**

- `python -m pip check` passed with no broken requirements.
- Atomic writes under `/runs/nvidia-omniverse` passed.
- GPU inventory and 125 GB shared memory were visible.
- Normal preflight failed because `/workspace/sonic-training` contains only
  Vast bootstrap files, not the private SONIC checkout.
- Strict preflight additionally treats the low memlock/nofile limits as fatal.
- A minimal source bundle was calculated at approximately 239 MB by excluding
  Git history, full datasets, logs, checkpoints, artifacts, virtualenvs and
  unrelated G1 robot USD assets.
- At this stage upload of that private bundle was paused pending explicit
  confirmation. The user later authorized it; see the authorized deployment
  and smoke sections below.

### 2026-08-03 — Minimal Isaac Sim CUDA/PhysX execution

Status: **GPU simulation passed; application startup/teardown failed clean-room gate**

- Isaac Sim 5.1 started headlessly on `cuda:0` and selected the RTX PRO 6000
  through Vulkan using driver 595.58.03.
- A CUDA `SimulationContext` completed ten physics steps and printed the
  expected final simulation time (`0.01`). This is stronger evidence than an
  import check: the host driver works through the Isaac/Vulkan/PhysX path.
- Isaac Lab task-extension startup raised `ModuleNotFoundError: h5py`. This
  dependency gap is not detected by `pip check`, because the editable/source
  Isaac Lab installation does not declare it to the active environment.
- Kit could not create/update its derived-data, shader, user-config and local
  data stores under the read-only Miniforge package tree. The image's native
  entrypoint was bypassed, and setting only `XDG_CACHE_HOME=/cache/xdg` was not
  sufficient to redirect all Kit-local stores.
- `app.close()` did not return within more than one minute after the physics
  steps, so the SSH test process was interrupted. The driver gate therefore
  passes, but this is not a clean Isaac application pass.
- A second run used NVIDIA Kit's documented
  `--portable-root /cache/isaac-portable` option. All prior package-local cache,
  data, shader and user-config write errors disappeared, and Kit created its
  stores below the writable `/cache` tree.
- For isolation, `h5py==3.13.0` was installed with `--no-deps` under
  `/cache/audit-deps` only (the immutable Conda environment was not changed).
  With this temporary dependency overlay, Isaac Lab started without extension
  errors and the physics steps still passed. Application close nevertheless
  timed out after 75 seconds, confirming that F-006 is not caused solely by
  missing `h5py`.

### 2026-08-03 — Local image remediation staged

Status: **implemented locally; rebuild and runtime verification pending**

- Added pinned `h5py==3.13.0` to the training environment (it previously
  existed only in optional data/tools requirements).
- Added `h5py` to both the build-time import gate and runtime preflight.
- Added `/cache/isaac-portable` to image and entrypoint writable directories.
- Baked non-root ComputeCache, Omniverse data and log links into the image so
  they still target `/cache` and `/runs` when Vast bypasses the entrypoint.
- Defaulted `SONIC_EXTRA_KIT_ARGS` to
  `--portable-root /cache/isaac-portable`; the current private SONIC trainer
  forwards this environment variable to Isaac Lab's `AppLauncher`.
- Made the checked-in H20 smoke runner set the same default explicitly.
- Shell syntax, Python compilation, whitespace checks and Docker BuildKit's
  static `--check` pass locally with no warnings. These edits do not alter the
  immutable image tested on instance `46687432`; a new image build is required
  after the audit.

### 2026-08-03 — Authorized source deployment and preflight

Status: **passed after removing generated metadata from the runtime source tree**

- The user explicitly authorized transfer of the filtered private SONIC/H20
  bundle to this third-party test instance.
- Rsync transferred 676 regular files (786 paths total), 238,949,472 bytes
  before compression. Git history, datasets, checkpoints, logs, virtualenvs,
  unrelated robot assets and other excluded trees were not transferred.
- Both paired fixture files matched their checked-in SHA-256 digests, and the
  H20 experiment configuration, training entrypoint and robot assets exist.
- The first preflight exposed a generated `gear_sonic.egg-info` directory from
  the local checkout. Its metadata made `pip check` enforce the project's
  incompatible `numpy==1.26.4` declaration against Isaac's `numpy==1.26.0`.
  The generated directory was moved (not deleted) to
  `/cache/audit-quarantine/gear_sonic.egg-info.uploaded`.
- After quarantine, preflight passed: source, datasets, atomic run writes,
  Python imports, CUDA, NCCL availability and the one visible GPU were valid.
  Memlock remained the already-recorded 8 MiB warning.

### 2026-08-03 — Real SONIC H20 training smoke

Status: **passed**

- Run directory:
  `/runs/sonic-h20-smoke-vast-46687432-attempt1/training`.
- Isaac created 24 real H20 environments on `cuda:0`, loaded
  `gear_sonic/data/assets/robot_description/mjcf/h20.xml`, and loaded the paired
  robot/SOMA motion fixture.
- Two PPO iterations completed with 8 rollout steps per environment: checkpoint
  totals progressed from 192 to 384 simulated transitions.
- Checkpoints `model_step_000001.pt` and `model_step_000002.pt` were created
  (about 111.5 MB each), along with `last.pt`, resolved Hydra configuration and
  logs.
- The verifier proved finite, non-zero updates in every required model group:
  H20 encoder (8 tensors), teleop encoder (10), SOMA encoder (10), H20 dynamics
  decoder (8), H20 kinematics decoder (8), and critic (8). The final optimizer
  contained 53 state slots.
- Maximum observed absolute parameter deltas were approximately
  `1.12e-4`; the process exited with status 0. The post-run GPU returned to
  1 MiB usage, 0% utilization and P8, so no orphan training process remained.
- A filtered error scan found no runtime errors; matches were only normal metric
  names such as `error_anchor_pos`.
- First training startup downloaded approximately 190 MB of missing Kit
  extensions into the user data store. This is recorded in F-008.

### 2026-08-03 — Stop/start persistence test

Status: **passed**

- The instance was stopped until Vast reported `actual_status=exited`, then
  started again. The control-plane status briefly lagged (`intended_status` was
  `running` while `actual_status` remained `exited`), but the on-start hook
  restored SSH and the proxy became reachable.
- Non-root SSH recovered without weakening the image's root-login policy.
- Private source, paired fixture, temporary `h5py` overlay, Kit extension cache,
  run logs and both checkpoints survived the stop/start cycle.
- Checkpoint SHA-256 values after restart were
  `a0de07e2b4c296731ce1ef187eff301aef37f1836e5b2647ab692807aa519848`
  (step 1) and
  `888296686679b6f048e01493be636830b034d085274bde23e258378c21b909a6`
  (step 2).
- The restarted container still exposed the RTX PRO 6000 and driver 595.58.03.
- After all remote checks completed, the instance was stopped. Vast confirmed
  `actual_status=exited`, `intended_status=stopped` and zero active GPU hourly
  charge. It was later permanently destroyed with its 400 GB disk before the
  clean-rebuild round; no source, cache, logs or checkpoints from this round
  remain on Vast.ai.

## Findings and remediation

### F-001 — Vast SSH mode replaces the image entrypoint

Status: **confirmed; open**

The image normally starts through `/usr/local/sbin/entrypoint.sh`, which selects
the Conda environment, prepares writable caches, writes runtime state, handles
credentials and drops from root to `fangzhengtian`. Vast SSH mode replaces that
entrypoint with its own injected startup layer. This may leave SSH sessions as
root with the wrong Python/Conda environment and without initialized cache/run
paths. On this instance the Conda environment and workspace mounts remained
usable through profile configuration, but runtime state, Kit cache setup and
production ulimits did not.

### F-002 — Vast-injected sshd cannot start because host keys are absent

Status: **confirmed; temporary instance fix active**

Evidence from the platform startup log:

```text
sshd: no hostkeys available -- exiting.
connect_to localhost port 22: failed.
```

The normal image entrypoint calls `ssh-keygen -A`, but only along its own sshd
path. Vast's replacement entrypoint sees OpenSSH installed and attempts to start
it directly. Candidate permanent fixes are either to bake host keys (not
recommended because every container would share them), or provide a documented
Vast on-start hook that generates unique keys at instance creation. A future
Vast-specific template is the safer option.

Temporary instance workaround: generate host keys in on-start, copy Vast's
injected authorization to the existing non-root user, then start sshd. This
works without permitting root login.

### F-003 — Vast SSH sessions do not inherit production training ulimits

Status: **confirmed; open**

Observed soft values are `nofile=1024`, `memlock=8 MiB` and stack `8 MiB`. The
corresponding hard values are `1,048,576`, `8 MiB` and unlimited. The non-root
shell successfully raised its nofile soft limit to `1,048,576` and stack to
`64 MiB`; these can be applied by a shell/profile wrapper before training.
Memlock remained `8 MiB` because its hard limit is also `8 MiB`. The normal
Docker launcher requests unlimited memlock, so only the Vast template/container
runtime can correct that limit. A small one-GPU smoke may pass despite it, but
this instance is not a valid NCCL/pinned-memory production baseline until the
hard memlock limit is fixed.

### F-004 — `h5py` is absent from the training environment

Status: **confirmed; local fix staged**

Isaac's `isaaclab_tasks` extension imports `h5py` during startup, but the module
is not installed. This prevents a clean task-extension initialization even
though core Isaac simulation can run. Add a version-pinned `h5py` package to
the authoritative training lock/spec, rebuild the image, and add an explicit
`import h5py` (or actual dataset read) to CI/preflight; `pip check` alone is not
an adequate completeness test.

### F-005 — Isaac/Kit runtime stores are not fully writable

Status: **confirmed; local fix staged**

The non-root application attempted to write derived data, shader cache,
user configuration and local data below the installed Isaac Sim package under
`/opt/miniforge3`. Those writes failed. The image must explicitly redirect all
Kit stores to `/cache` or `/runs`, or prepare writable links/directories during
startup. The correction must be validated from a fresh Vast launch where the
native entrypoint is bypassed; merely repairing an already-running container
would hide the launch-mode defect.

Validated remediation: passing `--portable-root /cache/isaac-portable` to Kit
redirects cache, data, logs, documents, derived data and shader cache to a
writable persistent tree and removes all observed write errors. This option
should be injected by the training launcher (or its equivalent Kit settings
should be pinned) so it works in both ordinary Docker and Vast SSH mode.

The Vast SSH user has no passwordless `sudo`, so this cannot reliably be
repaired after login by invoking the root-only image entrypoint. The fix belongs
in image startup/template configuration rather than an operator shell command.

### F-006 — Isaac application teardown hangs after minimal simulation

Status: **not reproduced by SONIC; closed as a container blocker**

Ten physics steps completed, but `app.close()` did not return within more than
one minute. A controlled retest with writable portable Kit state and a temporary
`h5py==3.13.0` overlay still timed out after 75 seconds, with no extension or
cache errors in the filtered log. This is now independent of F-004/F-005 and
must be reproduced with the real SONIC process lifecycle. It is not evidence of
a driver incompatibility because initialization and repeated CUDA physics steps
completed successfully.

The real SONIC training process subsequently completed two PPO iterations,
wrote and verified checkpoints, and exited with status 0. The standalone
minimal probe's shutdown behavior therefore does not block the supported
training lifecycle; keep it only as a test-harness cleanup issue.

### F-007 — SONIC package metadata conflicts with Isaac's NumPy pin

Status: **confirmed; open upstream/source issue**

`gear_sonic/pyproject.toml` declares `numpy==1.26.4`, while Isaac Sim 5.1 in the
training environment requires `numpy==1.26.0`. The image intentionally exposes
SONIC as source instead of installing its package metadata, so runtime imports
and real training pass. However, copying a generated `gear_sonic.egg-info` into
the workspace makes `pip check` fail. Deployment sync rules should exclude
`*.egg-info`, and SONIC's package requirements should ultimately use an
Isaac-compatible constraint or a training extra with the correct pin.

### F-008 — First Isaac training launch still downloads Kit extensions

Status: **confirmed; persistence fix staged, offline behavior still open**

Despite the installed Isaac extension-cache distributions, the first real H20
launch synchronized the Kit registries and downloaded at least the pip-archive
and URDF-importer extensions. The resulting user extension store was about
190 MB. Online startup succeeded, so this is not a Vast blocker; it is a
reproducibility/offline-server risk. Either document first-launch egress and
persist this store under `/cache`, or precache the exact training experience in
the image during CI and verify startup with network disabled.

The next-image Dockerfile now bakes the Omniverse user-data link to
`/cache/ov-data`, covering persistence even when Vast skips the entrypoint. It
does not remove the first-start network dependency.
