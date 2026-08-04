#!/usr/bin/env bash
set -Eeuo pipefail

mode="${1:-}"
container_files="/opt/agile-sonic/container"
conda_bin="${CONDA_DIR}/bin/conda"
training_python="${CONDA_PREFIX}/bin/python"
tools_prefix="${CONDA_DIR}/envs/agile-sonic-tools"

clone_pinned() {
    local repository="$1"
    local revision="$2"
    local destination="$3"
    local actual_revision

    if [[ -e "${destination}" ]]; then
        if [[ ! -d "${destination}/.git" ]]; then
            echo "Pinned clone destination is not a Git checkout: ${destination}" >&2
            return 2
        fi
        actual_revision="$(git -C "${destination}" rev-parse HEAD 2>/dev/null || true)"
        if [[ "${actual_revision}" != "${revision}" ]]; then
            echo "Pinned clone ${destination}: expected ${revision}, found ${actual_revision:-no HEAD}" >&2
            return 2
        fi
        return 0
    fi
    git init --quiet "${destination}"
    git -C "${destination}" remote add origin "${repository}"
    git -C "${destination}" fetch --quiet --depth 1 origin "${revision}"
    git -C "${destination}" checkout --quiet --detach FETCH_HEAD
    test "$(git -C "${destination}" rev-parse HEAD)" = "${revision}"
}

pip_install() {
    local python="$1"
    shift
    PIP_CACHE_DIR=/root/.cache/pip "${python}" -m pip install \
        --disable-pip-version-check \
        "$@"
}

install_torch() {
    local python="$1"
    local constraints="$2"
    local torch_version="$3"
    local torchvision_version="$4"
    local torchaudio_version="$5"
    local cuda_flavor="$6"
    pip_install "${python}" \
        --constraint "${constraints}" \
        --index-url https://pypi.org/simple \
        --extra-index-url "https://download.pytorch.org/whl/${cuda_flavor}" \
        "torch==${torch_version}+${cuda_flavor}" \
        "torchvision==${torchvision_version}+${cuda_flavor}" \
        "torchaudio==${torchaudio_version}+${cuda_flavor}"
}

validate_distribution() {
    local python="$1"
    local distribution="$2"
    local expected="$3"
    local actual
    actual="$("${python}" -c \
        "from importlib.metadata import version; print(version('${distribution}'))")"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "${distribution}: expected ${expected}, found ${actual}" >&2
        return 1
    fi
}

refresh_sources_manifest() {
    local manifests_dir="/opt/agile-sonic/manifests"
    local output="${manifests_dir}/sources.txt"
    local temporary
    local source_manifest
    local -a source_manifests=()

    install -d -m 0755 "${manifests_dir}"
    while IFS= read -r -d '' source_manifest; do
        source_manifests+=("${source_manifest}")
    done < <(
        find "${manifests_dir}" -maxdepth 1 -type f \
            -name 'agile-sonic*.sources.txt' -print0
    )
    temporary="$(mktemp "${manifests_dir}/sources.txt.XXXXXX")"
    {
        for source_manifest in "${source_manifests[@]}"; do
            cat "${source_manifest}"
        done
    } | LC_ALL=C sort -u > "${temporary}"
    mv -f "${temporary}" "${output}"
}

case "${mode}" in
    training-base)
        : "${PYTHON_VERSION:?PYTHON_VERSION is required}"
        : "${TORCH_VERSION:?TORCH_VERSION is required}"
        : "${TORCHVISION_VERSION:?TORCHVISION_VERSION is required}"
        : "${TORCHAUDIO_VERSION:?TORCHAUDIO_VERSION is required}"
        : "${TORCH_CUDA:?TORCH_CUDA is required}"
        : "${NCCL_RUNTIME_VERSION:?NCCL_RUNTIME_VERSION is required}"
        if [[ "${PYTHON_VERSION}" != "3.11" \
              && "${PYTHON_VERSION}" != 3.11.* ]]; then
            echo "Isaac Sim 5.1 requires PYTHON_VERSION=3.11" >&2
            exit 2
        fi

        if [[ ! -x "${training_python}" ]]; then
            training_env_file="$(mktemp)"
            sed -E \
                "s/^  - python=.*/  - python=${PYTHON_VERSION}/" \
                "${container_files}/agile-sonic.yml" \
                > "${training_env_file}"
            CONDA_PKGS_DIRS=/opt/conda-pkgs "${conda_bin}" env create \
                --prefix "${CONDA_PREFIX}" \
                --file "${training_env_file}"
            rm -f "${training_env_file}"
        fi
        validate_distribution "${training_python}" pip 25.1.1
        install_torch \
            "${training_python}" \
            "${container_files}/constraints.txt" \
            "${TORCH_VERSION}" \
            "${TORCHVISION_VERSION}" \
            "${TORCHAUDIO_VERSION}" \
            "${TORCH_CUDA}"
        validate_distribution "${training_python}" torch \
            "${TORCH_VERSION}+${TORCH_CUDA}"
        # PyTorch 2.7.0+cu128 pins NCCL 2.26.2 in distribution metadata, but
        # that runtime faults on Blackwell P2P/IPC. Keep the pinned wheel so
        # pip check remains truthful, and vendor NVIDIA's ABI-compatible patch
        # runtime for training launchers to preload explicitly.
        pip_install "${training_python}" \
            --no-deps \
            --target /opt/agile-sonic/nccl-runtime \
            "nvidia-nccl-cu12==${NCCL_RUNTIME_VERSION}"
        ;;

    training-isaac)
        : "${ISAACSIM_VERSION:?ISAACSIM_VERSION is required}"
        : "${ISAACLAB_VERSION:?ISAACLAB_VERSION is required}"
        : "${ISAACLAB_REPO:?ISAACLAB_REPO is required}"
        : "${ISAACLAB_REF:?ISAACLAB_REF is required}"

        [[ -x "${training_python}" ]] \
            || { echo "training-base must run before training-isaac" >&2; exit 2; }
        # Isaac Sim has several exact runtime requirements. Apply the same
        # constraints to all later installs so unrelated tools cannot silently
        # replace its NumPy, Numba, packaging or ASGI stack.
        pip_install "${training_python}" \
            --constraint "${container_files}/constraints.txt" \
            --extra-index-url https://pypi.nvidia.com \
            "isaacsim[all]==${ISAACSIM_VERSION}"

        clone_pinned "${ISAACLAB_REPO}" "${ISAACLAB_REF}" "${ISAACLAB_PATH}"
        test "$(cat "${ISAACLAB_PATH}/VERSION")" = "${ISAACLAB_VERSION}"
        validate_distribution "${training_python}" isaacsim \
            "${ISAACSIM_VERSION}.0"
        ;;

    training-isaac-cache)
        : "${ISAACSIM_VERSION:?ISAACSIM_VERSION is required}"
        : "${ISAACSIM_CACHE_PACKAGE:?ISAACSIM_CACHE_PACKAGE is required}"

        [[ -x "${training_python}" ]] \
            || { echo "training-base must run before training-isaac-cache" >&2; exit 2; }
        case "${ISAACSIM_CACHE_PACKAGE}" in
            isaacsim-extscache-kit|isaacsim-extscache-kit-sdk|isaacsim-extscache-physics)
                ;;
            *)
                echo "Unsupported Isaac extension-cache package: ${ISAACSIM_CACHE_PACKAGE}" >&2
                exit 2
                ;;
        esac
        # Extension-cache wheels account for most of the Isaac layer growth.
        # Install one distribution per Docker layer so each compressed registry
        # blob remains below GHCR's 10 GB limit.
        pip_install "${training_python}" \
            --constraint "${container_files}/constraints.txt" \
            --extra-index-url https://pypi.nvidia.com \
            "${ISAACSIM_CACHE_PACKAGE}==${ISAACSIM_VERSION}.0"
        validate_distribution "${training_python}" "${ISAACSIM_CACHE_PACKAGE}" \
            "${ISAACSIM_VERSION}.0"
        ;;

    training-project)
        : "${TORCH_VERSION:?TORCH_VERSION is required}"
        : "${TORCH_CUDA:?TORCH_CUDA is required}"
        : "${ISAACSIM_VERSION:?ISAACSIM_VERSION is required}"
        : "${ISAACLAB_REF:?ISAACLAB_REF is required}"
        : "${SONIC_REPO:?SONIC_REPO is required}"
        : "${SONIC_REF:?SONIC_REF is required}"
        : "${SMPLSIM_REPO:?SMPLSIM_REPO is required}"
        : "${SMPLSIM_REF:?SMPLSIM_REF is required}"
        : "${SMPLX_REPO:?SMPLX_REPO is required}"
        : "${SMPLX_REF:?SMPLX_REF is required}"

        [[ -x "${training_python}" ]] \
            || { echo "training-base must run before training-project" >&2; exit 2; }
        [[ -f "${ISAACLAB_PATH}/VERSION" ]] \
            || { echo "training-isaac must run before training-project" >&2; exit 2; }
        for isaac_cache_distribution in \
            isaacsim-extscache-kit \
            isaacsim-extscache-kit-sdk \
            isaacsim-extscache-physics; do
            validate_distribution \
                "${training_python}" \
                "${isaac_cache_distribution}" \
                "${ISAACSIM_VERSION}.0"
        done
        # The final Isaac runtime deliberately omits wheel because wheel 0.46.2
        # requires packaging>=24 while Isaac Sim pins packaging==23. Install it
        # transiently so this mode is also safe to rerun after finalization.
        pip_install "${training_python}" --no-deps "wheel==0.46.2"
        # SONIC uses Isaac Lab core directly, not isaaclab_rl/tasks/mimic.
        # Installing every source/* package introduces mutually incompatible
        # web/packaging requirements and is unnecessary for the current trainer.
        pip_install "${training_python}" \
            --constraint "${container_files}/constraints.txt" \
            --requirement "${container_files}/requirements-training.txt"
        pip_install "${training_python}" \
            --constraint "${container_files}/constraints.txt" \
            --no-build-isolation \
            "flatdict==4.0.1"
        # Bake a pinned public source fallback without installing SONIC package
        # metadata: its declared NumPy 1.26.4 conflicts with Isaac Sim's exact
        # NumPy 1.26.0 requirement. A private checkout mounted at
        # /workspace/sonic-training takes precedence through PYTHONPATH.
        GIT_LFS_SKIP_SMUDGE=1 clone_pinned \
            "${SONIC_REPO}" "${SONIC_REF}" "${SONIC_UPSTREAM_PATH}"
        clone_pinned "${SMPLSIM_REPO}" "${SMPLSIM_REF}" /opt/SMPLSim
        clone_pinned "${SMPLX_REPO}" "${SMPLX_REF}" /opt/smplx
        site_packages="$("${training_python}" -c \
            'import site; print(site.getsitepackages()[0])')"
        printf '%s\n' \
            "${ISAACLAB_PATH}/source/isaaclab" \
            /opt/SMPLSim \
            /opt/smplx \
            "${SONIC_UPSTREAM_PATH}" \
            > "${site_packages}/agile-sonic-source.pth"
        install -d -m 0755 /opt/agile-sonic/manifests
        "${training_python}" -c \
            'import torch, h5py, isaaclab, open3d, transformers, trl, accelerate, tensordict, pink, pinocchio; import gear_sonic.train_agent_trl; from smpl_sim.smpllib import smpl_eval; print("training imports: ok")' \
            > /opt/agile-sonic/manifests/agile-sonic.import-smoke.txt
        LD_PRELOAD="${AGILE_SONIC_NCCL_LIBRARY}" "${training_python}" -c \
            'import ctypes, os; value=ctypes.c_int(); lib=ctypes.CDLL(None); rc=lib.ncclGetVersion(ctypes.byref(value)); major, minor, patch=map(int, os.environ["AGILE_SONIC_NCCL_RUNTIME_VERSION"].split(".")); expected=major * 10000 + minor * 100 + patch; assert rc == 0 and value.value == expected, (rc, value.value, expected); print(f"NCCL runtime override: {value.value}")' \
            > /opt/agile-sonic/manifests/agile-sonic.nccl-runtime.txt

        validate_distribution "${training_python}" torch \
            "${TORCH_VERSION}+${TORCH_CUDA}"
        validate_distribution "${training_python}" isaacsim \
            "${ISAACSIM_VERSION}.0"
        # wheel 0.46.2 is needed while building source packages, but its runtime
        # packaging requirement is intentionally absent from the immutable
        # Isaac environment (which must retain packaging==23.0).
        "${training_python}" -m pip uninstall --yes wheel
        "${training_python}" -m pip check \
            | tee /opt/agile-sonic/manifests/agile-sonic.pip-check.txt
        "${training_python}" -m pip freeze --all \
            | LC_ALL=C sort \
            > /opt/agile-sonic/manifests/agile-sonic.freeze.txt
        printf '%s\n' \
            "IsaacLab ${ISAACLAB_REF}" \
            "SONIC ${SONIC_REF}" \
            "SMPLSim ${SMPLSIM_REF}" \
            "smplx ${SMPLX_REF}" \
            > /opt/agile-sonic/manifests/agile-sonic.sources.txt
        refresh_sources_manifest
        dpkg-query -W -f='${binary:Package}=${Version}\n' \
            | LC_ALL=C sort \
            > /opt/agile-sonic/manifests/system-packages.txt
        {
            nvcc --version
            aws --version
            aliyun version
            "${conda_bin}" --version
        } > /opt/agile-sonic/manifests/toolchain.txt 2>&1
        ;;

    tensorrt)
        : "${ONNXRUNTIME_VERSION:?ONNXRUNTIME_VERSION is required}"
        : "${TENSORRT_VERSION:?TENSORRT_VERSION is required}"
        : "${CUDA_PYTHON_VERSION:?CUDA_PYTHON_VERSION is required}"

        # training-project removes wheel from the final runtime. Reinstall it
        # only for this build step so rerunning the target remains deterministic.
        pip_install "${training_python}" --no-deps "wheel==0.46.2"
        pip_install "${training_python}" \
            --constraint "${container_files}/constraints.txt" \
            --requirement "${container_files}/requirements-tensorrt.txt"
        validate_distribution "${training_python}" onnxruntime \
            "${ONNXRUNTIME_VERSION}"
        validate_distribution "${training_python}" tensorrt-cu12 \
            "${TENSORRT_VERSION}"
        "${training_python}" -c \
            'import onnxruntime, tensorrt; from cuda import bindings; print("TensorRT imports: ok")' \
            > /opt/agile-sonic/manifests/agile-sonic-tensorrt.import-smoke.txt
        "${training_python}" -m pip uninstall --yes wheel
        "${training_python}" -m pip check \
            | tee /opt/agile-sonic/manifests/agile-sonic-tensorrt.pip-check.txt
        "${training_python}" -m pip freeze --all \
            | LC_ALL=C sort \
            > /opt/agile-sonic/manifests/agile-sonic-tensorrt.freeze.txt
        ;;

    tools)
        : "${CYCLONEDDS_REPO:?CYCLONEDDS_REPO is required}"
        : "${CYCLONEDDS_REF:?CYCLONEDDS_REF is required}"
        : "${ROBOSUITE_REPO:?ROBOSUITE_REPO is required}"
        : "${ROBOSUITE_REF:?ROBOSUITE_REF is required}"
        : "${TOOLS_TORCH_VERSION:?TOOLS_TORCH_VERSION is required}"
        : "${TOOLS_TORCHVISION_VERSION:?TOOLS_TORCHVISION_VERSION is required}"
        : "${TOOLS_TORCHAUDIO_VERSION:?TOOLS_TORCHAUDIO_VERSION is required}"
        : "${TOOLS_TORCH_CUDA:?TOOLS_TORCH_CUDA is required}"

        clone_pinned "${CYCLONEDDS_REPO}" "${CYCLONEDDS_REF}" /opt/cyclonedds
        cmake -S /opt/cyclonedds -B /opt/cyclonedds/build \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="${CYCLONEDDS_HOME}" \
            -DBUILD_EXAMPLES=OFF \
            -DBUILD_TESTING=OFF
        cmake --build /opt/cyclonedds/build \
            --target install \
            --parallel "$(nproc)"
        rm -rf /opt/cyclonedds/build

        if [[ ! -x "${tools_prefix}/bin/python" ]]; then
            CONDA_PKGS_DIRS=/opt/conda-pkgs "${conda_bin}" env create \
                --prefix "${tools_prefix}" \
                --file "${container_files}/agile-sonic-tools.yml"
        fi
        tools_python="${tools_prefix}/bin/python"
        install_torch \
            "${tools_python}" \
            "${container_files}/constraints-tools.txt" \
            "${TOOLS_TORCH_VERSION}" \
            "${TOOLS_TORCHVISION_VERSION}" \
            "${TOOLS_TORCHAUDIO_VERSION}" \
            "${TOOLS_TORCH_CUDA}"
        pip_install "${tools_python}" \
            --constraint "${container_files}/constraints-tools.txt" \
            --requirement "${container_files}/requirements-full.txt"

        clone_pinned "${ROBOSUITE_REPO}" "${ROBOSUITE_REF}" /opt/robosuite
        git -C "${SONIC_UPSTREAM_PATH}" lfs pull \
            --include="external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/**" \
            --exclude=""
        pip_install "${tools_python}" \
            --constraint "${container_files}/constraints-tools.txt" \
            --no-deps \
            --editable "${SONIC_UPSTREAM_PATH}/external_dependencies/unitree_sdk2_python"

        pybind11_cmake_dir="$("${tools_python}" -m pybind11 --cmakedir)"
        CMAKE_PREFIX_PATH="${pybind11_cmake_dir}:${CMAKE_PREFIX_PATH}" \
        pip_install "${tools_python}" \
            --constraint "${container_files}/constraints-tools.txt" \
            --no-build-isolation \
            --no-deps \
            --editable "${SONIC_UPSTREAM_PATH}/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"

        tools_site_packages="$("${tools_python}" -c \
            'import site; print(site.getsitepackages()[0])')"
        printf '%s\n' \
            /workspace/sonic-training \
            /workspace/sonic-training/decoupled_wbc/dexmg/gr00trobocasa \
            /opt/SMPLSim \
            /opt/smplx \
            /opt/robosuite \
            "${SONIC_UPSTREAM_PATH}" \
            "${SONIC_UPSTREAM_PATH}/decoupled_wbc/dexmg/gr00trobocasa" \
            > "${tools_site_packages}/agile-sonic-source.pth"
        "${tools_python}" -c \
            'import av, cv2, mujoco, robosuite, robocasa, depthai, cyclonedds, unitree_sdk2py, xrobotoolkit_sdk, pyrealsense2, onnxruntime; print("tools imports: ok")' \
            > /opt/agile-sonic/manifests/agile-sonic-tools.import-smoke.txt

        # depthai currently reports an unsupported-platform metadata warning on
        # generic x86_64 CI although its wheel is intentionally included for OAK
        # camera hosts. Any other dependency error remains fatal.
        check_log=/opt/agile-sonic/manifests/agile-sonic-tools.pip-check.txt
        if ! "${tools_python}" -m pip check >"${check_log}" 2>&1; then
            mapfile -t check_lines < "${check_log}"
            if (( "${#check_lines[@]}" != 1 )) \
                || ! [[ "${check_lines[0]}" =~ ^depthai\ .*is\ not\ supported\ on\ this\ platform$ ]]; then
                cat "${check_log}" >&2
                exit 1
            fi
            cat "${check_log}" >&2
        fi
        "${tools_python}" -m pip freeze --all \
            | LC_ALL=C sort \
            > /opt/agile-sonic/manifests/agile-sonic-tools.freeze.txt
        printf '%s\n' \
            "CycloneDDS ${CYCLONEDDS_REF}" \
            "robosuite ${ROBOSUITE_REF}" \
            "unitree_sdk2_python vendored in SONIC ${SONIC_REF}" \
            "XRoboToolkit vendored in SONIC ${SONIC_REF}" \
            > /opt/agile-sonic/manifests/agile-sonic-tools.sources.txt
        refresh_sources_manifest
        ;;

    data)
        : "${DATA_TORCH_VERSION:?DATA_TORCH_VERSION is required}"
        : "${DATA_TORCHVISION_VERSION:?DATA_TORCHVISION_VERSION is required}"
        : "${DATA_TORCHAUDIO_VERSION:?DATA_TORCHAUDIO_VERSION is required}"
        : "${DATA_TORCH_CUDA:?DATA_TORCH_CUDA is required}"
        : "${LEROBOT_REPO:?LEROBOT_REPO is required}"
        : "${LEROBOT_REF:?LEROBOT_REF is required}"

        data_prefix="${CONDA_DIR}/envs/agile-sonic-data"
        if [[ ! -x "${data_prefix}/bin/python" ]]; then
            CONDA_PKGS_DIRS=/opt/conda-pkgs "${conda_bin}" env create \
                --prefix "${data_prefix}" \
                --file "${container_files}/agile-sonic-data.yml"
        fi
        data_python="${data_prefix}/bin/python"
        install_torch \
            "${data_python}" \
            "${container_files}/constraints-data.txt" \
            "${DATA_TORCH_VERSION}" \
            "${DATA_TORCHVISION_VERSION}" \
            "${DATA_TORCHAUDIO_VERSION}" \
            "${DATA_TORCH_CUDA}"
        pip_install "${data_python}" \
            --constraint "${container_files}/constraints-data.txt" \
            --requirement "${container_files}/requirements-data.txt"

        GIT_LFS_SKIP_SMUDGE=1 clone_pinned \
            "${LEROBOT_REPO}" "${LEROBOT_REF}" /opt/lerobot
        pip_install "${data_python}" \
            --constraint "${container_files}/constraints-data.txt" \
            --no-deps \
            --editable /opt/lerobot

        data_site_packages="$("${data_python}" -c \
            'import site; print(site.getsitepackages()[0])')"
        printf '%s\n' \
            /workspace/sonic-training \
            "${SONIC_UPSTREAM_PATH}" \
            > "${data_site_packages}/agile-sonic-source.pth"
        "${data_python}" -c \
            'import av, torch, torchcodec, gymnasium, datasets, lerobot; from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; print("data imports: ok")' \
            > /opt/agile-sonic/manifests/agile-sonic-data.import-smoke.txt
        "${data_python}" -m pip check \
            | tee /opt/agile-sonic/manifests/agile-sonic-data.pip-check.txt
        "${data_python}" -m pip freeze --all \
            | LC_ALL=C sort \
            > /opt/agile-sonic/manifests/agile-sonic-data.freeze.txt
        printf '%s\n' "LeRobot ${LEROBOT_REF}" \
            > /opt/agile-sonic/manifests/agile-sonic-data.sources.txt
        refresh_sources_manifest
        ;;

    inference)
        : "${INFERENCE_TORCH_VERSION:?INFERENCE_TORCH_VERSION is required}"
        : "${INFERENCE_TORCHVISION_VERSION:?INFERENCE_TORCHVISION_VERSION is required}"
        : "${INFERENCE_TORCHAUDIO_VERSION:?INFERENCE_TORCHAUDIO_VERSION is required}"
        : "${INFERENCE_TORCH_CUDA:?INFERENCE_TORCH_CUDA is required}"
        : "${GROOT_REPO:?GROOT_REPO is required}"
        : "${GROOT_REF:?GROOT_REF is required}"

        inference_prefix="${CONDA_DIR}/envs/agile-sonic-inference"
        if [[ ! -x "${inference_prefix}/bin/python" ]]; then
            CONDA_PKGS_DIRS=/opt/conda-pkgs "${conda_bin}" env create \
                --prefix "${inference_prefix}" \
                --file "${container_files}/agile-sonic-inference.yml"
        fi
        inference_python="${inference_prefix}/bin/python"
        install_torch \
            "${inference_python}" \
            "${container_files}/constraints-inference.txt" \
            "${INFERENCE_TORCH_VERSION}" \
            "${INFERENCE_TORCHVISION_VERSION}" \
            "${INFERENCE_TORCHAUDIO_VERSION}" \
            "${INFERENCE_TORCH_CUDA}"
        pip_install "${inference_python}" \
            --constraint "${container_files}/constraints-inference.txt" \
            --extra-index-url https://pypi.nvidia.com \
            --requirement "${container_files}/requirements-inference.txt"

        GIT_LFS_SKIP_SMUDGE=1 clone_pinned \
            "${GROOT_REPO}" "${GROOT_REF}" /opt/Isaac-GR00T
        pip_install "${inference_python}" \
            --constraint "${container_files}/constraints-inference.txt" \
            --no-deps \
            --editable /opt/Isaac-GR00T

        inference_site_packages="$("${inference_python}" -c \
            'import site; print(site.getsitepackages()[0])')"
        printf '%s\n' \
            /workspace/sonic-training \
            "${SONIC_UPSTREAM_PATH}" \
            > "${inference_site_packages}/agile-sonic-source.pth"
        "${inference_python}" -c \
            'import torch, gr00t, pinocchio, flash_attn, deepspeed, torchcodec, tensorrt; import gear_sonic.scripts.run_vla_inference; print("inference imports: ok")' \
            > /opt/agile-sonic/manifests/agile-sonic-inference.import-smoke.txt
        "${inference_python}" -m pip check \
            | tee /opt/agile-sonic/manifests/agile-sonic-inference.pip-check.txt
        "${inference_python}" -m pip freeze --all \
            | LC_ALL=C sort \
            > /opt/agile-sonic/manifests/agile-sonic-inference.freeze.txt
        printf '%s\n' "Isaac-GR00T ${GROOT_REF}" \
            > /opt/agile-sonic/manifests/agile-sonic-inference.sources.txt
        refresh_sources_manifest
        ;;

    *)
        echo "Usage: $0 {training-base|training-isaac|training-isaac-cache|training-project|tensorrt|tools|data|inference}" >&2
        exit 2
        ;;
esac
