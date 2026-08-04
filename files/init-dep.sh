#!/usr/bin/env bash
set -Eeuo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get -o Acquire::Retries=5 update
apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
    apt-utils \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    espeak \
    ffmpeg \
    git \
    git-lfs \
    gpg \
    jq \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libdbus-1-3 \
    libegl1 \
    libegl1-mesa \
    libegl1-mesa-dev \
    libgl1 \
    libgl1-mesa-dev \
    libgl1-mesa-dri \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libglvnd-dev \
    libglu1-mesa \
    libgtk2.0-dev \
    libncurses5-dev \
    libsm6 \
    libswresample-dev \
    libswscale-dev \
    libudev-dev \
    libusb-1.0-0-dev \
    libvulkan1 \
    libx11-6 \
    libx11-xcb1 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libxcb-xkb1 \
    libxcursor1 \
    libxext6 \
    libxi6 \
    libxinerama1 \
    libxkbcommon-x11-0 \
    libxrandr2 \
    libxrender1 \
    locales \
    lsb-release \
    mesa-utils \
    net-tools \
    openssh-client \
    pkg-config \
    rsync \
    sudo \
    tmux \
    tzdata \
    unzip \
    vim-tiny \
    vulkan-tools \
    wget \
    xauth \
    xvfb

locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
git lfs install --system --skip-repo

apt-get clean
rm -rf /var/lib/apt/lists/*
