#!/bin/bash

# Comprehensive installation script for OFED, CUDA, NCCL, UCX, Docker, and other utilities
# This script is intended for Ubuntu 22.04 systems
# Run with sudo privileges

set -e  # Exit on error
set -x  # Print commands for debugging

echo "Starting comprehensive system setup and installation"

# Install basic utilities and dependencies
echo "Installing basic utilities"
sudo apt-get update
sudo apt-get install -y vim git htop build-essential pciutils curl wget gcc ca-certificates \
    gnupg2 software-properties-common

# Setup Mellanox OFED
echo "Setting up Mellanox OFED"
wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | sudo gpg --dearmor -o /usr/share/keyrings/GPG-KEY-Mellanox.gpg
echo "deb [signed-by=/usr/share/keyrings/GPG-KEY-Mellanox.gpg] https://linux.mellanox.com/public/repo/mlnx_ofed/latest/ubuntu22.04/x86_64 /" | sudo tee /etc/apt/sources.list.d/mlnx.list > /dev/null
sudo apt-get update
sudo apt-get install -y mlnx-fw-updater mlnx-ofed-all

# Check if NVIDIA drivers are already installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers not found, installing..."
    sudo apt install -y pkg-config libglvnd-dev dkms build-essential libegl-dev libegl1 libgl-dev libgl1 libgles-dev libgles1 libglvnd-core-dev libglx-dev libopengl-dev gcc make
    sudo add-apt-repository -y ppa:graphics-drivers/ppa
    sudo apt update
    sudo apt install -y nvidia-driver-575
    sudo reboot
else
    echo "NVIDIA drivers already installed, skipping installation"
fi

# Install CUDA 12.4.1
echo "Installing CUDA 12.4.1"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-12-4=12.4.1-1 nvidia-container-toolkit

# Set CUDA environment variables
echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' | sudo tee /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' | sudo tee -a /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh

# Install NCCL >= 2.16
echo "Installing NCCL >= 2.16"
sudo apt-get install -y libnccl2=2.16.* libnccl-dev=2.16.*

# Install additional NVIDIA libraries
sudo apt-get install -y libcudnn8 libcudnn8-dev
sudo apt-get install -y libnvidia-compute-560:i386 libnvidia-decode-560:i386 \
 libnvidia-encode-560:i386 libnvidia-extra-560:i386 libnvidia-fbc1-560:i386 \
 libnvidia-gl-560:i386

# Install UCX >= 1.13
echo "Installing UCX >= 1.13"
sudo apt-get install -y autoconf automake libtool
cd /tmp
git clone https://github.com/openucx/ucx.git
cd ucx
git checkout v1.13.0
./autogen.sh
./configure --prefix=/usr/local
make -j$(nproc)
sudo make install
sudo ldconfig

# Add Docker's official GPG key
echo "Setting up Docker"
sudo apt-get update
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the Docker repository to Apt sources
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install Docker
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "Installation completed!"
echo "Please reboot your system to ensure all components are properly loaded."