#!/bin/bash

# Script to install OFED >= 5.9, CUDA 12.4.1, NCCL >= 2.16, and UCX >= 1.13
# This script is intended for Ubuntu/Debian-based systems
# Run with sudo privileges

set -e  # Exit on error
set -x  # Print commands for debugging

echo "Starting installation of OFED, CUDA, NCCL, and UCX packages"

# Install basic dependencies
apt-get update
apt-get install -y wget gnupg2 software-properties-common build-essential

# Install OFED (Mellanox OFED) >= 5.9
echo "Installing OFED >= 5.9"
OFED_VERSION="5.9-0.5.9.0"
OFED_OS="ubuntu20.04"
OFED_ARCH="x86_64"
OFED_PACKAGE="MLNX_OFED_LINUX-${OFED_VERSION}-${OFED_OS}-${OFED_ARCH}"
OFED_URL="https://content.mellanox.com/ofed/${OFED_PACKAGE}/${OFED_PACKAGE}.tgz"

cd /tmp
wget ${OFED_URL}
tar -xzf ${OFED_PACKAGE}.tgz
cd ${OFED_PACKAGE}
./mlnxofedinstall --auto-add-kernel-support --force
/etc/init.d/openibd restart

# Install CUDA 12.4.1
echo "Installing CUDA 12.4.1"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get install -y cuda-12-4=12.4.1-1

# Set CUDA environment variables
echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >> /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh

# Install NCCL >= 2.16
echo "Installing NCCL >= 2.16"
apt-get install -y libnccl2=2.16.* libnccl-dev=2.16.*

# Install UCX >= 1.13
echo "Installing UCX >= 1.13"
apt-get install -y autoconf automake libtool
cd /tmp
git clone https://github.com/openucx/ucx.git
cd ucx
git checkout v1.13.0
./autogen.sh
./configure --prefix=/usr/local
make -j$(nproc)
make install
ldconfig

echo "Installation completed!"
echo "Please reboot your system to ensure all components are properly loaded."