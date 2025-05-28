# Basics
sudo apt-get install vim git htop build-essential pciutils curl wget gcc

# Setup Mellanox
wget -qO - http://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | sudo gpg --dearmor -o /usr/share/keyrings/GPG-KEY-Mellanox.gpg
echo "deb [signed-by=/usr/share/keyrings/GPG-KEY-Mellanox.gpg] https://linux.mellanox.com/public/repo/mInx_ofed/latest/ubuntu22.04/x86_64 /" | sudo tee /etc/apt/sources.list.d/mlnx.list > /dev/null
sudo apt-get install mlnx-fw-updater mlnx-ofed-all -y

# Setup CUDA
wget -O /tmp/cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb
sudo apt-get install cuda-drivers-555 nvidia-kernel-open-555 linux-tools-$(uname -r) -y

# Nvidia Toolkit
sudo apt install cuda-toolkit nvidia-container-toolkit cudnn libnccl2 nvidia-gds libnccl-dev -y

sudo apt-get update
sudo apt-get install libnccl2 libnccl-dev -y
sudo apt-get install libnvidia-compute-560:i386 libnvidia-decode-560:i386 \
 libnvidia-encode-560:i386 libnvidia-extra-560:i386 libnvidia-fbc1-560:i386 \
 libnvidia-gl-560:i386 -y
