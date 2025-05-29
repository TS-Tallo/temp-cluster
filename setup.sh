# Basics
sudo apt-get install vim git htop build-essential pciutils curl wget gcc -y

# Setup Mellanox
wget -qO - http://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | sudo gpg --dearmor -o /usr/share/keyrings/GPG-KEY-Mellanox.gpg
echo "deb [signed-by=/usr/share/keyrings/GPG-KEY-Mellanox.gpg] https://linux.mellanox.com/public/repo/mInx_ofed/latest/ubuntu22.04/x86_64 /" | sudo tee /etc/apt/sources.list.d/mlnx.list > /dev/null
sudo apt-get install mlnx-fw-updater mlnx-ofed-all -y

# Nvidia Toolkit
sudo apt install cuda-toolkit nvidia-container-toolkit cudnn libnccl2 nvidia-gds libnccl-dev -y

sudo apt-get update
sudo apt-get install libnccl2 libnccl-dev -y
sudo apt-get install libnvidia-compute-560:i386 libnvidia-decode-560:i386 \
 libnvidia-encode-560:i386 libnvidia-extra-560:i386 libnvidia-fbc1-560:i386 \
 libnvidia-gl-560:i386 -y

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install Docker
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
