#!/bin/bash

# install pyenv. https://github.com/pyenv/pyenv?tab=readme-ov-file#unixmacos

sudo apt-get install build-essential
pyenv install 3.9.2
#
# https://gist.github.com/tranctan/7136955aaf2a1457301b68ed2b2ea4d4

# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_network

# https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_local
# 1) install CUDA Toolkit Installer

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-debian11-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda

# .. and 2) Driver Installer

sudo apt-get install -y nvidia-open

# https://pytorch.org/get-started/locally/