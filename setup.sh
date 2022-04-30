#!/bin/bash

# setup conda
source ~/miniconda3/etc/profile.d/conda.sh

# create conda env
read -rp "Enter environment name: " env_name
read -rp "Enter python version (e.g. 3.7) " python_version
conda create -yn "$env_name" python="$python_version"
conda activate "$env_name"

# install torch
read -rp "Enter cuda version (e.g. '11.1' or 'none' to avoid installing cuda support): " cuda_version
if [ "$cuda_version" == "none" ]; then
    conda install -y pytorch torchvision cpuonly -c pytorch
else
    conda install -y pytorch torchvision cudatoolkit=$cuda_version -c pytorch
fi

# download STEPS dataset
read -p "Download STEPS dataset? [y/N] "
if [[ $REPLY =~ ^[Yy]$ ]]
then
  wget -O data/STEPS_dataset.zip "https://drive.google.com/uc?export=download&id=1637VyjvHvrtgzn4sAY7MA6QuwY6FNRG_" 
  unzip -d data/ data/STEPS_dataset.zip
  rm data/STEPS_dataset.zip
fi

# install python requirements
pip install -r requirements.txt

