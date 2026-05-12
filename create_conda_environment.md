pip install "lightly[timm]"Run this command to create a conda environment
conda create --name lightly_env python=3.11 -y

activate the environment with
conda activate lightly_env

This next line with install pytorch, you will need to adjust the pytorch-cuda version to match your current GPU setup.
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

then install lightly

!pip install lightly
pip install "lightly[video]"
pip install "lightly[timm]"



The mill is a nightmare
# 1. Remove the broken env
conda deactivate
conda env remove -n lightly_env

# 2. Create a new one with the core math libraries pinned
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda activate lightly_env

# 3. Install PyTorch via the official channel (use cpuonly since you're on a CPU node)
conda install pytorch torchvision cpuonly -c pytorch -y

# 4. Install lightly WITHOUT its dependencies (to prevent it from reinstalling a bad torch)
pip install lightly --no-deps