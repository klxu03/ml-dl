#!/bin/bash
#
#SBATCH --job-name=VGG_DL
#SBATCH --output=VGG_DL.out.log
#SBATCH --error=VGG_DL.err.log
#
# Number of tasks needed for this job. Generally, used with MPI jobs
#SBATCH --ntasks=1
#SBATCH --partition=a100
#
# Time format = HH:MM:SS, DD-HH:MM:SS
#SBATCH --time=72:00:00
#
# Minimum memory required per allocated  CPU  in  MegaBytes.
#SBATCH --mem-per-cpu=48000
#SBATCH --gres=gpu:1
#SBATCH -A danielk_gpu
#
# Send mail to the email address when the job fails
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kxu39@jhu.edu

# Load necessary modules
module load anaconda
module load cuda/11.8

source ~/.bashrc

conda activate vgg

module list

python --version
which python

source .env

# Run the Python script
~/miniconda3/envs/vgg/bin/python VGG.py