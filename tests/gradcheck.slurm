#!/bin/bash
#SBATCH --job-name=gradcheck
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a10:1
#SBATCH --partition=gpu
#SBATCH --output=/scratch/user/tahaismail424/coursework/cs633/clip-project/tests/gradcheck.%j.out
cd /scratch/user/tahaismail424/coursework/cs633/clip-project
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python /scratch/user/tahaismail424/coursework/cs633/clip-project/tests/gradcheck.py
