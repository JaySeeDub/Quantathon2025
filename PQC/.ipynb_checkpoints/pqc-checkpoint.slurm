#!/bin/bash
#SBATCH --job-name=pqc_SCQ
#SBATCH --output=logs/pqc_SCQ.txt
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=32
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amacieju@odu.edu


module load tensorflow-gpu/2.16.1
crun -p /home/amaci003/envs/snrl_SCQ python -u pqc.py

