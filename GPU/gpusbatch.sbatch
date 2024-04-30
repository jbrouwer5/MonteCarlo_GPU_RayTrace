#!/bin/bash
#SBATCH --time=0:00:25
#SBATCH --account=mpcs51087
#SBATCH --partition=gpu
#SBATCH --job-name=gpuexc
#SBATCH --output=gpuexc.out
#SBATCH --error=gpuexc.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=jbrouwer@uchicago.edu
#SBATCH --constraint=v100

./raytrace 1000000000 1000 1000000 100