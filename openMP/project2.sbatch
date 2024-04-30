#!/bin/bash
#SBATCH --time=0:02:00
#SBATCH --account=mpcs51087
#SBATCH --partition=caslake
#SBATCH --job-name=parallelDP
#SBATCH --output=parallelDP.out
#SBATCH --error=parallelDP.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-user=jbrouwer@uchicago.edu
#SBATCH --mail-type=ALL  
#SBATCH --exclusive

./raytrace 1000000000 1000 16