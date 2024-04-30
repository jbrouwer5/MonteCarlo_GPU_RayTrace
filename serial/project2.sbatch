#!/bin/bash
#SBATCH --time=0:01:00
#SBATCH --account=mpcs51087
#SBATCH --partition=caslake
#SBATCH --job-name=parallelDP
#SBATCH --output=parallelDP.out
#SBATCH --error=parallelDP.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2000 
#SBATCH --mail-user=jbrouwer@uchicago.edu
#SBATCH --mail-type=ALL 

./raytrace 1000000000 1000 48 -count=1