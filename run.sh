#!/bin/bash

# job name
#SBATCH --job-name=test
# output file
#SBATCH --output=status.%J.out
# error file
#SBATCH --error=status.%J.err


ulimit -s unlimited

echo BEGIN TIME: $(date +"%F %T")
#echo ${SLURM_SUBMIT_DIR}
#cd ${SLURM_SUBMIT_DIR}
cp temp/* back/

export OMP_NUM_THREADS=9

yhrun -N 1 -p gpu --gpus-per-node=1 --cpus-per-gpu=9 dmrg

rm temp/*bl*

echo END TIME: $(date +"%F %T")

