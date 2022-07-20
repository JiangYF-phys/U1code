#!/bin/bash

# job name
#SBATCH -x g0001
#SBATCH --job-name=x36y6d9Utest
# output file
#SBATCH --output=status.%J.out
# error file
#SBATCH --error=status.%J.err


ulimit -s unlimited
module load cuda/11.4 intel/mkl/2019

echo BEGIN TIME: $(date +"%F %T")
#echo ${SLURM_SUBMIT_DIR}
#cd ${SLURM_SUBMIT_DIR}
cp temp/* back/
#cp back/* temp/

OMP_NUM_THREADS=10

srun -n 1 dmrg
#nvprof --unified-memory-profiling off --log-file log ./dmrg 
#cuda-memcheck ./dmrg

rm temp/*bl*

echo END TIME: $(date +"%F %T")

