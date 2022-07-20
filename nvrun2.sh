#!/bin/bash

# job name
#SBATCH -x g0001
#SBATCH --job-name=SU2_test
# output file
#SBATCH --output=status.%J.out
# error file
#SBATCH --error=status.%J.err


ulimit -s unlimited
#module load cuda/11.4 intel/mkl/2019
for i in `export | grep SLURM | awk '{print $3}' | awk -F '=' '{print $1}'`
do
unset ${i}
done

echo BEGIN TIME: $(date +"%F %T")
#echo ${SLURM_SUBMIT_DIR}
#cd ${SLURM_SUBMIT_DIR}
#cp temp/* back/
cp back/* temp/

OMP_NUM_THREADS=10

#srun -n 1 dmrg
#nvprof --log-file log ./dmrg 
#nvprof --unified-memory-profiling off --log-file log ./dmrg 
nsys profile -o result --stats=true ./dmrg 
#cuda-memcheck ./dmrg

#rm *bl* *trun*

echo END TIME: $(date +"%F %T")

