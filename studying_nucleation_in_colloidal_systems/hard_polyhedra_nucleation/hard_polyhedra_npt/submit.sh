#!/bin/bash
#PBS -q SmallCPUshort
#PBS -N ThP70
#PBS -o out2.log
#PBS -e err1.log
#PBS -l nodes=2:ppn=50

cd $PBS_O_WORKDIR

module purge

module load libs/openblas

cat $PBS_NODEFILE

export CC=/apps/compilers/gcc/9.3.0/bin/gcc
export CXX=/apps/compilers/gcc/9.3.0/bin/g++

export PATH=/home/avisek/my_install/Python-3.8.0/bin:/home/avisek/my_install/cmake-v3.20.1/bin:/home/avisek/my_install/openmpi-4.1.4/bin:/apps/compilers/gcc/9.3.0/bin:$PATH

export LD_LIBRARY_PATH=/home/avisek/my_install/Python-3.8.0/lib:/home/avisek/my_install/openmpi-4.1.4/lib:/home/avisek/my_install/cmake-v3.20.1/lib:/apps/compilers/gcc/9.3.0/lib64:$LD_LIBRARY_PATH

unset PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/avisek/my_install/hoomd-v4.4.0/lib/python

export OMPI_MCA_btl="^openib"

echo "Running on:"
hostname
echo "Python path:"
which python3.8

python3.8 -c "import hoomd; print('HOOMD version:', hoomd.version.version)"

NP=`cat $PBS_NODEFILE|wc -l`

time /home/avisek/my_install/openmpi-4.1.4/bin/mpirun -x LD_LIBRARY_PATH -x PYTHONPATH --hostfile $PBS_NODEFILE -n $NP /home/avisek/my_install/Python-3.8.0/bin/python3.8 HOOMD_hard_polyhedra_NPT.py --simulparam_file=simulparam_hard_polyhedra_npt.json > output_NPT.log 2>&1

mpi_status=$?
echo "[PBS] mpirun exit status: ${mpi_status}" >> output_NPT.log
exit "${mpi_status}"