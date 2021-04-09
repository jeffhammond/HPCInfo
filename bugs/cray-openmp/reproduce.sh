module load cgpu
module load PrgEng-cray
module load craype-accel-nvidia70

ftn --version
Cray Fortran : Version 9.1.0

ftn -O0 -h omp -hmsgs -hlist=m nstream-openmp-target.F90 -o nstream-openmp-target >& comp.log && srun ./nstream-openmp-target >& run.log
ftn -O0 -h omp -hmsgs -hlist=m nstream-openmp-target2.F90 -o nstream-openmp-target2 >& comp2.log && srun ./nstream-openmp-target2 >& run2.log

