export NWCHEM_TOP=${HOME}/NWCHEM/gcc
export NWCHEM_TARGET=LINUX64
export NWCHEM_MODULES="all"

MPI_PATH=/opt/amazon/openmpi
export PATH=${MPI_PATH}/bin/:$PATH
export LD_LIBRARY_PATH=${MPI_PATH}/lib:$LD_LIBRARY_PATH

export FC=gfortran
export CC=gcc
export CXX=g++

export USE_MPI=T
export MPI_INCLUDE=${MPI_PATH}/include
export MPI_LIB=${MPI_PATH}/lib64
export LIBMPI="-lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lpthread"
#export LIBMPI="-lmpi -lpthread"

#export ARMCI_NETWORK=MPI-PR
export ARMCI_NETWORK=ARMCI
export EXTERNAL_ARMCI_PATH=${NWCHEM_TOP}/external-armci

export MPIFC=${MPI_PATH}/bin/mpifort
export MPIF77=${MPI_PATH}/bin/mpifort
export MPICC=${MPI_PATH}/bin/mpicc
export MPICXX=${MPI_PATH}/bin/mpicxx
export NO_MPIF=y

#export USE_INTERNALBLAS=y
export BLASOPT="-DOPENBLAS /software/ACFL/21.0/armpl-21.0.0_AArch64_RHEL-7_gcc_aarch64-linux/lib/libarmpl_ilp64.a"
export BLAS_LIB=${BLASOPT}
export LAPACK_LIB=${BLASOPT}

#export BUILD_OPENBLAS=y

export USE_OPENMP=y
export USE_F90_ALLOCATABLE=y

export USE_SIMINT=1 
export SIMINT_MAXAM=5
