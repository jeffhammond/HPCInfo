export NWCHEM_TOP=${HOME}/NWCHEM/nvhpc
export NWCHEM_TARGET=LINUX64
export NWCHEM_MODULES="all"

export PATH=${HOME}/NVHPC/inux_aarch64/21.2/compilers/bin:$PATH
export LD_LIBRARY_PATH=${HOME}/NVHPC/Linux_aarch64/21.2/compilers/lib:$LD_LIBRARY_PATH

MPI_PATH=${HOME}/MPI/nvhpc-ompi-4.1.1
export PATH=${MPI_PATH}/bin/:$PATH
export LD_LIBRARY_PATH=${MPI_PATH}/lib:$LD_LIBRARY_PATH

module load nvhpc-nompi/21.2
export FC=nvfortran
export CC=nvc
export CXX=g++

export USE_MPI=T
export MPI_INCLUDE=${MPI_PATH}/include
export MPI_LIB=${MPI_PATH}/lib
export LIBMPI="-lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lpthread"
#export LIBMPI="-lmpi -lpthread"

#export ARMCI_NETWORK=MPI-PR
export ARMCI_NETWORK=ARMCI
export EXTERNAL_ARMCI_PATH=${NWCHEM_TOP}/external-armci

unset MPIFC
unset MPIF77
#export MPIFC=${MPI_PATH}/bin/mpifort
#export MPIF77=${MPI_PATH}/bin/mpifort
export MPICC=${MPI_PATH}/bin/mpicc
export MPICXX=${MPI_PATH}/bin/mpicxx
export NO_MPIF=y

#export USE_INTERNALBLAS=y
export BLASOPT="-DOPENBLAS ${HOME}/NVHPC/Linux_aarch64/21.2/compilers/lib/liblapack.a ${HOME}/NVHPC/Linux_aarch64/21.2/compilers/lib/libblas.a"
export BLAS_LIB=${BLASOPT}
export LAPACK_LIB=${BLASOPT}

#export BUILD_OPENBLAS=y

# ../ga-5.7.2/configure --prefix=/home/jrhammon/NWCHEM/nvhpc/src/tools/install --with-tcgmsg --with-mpi --enable-peigs --enable-underscoring --disable-mpi-tests --without-scalapack --without-lapack --without-blas --with-mpi-ts CC=mpiicc CXX=mpiicpc F77=mpiifort CFLAGS=-fPIC ARMCI_DEFAULT_SHMMAX_UBOUND=131072

export USE_OPENMP=y
export USE_F90_ALLOCATABLE=y

export USE_SIMINT=1 
export SIMINT_MAXAM=5
