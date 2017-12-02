# Platforms

## Kayla

* https://developer.nvidia.com/content/kayla-platform

## Parallella

* http://www.parallella.org/board/
* https://github.com/adapteva

## Rasberry Pi

* http://www.raspberrypi.org/

# Software

## MPICH

No special options required.

```
../../git/configure CC=gcc CXX=g++ FC=gfortran F77=gfortran --enable-fc --enable-f77 \
--enable-threads=runtime --with-pm=hydra --prefix=$HOME/MPICH/kayla/install
```

## BLAS/LAPACK

### Netlib

Working fine.

### BLIS

Required a bug fix but fine now.

### OpenBLAS

#### Binary

* http://sourceforge.net/projects/openblas/files/v0.2.8-arm/

#### Source

Required a bug fix but fine now.

* https://github.com/xianyi/OpenBLAS and ``make TARGET=ARMV7 NO_LAPACK=1 NO_LAPACKE=1 NO_SHARED=1``

### ATLAS

The 3.10.1 release is detecting 32KB L1 cache and Neon FPU ISA while 3.11.17 and later are detecting 16KB L1 cache and FPV3D32MAC FPU ISA.

I have not built ATLAS successfully but the following is my latest attempt:

```
 ../../atlas-3.11.22/configure --prefix=$HOME/ATLAS/kayla/install -m 1400 \
-Fa ac "-I/usr/include/arm-linux-gnueabihf" -D c -DATL_NONIEEE=1 \
-D c -DATL_ARM_HARDFP=1 -Si archdef 0 -Fa alg -mfloat-abi=hard
```

## TBB

Just type ``make``.

## NWChem

No source changes were required.  This is an absolutely straightforward build for 32-bit Linux.

Please see Phase 1 and Phase 2 of http://wiki.mpich.org/armci-mpi/index.php/NWChem for building ARMCI-MPI.

```
#=================================================
# GA Settings
#=================================================

export TARGET=LINUX
export USE_MPI=yes

export ARMCI_NETWORK=ARMCI
export EXTERNAL_ARMCI_PATH="${HOME}/ARMCI-MPI/${HOST}/install"

MPI_DIR="${HOME}/MPICH/${HOST}/install"
MPICH_LIBS="-lmpich -lopa -lmpl -lrt"
export MPI_LIB="${MPI_DIR}/lib"
export MPI_INCLUDE="${MPI_DIR}/include"
export LIBMPI="-L${MPI_DIR}/lib -Wl,-rpath -Wl,${MPI_DIR}/lib ${MPICH_LIBS}"

export MPICC=mpicc
export MPICXX=mpicxx
export MPIF77=mpif77

#=================================================
# NWChem Settings
#=================================================

export FC=gfortran
export CC=gcc

export NWCHEM_TARGET=LINUX
export NWCHEM_MODULES=all
export NWCHEM_TOP=/tmp/nwchem-6.3.revision2-src.2013-10-17

export BLASOPT="-llapack_atlas -llapack -lf77blas -latlas"

alias makenw="make FC=$FC CC=$CC NWCHEM_TOP=$NWCHEM_TOP"
```
