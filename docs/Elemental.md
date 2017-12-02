# Overview

* [Elemental Home Page](http://libelemental.org/)
* [Elemental on Github](https://github.com/poulson/Elemental ) (for development)

# Building

## Mac Laptop


### Latest
```
rm -rf * ; \
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/Mac-MPI-Accelerate.cmake \
         -DEL_DISABLE_PARMETIS=ON -DEL_DISABLE_QUAD=TRUE
```

### Newer

```
rm -rf * ; cmake .. \
-DCMAKE_C_COMPILER=/opt/mpich/dev/clang/default/bin/mpicc \
-DCMAKE_CXX_COMPILER=/opt/mpich/dev/clang/default/bin/mpicxx \
-DCMAKE_C_FLAGS="-g -Wall -std=c11" \
-DCMAKE_CXX_FLAGS="-g -Wall -std=c++11" \
-DMATH_LIBS="-framework Accelerate" \
-DCMAKE_BUILD_TYPE=Release \
-DEL_DISABLE_PARMETIS=ON \
-DCMAKE_INSTALL_PREFIX=/tmp
```

### Older

Default toolchain:
```
cmake .. -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc \
-DCMAKE_Fortran_COMPILER=mpif90 -DMATH_LIBS="-framework Accelerate" \
-DELEM_EXAMPLES=ON -DELEM_TESTS=ON
```
I recently contributed a CMake toolchain file for this configuration.

Intel 15 toolchain:
```
rm -rf * ; cmake .. -DCMAKE_C_COMPILER=/opt/mpich/dev/intel/default/bin/mpicc \
-DCMAKE_CXX_COMPILER=/opt/mpich/dev/intel/default/bin/mpicxx \
-DCMAKE_Fortran_COMPILER=/opt/mpich/dev/intel/default/bin/mpif90 \
-DMPI_C_COMPILER=/opt/mpich/dev/intel/default/bin/mpicc \
-DMPI_CXX_COMPILER=/opt/mpich/dev/intel/default/bin/mpicxx \
-DMPI_Fortran_COMPILER=/opt/mpich/dev/intel/default/bin/mpif90 \
-DMATH_LIBS="-mkl" -DCMAKE_BUILD_TYPE=HybridRelease \
-DCXX_FLAGS="-O3 -qopenmp -xHOST" \
-DCMAKE_INSTALL_PREFIX=$HOME/Work/Elemental/install-intel15 && make && make install
```
Note that the Intel build of MPI must be first (from the left) in ```PATH``` otherwise CMake is stupid and refuses to abide by the wrappers (assuming they are prepended with the absolute path), hence detects the wrong libraries.

## Cray XC30

**Warning: I have not done anything with Elemental on NERSC machines in a long time, so do not assume the following works.**

[Edison](http://www.nersc.gov/users/computational-systems/edison/) is the [[Cray]] XC30 at NERSC.

Make this change in `CMakeLists.txt`:
```
cmake_minimum_required(VERSION 2.8.11)
#cmake_minimum_required(VERSION 2.8.12)
```

Then do this:
```
module unload darshan # Darshan ruins everything
#module load cmake/2.8.12.2 # see above
export CRAY_LINK_TYPE=static
rm -rf * ; cmake .. -DBUILD_SHARED_LIBS=OFF \
-DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/Edison-intel-mpich-mkl.cmake \
-DCMAKE_INSTALL_PREFIX=$HOME/ELEMENTAL/git/install-intel && make -j16
```

## Blue Gene/Q

**Warning: I have not done anything with Elemental on ALCF machines in YEARS, so do not assume the following works.**

This is how I did a site installation at ALCF:

```
#!/bin/sh +x

# This date should give you pause about how accurate the following information is...
export DATE="20121208"

cd build-gnu-netlib && rm -rf * && cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/BGQ-Vesta-gnu-netlib.cmake -DELEM_EXAMPLES=ON -DELEM_TESTS=ON -DCMAKE_INSTALL_PREFIX=/soft/libraries/unsupported/elemental/${DATE}/gcc/netlib/ && make -j16 && make install && cd ..

cd build-gnu-essl && rm -rf * && cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/BGQ-Vesta-gnu-essl.cmake -DELEM_EXAMPLES=ON -DELEM_TESTS=ON -DCMAKE_INSTALL_PREFIX=/soft/libraries/unsupported/elemental/${DATE}/gcc/essl/ && make -j16 && make install && cd ..

cd build-xl-netlib && rm -rf * && cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/BGQ-Vesta-xl-netlib.cmake -DELEM_EXAMPLES=ON -DELEM_TESTS=ON -DCMAKE_INSTALL_PREFIX=/soft/libraries/unsupported/elemental/${DATE}/xl/netlib/ && make -j16 && make install && cd ..

cd build-xl-essl && rm -rf * && cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/BGQ-Vesta-xl-essl.cmake -DELEM_EXAMPLES=ON -DELEM_TESTS=ON -DCMAKE_INSTALL_PREFIX=/soft/libraries/unsupported/elemental/${DATE}/xl/essl/ && make -j16 && make install && cd ..

cd build-clang-netlib && rm -rf * && cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/BGQ-Vesta-clang-netlib.cmake -DELEM_EXAMPLES=ON -DELEM_TESTS=ON -DCMAKE_INSTALL_PREFIX=/soft/libraries/unsupported/elemental/${DATE}/llvm/netlib/ && make -j16 && make install && cd ..

cd build-clang-essl && rm -rf * && cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/BGQ-Vesta-clang-essl.cmake -DELEM_EXAMPLES=ON -DELEM_TESTS=ON -DCMAKE_INSTALL_PREFIX=/soft/libraries/unsupported/elemental/${DATE}/llvm/essl/ && make -j16 && make install && cd ..
```

# Running Tests

```
for h in 1024 2048 4096 8192 16384 32768 ; do qsub -t 15 -n 512 --mode=c32 -O HermitianGenDefiniteEig.h$h.\$jobid --env PAMID_VERBOSE=1 ./HermitianGenDefiniteEig --eigType 1 --range A --height $h --correctness 1 --print 0 --gridHeight 512 ; done
```

```
for t in Cholesky HermitianEig HermitianGenDefiniteEig HermitianTridiag LDL LQ LU QR RQ TriangularInverse ; do for h in 8192 ; do qsub -t 15 -n 128 --mode=c32 -O $t.h$h.\$jobid --env PAMID_VERBOSE=1  ./HermitianGenDefiniteEig --eigType 1 --range A --height $h --correctness 1 --print 0 --gridHeight 64 ; done ; done
```
