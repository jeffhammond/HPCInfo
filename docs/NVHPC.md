# Notes for building things with NVHPC

[Home Page](https://developer.nvidia.com/hpc-sdk)

[Download](https://developer.nvidia.com/nvidia-hpc-sdk-downloads)

# Open-MPI 4.0.x

```sh
../configure CC="nvc -nomp" CXX="nvc++ -nomp" FC="nvfortran -nomp" \
             CFLAGS="-O1 -fPIC -c99" CXXFLAGS="-O1 -fPIC" FCFLAGS="-O1 -fPIC" \
             LD="ld" --enable-shared --enable-static \
             --without-tm --with-libevent=internal --without-libnl \
             --enable-mpirun-prefix-by-default --disable-wrapper-runpath \
             --prefix=${HOME}/MPI/ompi-nvhpc-arm
```
