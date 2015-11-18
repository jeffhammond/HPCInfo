OFI
===

See the [home page](http://ofiwg.github.io/libfabric/) for details.

# Git repositories

## libfabric (OFI implementation)

* [Main libfabric](https://github.com/ofiwg/libfabric)
* [Cray libfabric](https://github.com/ofi-cray/libfabric-cray)* 

## fabtests (OFI test suite)

* [Main fabtests](https://github.com/ofiwg/fabtests)
* [Cray fabtests](https://github.com/ofi-cray/fabtests-cray)

## MPI implementations that use OFI

* [MPICH](http://git.mpich.org/mpich.git/)
* [Open-MPI](https://github.com/open-mpi/ompi.git)
* Intel MPI ([Documentation](https://software.intel.com/en-us/node/561773))

## Other OFI clients

* [OpenSHMEM tutorial](https://github.com/ofiwg/openshmem-tutorial)
* [Sandia SHMEM](https://github.com/regrant/sandia-shmem)
* [GASNet](https://bitbucket.org/berkeleylab/gasnet)

# Building stuff

## Generic generic (sockets provider)


## Intel networks (i.e. PSM and PSM2 providers)


## Cray XC systems (uGNI provider for Aries)

Use Cray's libfabric until it is merged upstream.

### libfabric

The uGNI provider uses C11 atomics, so you must `module load gcc` to get a more recent version than the GCC that comes with the system.

```sh
../configure CC=gcc \
             --disable-sockets --enable-gni \
             --enable-static --disable-shared \
             --prefix=$HOME/OFI/install-ofi-gcc-gni-edison 
```

### Criterion

#### Right

You must use version 1.2.2.  This version uses autotools instead of CMake.   However, to run `./autogen.sh`, you need to install a more recent version of `gettext` (e.g. `0.19.6`).  Then you build as follows.

```
../configure CC=gcc --prefix=$HOME/OFI/install-Criterion-v1.2.2 && make -j20 && make install
```

`make check` fails, but it appears to be a build system problem, which is not worth reporting due to the age of this version of Criterion.

#### Wrong

*Do not use the latest version!*

This is required for unit testing.  See [this](https://github.com/ofi-cray/libfabric-cray/wiki/Building-and-running-the-unit-tests-(gnitest)) for details.

One must disable internationalization because of a problem with `msgmerge --lang=fr`.  See [this](https://github.com/Snaipe/Criterion/issues/77) for details.

```
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
         -DCMAKE_INSTALL_PREFIX:PATH=$HOME/OFI/install-Criterion -DI18N=OFF
```

### fabtests

```sh
../configure CC=gcc \
             --with-libfabric=$HOME/OFI/install-ofi-gcc-gni-edison \
             --with-pmi=/opt/cray/pmi/default \
             --prefix=$HOME/OFI/install-fabtest-gcc-gni-cori \
             --with-criterion=/global/homes/j/jhammond/OFI/install-Criterion-v1.2.2 \
             LDFLAGS="-L/opt/cray/ugni/default/lib64 -lugni \
                      -L/opt/cray/alps/default/lib64 -lalps -lalpslli -lalpsutil \
                      -ldl -lrt"
```

### MPICH

_This is very much a work in progress_

### Open-MPI

TODO

### Sandia SHMEM

TODO
