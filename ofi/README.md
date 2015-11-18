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

### OFI

The uGNI provider uses C11 atomics, so you must `module load gcc` to get a more recent version than the GCC that comes with the system.

```sh
../configure CC=gcc --disable-sockets --enable-gni --enable-static --disable-shared \
                    --prefix=$HOME/OFI/install-ofi-gcc-gni-edison 
```

### MPICH

_This is very much a work in progress_

### Open-MPI

TODO

### Sandia SHMEM

TODO
