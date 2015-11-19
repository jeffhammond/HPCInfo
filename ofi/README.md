OFI
===

See the [home page](http://ofiwg.github.io/libfabric/) for details.

# Git repositories

## libfabric (OFI implementation)

* [Main libfabric](https://github.com/ofiwg/libfabric)
* [Cray libfabric](https://github.com/ofi-cray/libfabric-cray)

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

### Criterion

This is required for unit testing.  See [this](https://github.com/ofi-cray/libfabric-cray/wiki/Building-and-running-the-unit-tests-(gnitest)) for details.

#### Right

You must use version 1.2.2.  This version uses autotools instead of CMake.   However, to run `./autogen.sh`, you need to install a more recent version of `gettext` (e.g. `0.19.6`).  Then you build as follows.

```
../configure CC=gcc --prefix=$HOME/OFI/install-Criterion-v1.2.2 && make -j20 && make install
```

`make check` fails, but it appears to be a build system problem, which is not worth reporting due to the age of this version of Criterion.

#### Wrong

*Do not use the latest version!*

One must disable internationalization because of a problem with `msgmerge --lang=fr`.  See [this](https://github.com/Snaipe/Criterion/issues/77) for details.

```
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
         -DCMAKE_INSTALL_PREFIX:PATH=$HOME/OFI/install-Criterion -DI18N=OFF
```

### libfabric

The uGNI provider uses C11 atomics, so you must `module load gcc` to get a more recent version than the GCC that comes with the system.

```sh
../configure CC=gcc \
             --disable-sockets --enable-gni \
             --enable-static --disable-shared \
             --with-criterion=$HOME/OFI/install-Criterion-v1.2.2 \
             --prefix=$HOME/OFI/install-ofi-gcc-gni-edison \
             LDFLAGS="-L/opt/cray/ugni/default/lib64 -lugni \
                      -L/opt/cray/alps/default/lib64 -lalps -lalpslli -lalpsutil \
                      -ldl -lrt"
```

Assuming all goes well until this point, you can run the unit tests like this:
```sh
[----] Warning! This test crashed during its setup or teardown.
[====] Synthesis: Tested: 316 | Passing: 316 | Failing: 0 | Crashing: 0 

real     3m0.046s
user     1m10.940s
sys      0m40.079s
```
Note that this test runs for a while and provides not incremental status update, so be patient.

(Grab a compute node with e.g. `salloc -N 1 -p shared -t 00:30:00`)

### fabtests

_This is not working for me yet._

```sh
../configure CC=gcc \
             --with-libfabric=$HOME/OFI/install-ofi-gcc-gni-edison \
             --with-pmi=/opt/cray/pmi/default \
             --prefix=$HOME/OFI/install-fabtest-gcc-gni-cori \
             LDFLAGS="-L/opt/cray/ugni/default/lib64 -lugni \
                      -L/opt/cray/alps/default/lib64 -lalps -lalpslli -lalpsutil \
                      -ldl -lrt"
```

### MPICH

_This is very much a work in progress_

```sh
../configure CC=gcc CXX=g++ FC=gfortran F77=gfortran \
             --with-ofi=$HOME/OFI/install-ofi-gcc-gni-cori \
             --with-device=ch3:nemesis:ofi \
             --disable-shared \
             --prefix=$HOME/MPI/install-mpich-gcc-ofi-gni-cori \
             LDFLAGS="-L/opt/cray/ugni/default/lib64 -lugni \
                      -L/opt/cray/alps/default/lib64 -lalps -lalpslli -lalpsutil \
                      -ldl -lrt"
```

### Open-MPI

See [Cray's docs](https://github.com/ofi-cray/libfabric-cray/wiki/Building-and-Running-OpenMPI).

```sh
../configure \
             --with-libfabric=$HOME/OFI/install-ofi-gcc-gni-cori \
             --disable-shared \
             --prefix=$HOME/MPI/install-ompi-ofi-gcc-gni-cori
```

Unfortunately, this leads to an `mpicc` that indicates support for IB Verbs, not OFI.

```sh
../configure --with-libfabric=$HOME/OFI/install-ofi-gcc-gni-cori \
             --enable-mca-static=mtl-ofi \
             --enable-mca-no-build=btl-openib,btl-vader,btl-ugni,btl-tcp \
             --enable-static --disable-shared --disable-dlopen \
             --prefix=$HOME/MPI/install-ompi-ofi-gcc-gni-xpmem-cori \
             --with-cray-pmi --with-alps --with-cray-xpmem --with-slurm \
             --without-verbs --without-fca --without-mxm --without-ucx \
             --without-portals4 --without-psm --without-psm2 \
             --without-udreg --without-ugni --without-munge \
             --without-sge --without-loadleveler --without-tm --without-lsf \
             --without-pvfs2 --without-plfs \
             --without-cuda --disable-oshmem \
             --disable-mpi-fortran --disable-oshmem-fortran \
             LDFLAGS="-L/opt/cray/ugni/default/lib64 -lugni \
                      -L/opt/cray/alps/default/lib64 -lalps -lalpslli -lalpsutil \
                      -ldl -lrt"
```

### Sandia SHMEM

Because of https://github.com/regrant/sandia-shmem/issues/49, one has to do a disgusting in-place build :-(

You need to `module load PrgEnv-intel` if you use `cc` and `ftn` used instead of `icc` and `ifort` (same for GNU).  If you use the Cray compiler wrappers, you do not need `LDFLAGS` below.

```sh
./configure CC=cc FC=ftn \
            --with-pmi=/opt/cray/pmi/default \
            --with-xpmem=/opt/cray/xpmem/default  \
            --with-ofi=$HOME/OFI/install-ofi-gcc-gni-cori \
            --enable-remote-virtual-addressing \
            --enable-static --disable-shared \
            --disable-fortran \
            --prefix=$HOME/SHMEM/install-sandia-shmem-ofi-xpmem-icc \
            LDFLAGS="-L/opt/cray/ugni/default/lib64 -lugni \
                     -L/opt/cray/alps/default/lib64 -lalps -lalpslli -lalpsutil \
                     -ldl -lrt"
```
