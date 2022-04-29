#!/bin/bash

set -ex

OMPI_SRC=$HOME/MPI/ompi

pushd $OMPI_SRC ; git fetch --all ; git remote prune origin ; git remote set-head origin --auto ; git gc

#VERSIONS=`pushd $OMPI_SRC >& /dev/null ; git tag | sort -k1r | grep -v "v[1234]" ; popd >& /dev/null`
#VERSIONS=main
VERSIONS=$1

#echo $VERSIONS

for VER in $VERSIONS ; do

    # source
    pushd $OMPI_SRC
    git clean -dfx
    git reset --hard
    git checkout .
    git checkout $VER
    ./autogen.pl
    popd

    # build MPI
    pushd /tmp
    #rm -rf build-ompi-$VER install-ompi-$VER armci-mpi-ompi-$VER
    mkdir -p build-ompi-$VER install-ompi-$VER
    pushd build-ompi-$VER
    $OMPI_SRC/configure CC="$2" CFLAGS="$3" --prefix=/tmp/install-ompi-$VER --without-psm2 --without-libfabric --without-ofi --without-cuda --enable-mpi-fortran=none #--enable-static --disable-shared
    make -j`nproc` install
    popd

    # test ARMCI-MPI
    export V=1
    export VERBOSE=1
    rm -rf /tmp/armci-mpi-ompi-$VER
    git clone --depth 1 https://github.com/pmodels/armci-mpi.git /tmp/armci-mpi-ompi-$VER
    pushd /tmp/armci-mpi-ompi-$VER
    ./autogen.sh
    ./configure CC=/tmp/install-ompi-$VER/bin/mpicc --enable-g CFLAGS="$3"
    make -j`nproc` checkprogs
    #MPIRUN=/tmp/install-ompi-$VER/bin/mpirun make check
    /tmp/install-ompi-$VER/bin/mpirun -n 1 ./tests/mpi/test_win_model
    /tmp/install-ompi-$VER/bin/mpirun -n 1 gdb ./tests/mpi/test_win_model -ex "set width 1000" -ex "thread apply all bt" -ex run -ex bt -ex "set confirm off" -ex quit
    popd

done

