#!/bin/bash

set -ex

OMPI_SRC=$HOME/MPI/ompi

pushd $OMPI_SRC ; git fetch --all ; git remote prune origin ; git remote set-head origin --auto ; git gc

VERSIONS=`pushd $OMPI_SRC >& /dev/null ; git tag | sort -k1r | grep -v "v[125]" ; popd >& /dev/null`
#VERSIONS=main

#echo $VERSIONS

for VER in $VERSIONS ; do

    # source
    pushd $OMPI_SRC
    #git clean -dfx
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
    $OMPI_SRC/configure --prefix=/tmp/install-ompi-$VER --without-psm2 --without-cuda --without-ofi --without-libfabric
    make -j`nproc` install
    popd

    # test ARMCI-MPI
    export V=1
    export VERBOSE=1
    git clone --depth 1 https://github.com/pmodels/armci-mpi.git /tmp/armci-mpi-ompi-$VER
    pushd /tmp/armci-mpi-ompi-$VER
    ./autogen.sh
    ./configure CC=/tmp/install-ompi-$VER/bin/mpicc --enable-g
    make -j`nproc` checkprogs
    MPIRUN=/tmp/install-ompi-$VER/bin/mpirun make check
    popd

done

