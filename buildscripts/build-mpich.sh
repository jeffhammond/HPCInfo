#!/bin/bash

export MAKE_INSTALL="make -j36 install"

export VERSION=dev

export GCC_VERSION=-7

#export CXX_YES_NO=--disable-cxx
export CXX_YES_NO=--enable-cxx

export STATIC_YES_NO=--enable-static
#export STATIC_YES_NO=--disable-static

#export LIBFABRIC_PATH=/usr/local/Cellar/libfabric/1.4.1
export LIBFABRIC_PATH=/opt/libfabric

#export DEVICE=""
#export DEVICE="--with-device=ch4"
export DEVICE="--with-device=ch4:ofi --with-libfabric=${LIBFABRIC_PATH}"
#export DEVICE="--with-device=ch4:ofi:sockets --with-libfabric=/usr/local/Cellar/libfabric/1.4.1"

#export INSTALL_NAME=default
#export INSTALL_NAME=ch4
export INSTALL_NAME=ch4-ofi
#export INSTALL_NAME=ch4-ofi-sockets

# INTEL - GCC

../configure CC=icc CXX=icpc FC=ifort F77=ifort ${CXX_YES_NO} --enable-fortran --enable-threads=runtime --enable-g=dbg --with-pm=hydra --prefix=/opt/mpich/$VERSION/intel/$INSTALL_NAME --enable-wrapper-rpath --enable-two-level-namespace ${STATIC_YES_NO} --enable-shared ${DEVICE} && ${MAKE_INSTALL}

../configure CC=icc CXX=icpc FC=ifort F77=ifort ${CXX_YES_NO} --enable-fortran --enable-threads=runtime --enable-g=all --with-pm=hydra --prefix=/opt/mpich/$VERSION/intel/debug --enable-wrapper-rpath --enable-two-level-namespace ${STATIC_YES_NO} --enable-shared ${DEVICE} --enable-nemesis-dbg-localoddeven && ${MAKE_INSTALL}

../configure CC=icc CXX=icpc FC=ifort F77=ifort ${CXX_YES_NO} --enable-fortran --with-pm=hydra --prefix=/opt/mpich/$VERSION/intel/fast ${STATIC_YES_NO} --enable-fast=O3,nochkmsg,notiming,ndebug,nompit --disable-weak-symbols --enable-threads=funneled --enable-wrapper-rpath --enable-two-level-namespace ${STATIC_YES_NO} --enable-shared ${DEVICE} && ${MAKE_INSTALL}

# GCC

../configure CC=gcc${GCC_VERSION} CXX=g++${GCC_VERSION} FC=gfortran${GCC_VERSION} F77=gfortran${GCC_VERSION} ${CXX_YES_NO} --enable-fortran --enable-threads=runtime --enable-g=dbg --with-pm=hydra --prefix=/opt/mpich/$VERSION/gcc/$INSTALL_NAME --enable-wrapper-rpath --enable-two-level-namespace ${STATIC_YES_NO} --enable-shared ${DEVICE} && ${MAKE_INSTALL}

../configure CC=gcc${GCC_VERSION} CXX=g++${GCC_VERSION} FC=gfortran${GCC_VERSION} F77=gfortran${GCC_VERSION} ${CXX_YES_NO} --enable-fortran --enable-threads=runtime --enable-g=all --with-pm=hydra --prefix=/opt/mpich/$VERSION/gcc/debug --enable-wrapper-rpath --enable-two-level-namespace ${STATIC_YES_NO} --enable-shared ${DEVICE} --enable-nemesis-dbg-localoddeven && ${MAKE_INSTALL}

../configure CC=gcc${GCC_VERSION} CXX=g++${GCC_VERSION} FC=gfortran${GCC_VERSION} F77=gfortran${GCC_VERSION} ${CXX_YES_NO} --enable-fortran --with-pm=hydra --prefix=/opt/mpich/$VERSION/gcc/fast ${STATIC_YES_NO} --enable-fast=O3,nochkmsg,notiming,ndebug,nompit --disable-weak-symbols --enable-threads=funneled --enable-wrapper-rpath --enable-two-level-namespace ${STATIC_YES_NO} --enable-shared ${DEVICE} && ${MAKE_INSTALL}

exit

# LLVM

../configure CC=clang CXX=clang++ FC=false F77=false ${CXX_YES_NO} --disable-fortran --with-pm=hydra --prefix=/opt/mpich/$VERSION/clang/$INSTALL_NAME ${CXX_YES_NO} --enable-wrapper-rpath --enable-two-level-namespace ${STATIC_YES_NO} --enable-shared ${DEVICE} && ${MAKE_INSTALL}

../configure CC=clang CXX=clang++ FC=false F77=false ${CXX_YES_NO} --disable-fortran --with-pm=hydra --prefix=/opt/mpich/$VERSION/clang/debug ${CXX_YES_NO} --enable-threads=runtime --enable-g=all --enable-wrapper-rpath  --enable-two-level-namespace ${STATIC_YES_NO} --enable-shared ${DEVICE} --enable-nemesis-dbg-localoddeven && ${MAKE_INSTALL}

../configure CC=clang CXX=clang++ FC=false F77=false ${CXX_YES_NO} --disable-fortran --with-pm=hydra --prefix=/opt/mpich/$VERSION/clang/fast ${CXX_YES_NO} --enable-fast=O3,nochkmsg,notiming,ndebug,nompit --disable-weak-symbols --enable-threads=funneled --enable-wrapper-rpath --enable-two-level-namespace ${STATIC_YES_NO} --enable-shared ${DEVICE} && ${MAKE_INSTALL}

