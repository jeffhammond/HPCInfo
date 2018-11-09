#!/bin/bash -ex

export MAKE_JNUM="make -j1"

# where LLVM source and install will live
export LLVM_TOP=/opt/llvm/pgi-flang

mkdir -p $LLVM_TOP

WHAT=$LLVM_TOP/llvm-git
if [ -d $WHAT ] ; then
    cd $WHAT
    git fetch --all
    git checkout release_60
    git pull
else
    cd $LLVM_TOP
    git clone -b release_60 https://github.com/llvm-mirror/llvm.git llvm-git
fi
cd $WHAT
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$LLVM_TOP && $MAKE_JNUM install

WHAT=$LLVM_TOP/flang-driver-git
if [ -d $WHAT ] ; then
    cd $WHAT
    git fetch --all
    git checkout release_60
    git pull
else
    cd $LLVM_TOP
    git clone -b release_60 https://github.com/flang-compiler/flang-driver.git flang-driver-git
fi
cd $WHAT
mkdir -p build && cd build
cmake .. -DLLVM_CONFIG=$LLVM_TOP/bin/llvm-config \
         -DCMAKE_INSTALL_PREFIX=$LLVM_TOP && $MAKE_JNUM install

WHAT=$LLVM_TOP/openmp-git
if [ -d $WHAT ] ; then
    cd $WHAT
    git fetch --all
    git checkout release_60
    git pull
else
    cd $LLVM_TOP
    git clone -b release_60 https://github.com/llvm-mirror/openmp.git openmp-git
fi
cd $WHAT
mkdir -p build && cd build
cmake .. \
         -DCMAKE_INSTALL_PREFIX=$LLVM_TOP && $MAKE_JNUM install

WHAT=$LLVM_TOP/flang-git
if [ -d $WHAT ] ; then
    cd $WHAT
    git fetch --all
    git pull
else
    cd $LLVM_TOP
    git clone https://github.com/flang-compiler/flang.git flang-git
fi
cd $WHAT
mkdir -p build && cd build
cmake .. -DLLVM_CONFIG=$LLVM_TOP/bin/llvm-config \
         -DCMAKE_INSTALL_PREFIX=$LLVM_TOP \
         -DCMAKE_C_COMPILER=$LLVM_TOP/bin/clang \
         -DCMAKE_CXX_COMPILER=$LLVM_TOP/bin/clang++ \
         -DCMAKE_Fortran_COMPILER=$LLVM_TOP/bin/flang && $MAKE_JNUM install

cd $WHAT/runtime/libpgmath
mkdir -p build && cd build
cmake .. -DLLVM_CONFIG=$LLVM_TOP/bin/llvm-config \
         -DCMAKE_INSTALL_PREFIX=$LLVM_TOP \
         -DCMAKE_C_COMPILER=$LLVM_TOP/bin/clang \
         -DCMAKE_CXX_COMPILER=$LLVM_TOP/bin/clang++ \
         -DCMAKE_Fortran_COMPILER=$LLVM_TOP/bin/flang && $MAKE_JNUM install

