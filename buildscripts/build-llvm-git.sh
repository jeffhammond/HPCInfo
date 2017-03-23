#!/bin/bash -ex

# where LLVM source and install will live
export LLVM_TOP=$HOME/LLVM

# where LLVM is compiled
export LLVM_TMP=/tmp/$USER-llvm-build

mkdir -p $LLVM_TOP
mkdir -p $LLVM_TMP

cd $LLVM_TOP
git clone http://llvm.org/git/llvm.git git
cd $LLVM_TOP/git
git config branch.master.rebase true

cd $LLVM_TOP/git/tools
git clone http://llvm.org/git/clang.git

cd $LLVM_TOP/git/projects
git clone http://llvm.org/git/compiler-rt.git

cd $LLVM_TOP/git/projects
git clone http://llvm.org/git/openmp.git

cd $LLVM_TOP/git/projects
git clone http://llvm.org/git/libcxx.git
git clone http://llvm.org/git/libcxxabi.git

cd $LLVM_TOP/git/projects
git clone http://llvm.org/git/test-suite.git

cd $LLVM_TMP
cmake $LLVM_TOP/git  -G "Unix Makefiles" \
    -DCMAKE_INSTALL_PREFIX=$LLVM_TOP/HEAD \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=X86
make -j
make -j install
