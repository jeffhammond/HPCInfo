#!/bin/bash -ex

export GCC_VERSION=-7

# where LLVM source and install will live
export LLVM_TOP=/opt/llvm

# where LLVM is compiled
export LLVM_TMP=/tmp/$USER-llvm-build

mkdir -p $LLVM_TOP

WHAT=$LLVM_TOP/git
if [ -d $WHAT ] ; then
    cd $WHAT
    git pull
else
    cd $LLVM_TOP
    git clone http://llvm.org/git/llvm.git git
    cd git
    git config branch.master.rebase true
fi

WHAT=$LLVM_TOP/git/tools/clang
if [ -d $WHAT ] ; then
    cd $WHAT
    git pull
else
    cd $WHAT/..
    git clone http://llvm.org/git/clang.git
fi

WHAT=$LLVM_TOP/git/projects/compiler-rt
if [ -d $WHAT ] ; then
    cd $WHAT
    git pull
else
    cd $WHAT/..
    git clone http://llvm.org/git/compiler-rt.git
fi

WHAT=$LLVM_TOP/git/projects/openmp
if [ -d $WHAT ] ; then
    cd $WHAT
    git pull
else
    cd $WHAT/..
    git clone http://llvm.org/git/openmp.git
fi

WHAT=$LLVM_TOP/git/projects/libcxx
if [ -d $WHAT ] ; then
    cd $WHAT
    git pull
else
    cd $WHAT/..
    git clone http://llvm.org/git/libcxx.git
fi

WHAT=$LLVM_TOP/git/projects/libcxxabi
if [ -d $WHAT ] ; then
    cd $WHAT
    git pull
else
    cd $WHAT/..
    git clone http://llvm.org/git/libcxxabi.git
fi

WHAT=$LLVM_TOP/git/projects/test-suite
if [ -d $WHAT ] ; then
    cd $WHAT
    git pull
else
    cd $WHAT/..
    git clone http://llvm.org/git/test-suite.git
fi

rm -rf $LLVM_TMP
mkdir -p $LLVM_TMP
cd $LLVM_TMP
cmake $LLVM_TOP/git  -G "Unix Makefiles" \
    -DCMAKE_INSTALL_PREFIX=$LLVM_TOP/HEAD \
    -DCMAKE_C_COMPILER=gcc$GCC_VERSION \
    -DCMAKE_CXX_COMPILER=g++$GCC_VERSION \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=X86 \
    -DPYTHON_EXECUTABLE=`which python`
make -j
make -j install
