#!/bin/bash -ex

export MAKE_JNUM="make -j8"

#export GCC_VERSION=-7
#export GCC_PREFIX=/opt/gcc/latest

export GIT_HOST=https://github.com/llvm-mirror

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
    git clone $GIT_HOST/llvm.git git
    cd git
    git config branch.master.rebase true
fi

WHAT=$LLVM_TOP/git/tools/clang
if [ -d $WHAT ] ; then
    cd $WHAT
    git pull
else
    cd $LLVM_TOP/git/tools
    git clone $GIT_HOST/clang.git
fi

WHAT=$LLVM_TOP/git/tools/polly
if [ -d $WHAT ] ; then
    cd $WHAT
    git pull
else
    cd $LLVM_TOP/git/tools
    git clone $GIT_HOST/polly.git
fi

WHAT=$LLVM_TOP/git/projects/compiler-rt
if [ -d $WHAT ] ; then
    cd $WHAT
    git pull
else
    cd $LLVM_TOP/git/projects
    git clone $GIT_HOST/compiler-rt.git
fi

WHAT=$LLVM_TOP/git/projects/openmp
if [ -d $WHAT ] ; then
    cd $WHAT
    git pull
else
    cd $LLVM_TOP/git/projects
    git clone $GIT_HOST/openmp.git
fi

WHAT=$LLVM_TOP/git/projects/libcxx
if [ -d $WHAT ] ; then
    cd $WHAT
    git pull
else
    cd $LLVM_TOP/git/projects
    git clone $GIT_HOST/libcxx.git
fi

WHAT=$LLVM_TOP/git/projects/libcxxabi
if [ -d $WHAT ] ; then
    cd $WHAT
    git pull
else
    cd $LLVM_TOP/git/projects
    git clone $GIT_HOST/libcxxabi.git
fi

WHAT=$LLVM_TOP/git/projects/lld
if [ -d $WHAT ] ; then
    cd $WHAT
    git pull
else
    cd $LLVM_TOP/git/projects
    git clone $GIT_HOST/lld.git
fi


WHAT=$LLVM_TOP/git/projects/test-suite
if [ -d $WHAT ] ; then
    cd $WHAT
    git pull
else
    cd $LLVM_TOP/git/projects
    git clone $GIT_HOST/test-suite.git
fi

rm -rf $LLVM_TMP
mkdir -p $LLVM_TMP
cd $LLVM_TMP
cmake $LLVM_TOP/git  -G "Unix Makefiles" \
    -DCMAKE_C_COMPILER=/opt/llvm/latest/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/llvm/latest/bin/clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_LLD=YES \
    -DLLVM_ENABLE_CXX1Y=YES \
    -DLLVM_ENABLE_LIBCXX=YES \
    -DPYTHON_EXECUTABLE=`which python` \
    -DCMAKE_INSTALL_PREFIX=$LLVM_TOP/HEAD
    #-DLLVM_TARGETS_TO_BUILD=X86 \
    #-DLLVM_ENABLE_LTO=Full \
    #-DLLVM_EXTERNAL_PROJECTS=clang,lld,polly \
    #-DLLVM_EXTERNAL_CLANG_SOURCE_DIR=$LLVM_TOP/git/tools/clang \
    #-DLLVM_EXTERNAL_POLLY_SOURCE_DIR=$LLVM_TOP/git/polly/polly \
    #-DLLVM_EXTERNAL_LLD_SOURCE_DIR=$LLVM_TOP/git/projects/lld \
    #-DGCC_INSTALL_PREFIX=$GCC_PREFIX \
    #-DCMAKE_C_COMPILER=gcc$GCC_VERSION \
    #-DCMAKE_CXX_COMPILER=g++$GCC_VERSION \

${MAKE_JNUM} && ${MAKE_JNUM} install ; ${MAKE_JNUM} check
