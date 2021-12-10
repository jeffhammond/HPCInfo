#!/bin/bash -xe

MAKE_JNUM="-j`nproc`"

LLVM_HOME=/opt/llvm
mkdir -p $LLVM_HOME

LLVM_TEMP=/tmp/llvm-build
mkdir -p $LLVM_TEMP

# Download/update the source
cd $LLVM_HOME
if [ -d $LLVM_HOME/git ] ; then
  cd $LLVM_HOME/git
  git pull
else
  git clone https://github.com/llvm/llvm-project.git git
fi

cd $LLVM_TEMP

cmake \
      -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_RUNTIMES="all" \
      -DLLVM_ENABLE_PROJECTS="lld;lldb;polly;mlir;clang;clang-tools-extra;compiler-rt;libc;libclc;openmp;flang;pstl" \
      -DPYTHON_EXECUTABLE=`which python` \
      -DCMAKE_C_COMPILER=`which gcc` \
      -DCMAKE_CXX_COMPILER=`which g++` \
      -DLLVM_USE_LINKER=gold \
      $LLVM_HOME

make $MAKE_JNUM

