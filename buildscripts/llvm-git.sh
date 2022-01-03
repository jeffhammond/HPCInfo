#!/bin/bash -xe

if [ `uname -s` == Darwin ] ; then
    NUM_HWTHREADS=`sysctl -n hw.ncpu`

    MEMORY_BYTES=`sysctl -n hw.memsize`
    MEMORY_GIGS=$(( $MEMORY_BYTES / 1000000000 ))

    MEMORY_COMPILE_LIMIT=$(( $MEMORY_GIGS / 4 ))
    MEMORY_LINK_LIMIT=$(( $MEMORY_GIGS / 12 ))

    NUM_COMPILE=$MEMORY_COMPILE_LIMIT
    NUM_LINK=$MEMORY_LINK_LIMIT
else
    NUM_HWTHREADS=`nproc`

    MEMORY_KILOS=`grep MemTotal /proc/meminfo | awk '{print $2}'`
    MEMORY_GIGS=$(( $MEMORY_KILOS / 1000000 ))

    MEMORY_COMPILE_LIMIT=$(( $MEMORY_GIGS / 4 ))
    MEMORY_LINK_LIMIT=$(( $MEMORY_GIGS / 12 ))

    NUM_COMPILE=$MEMORY_COMPILE_LIMIT
    NUM_LINK=$MEMORY_LINK_LIMIT
fi

if [ `uname -m` == arm64 ] || [ `uname -m` == aarch64 ] ; then
    MYARCH=AArch64
else
    MYARCH=X86
fi

CC=gcc-11
CXX=g++-11

LLVM_HOME=/local/home/jehammond/LLVM
mkdir -p $LLVM_HOME

LLVM_TEMP=/tmp/llvm-build

#REPO=https://github.com/llvm/llvm-project.git
REPO=https://github.com/flang-compiler/f18-llvm-project.git
#REPO=https://github.com/Sezoir/f18-llvm-project.git # SPECIAL

# Download/update the source
cd $LLVM_HOME
if [ -d $LLVM_HOME/git ] ; then
  cd $LLVM_HOME/git
  git remote remove origin
  git remote add origin $REPO
  git fetch origin
  git checkout fir-dev
  git pull
  git submodule update --init --recursive
else
  git clone --recursive $REPO $LLVM_HOME/git
fi

if [ `which ninja` ] ; then
    BUILDTOOL="Ninja"
else
    BUILDTOOL="Unix Makefiles"
fi

#rm -rf $LLVM_TEMP
#mkdir -p $LLVM_TEMP
cd $LLVM_TEMP

# lldb busted on MacOS
# libcxx requires libcxxabi
cmake \
      -G $BUILDTOOL \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_PARALLEL_LINK_JOBS=$NUM_LINK \
      -DLLVM_PARALLEL_COMPILE_JOBS=$NUM_COMPILE \
      -DLLVM_TARGETS_TO_BUILD=$MYARCH \
      -DLLVM_ENABLE_RUNTIMES="libcxxabi;libcxx" \
      -DLLVM_ENABLE_PROJECTS="lld;mlir;clang;flang;openmp;pstl;polly" \
      -DPYTHON_EXECUTABLE=`which python` \
      -DCMAKE_C_COMPILER=$CC \
      -DCMAKE_CXX_COMPILER=$CXX \
      -DLLVM_USE_LINKER=gold \
      -DCMAKE_INSTALL_PREFIX=$LLVM_HOME/latest \
      $LLVM_HOME/git/llvm

cmake --build . 

