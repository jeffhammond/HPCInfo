#!/bin/bash -xe

if [ `hostname` == "xavier-agx" ] || [ `hostname` == "orin" ]; then
    CC=/samsung/GCC/11.2.0/bin/gcc-11
    CXX=/samsung/GCC/11.2.0/bin/g++-11
    LLVM_HOME=/samsung/LLVM
    LLVM_TEMP=/samsung/LLVM/build
elif [ `uname -s` == Darwin ] ; then
    # can't use GCC now (https://github.com/llvm/llvm-project/issues/34628)
    #CC=gcc-12
    #CXX=g++-12
    CC=clang
    CXX=clang++
    LLVM_HOME=/opt/llvm
    LLVM_TEMP=/tmp/llvm-build
else
    CC=gcc-12
    CXX=g++-12
    if [ `hostname` == "nuclear" ] || [ `hostname` == "oppenheimer" ]; then
        LLVM_HOME=/opt/llvm
    elif [ `hostname` == "fi-kermit" ] || [ `hostname` == "gorby" ]; then
        #LLVM_HOME=/home/${USER}/LLVM
        LLVM_HOME=/opt/llvm
    else
        LLVM_HOME=/local/home/${USER}/LLVM
    fi
    LLVM_TEMP=/tmp/$USER/build
fi
mkdir -p $LLVM_HOME

#rm -rf $LLVM_TEMP
mkdir -p $LLVM_TEMP

REPO=https://github.com/llvm/llvm-project.git
#REPO=https://github.com/llvm/llvm-project.git
#REPO=https://github.com/flang-compiler/f18-llvm-project.git # OBSOLETE
#REPO=https://github.com/Sezoir/f18-llvm-project.git # SPECIAL
#REPO=https://github.com/kiranchandramohan/llvm-project.git # SPECIAL
#BRANCH=w-option

# Download/update the source
cd $LLVM_HOME
if [ -d $LLVM_HOME/git ] ; then
  cd $LLVM_HOME/git
  git remote remove origin
  git remote add origin $REPO
  git fetch origin
  # SPECIAL
  #git checkout $BRANCH || echo exists
  #git branch --set-upstream-to=origin/$BRANCH $BRANCH || echo dunno
  # NORMAL
  git checkout origin/main -b main || git checkout main
  git branch --set-upstream-to=origin/main main || echo dunno
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

if [ `uname -m` == arm64 ] || [ `uname -m` == aarch64 ] ; then
    MYARCH=AArch64
else
    MYARCH=X86
fi

########################################################
# throttle the build so the machine remains responsive #
########################################################

EXTRA_CMAKE_ARGS=''
if [ `uname -s` == Darwin ] ; then
    NUM_HWTHREADS=`sysctl -n hw.ncpu`

    MEMORY_BYTES=`sysctl -n hw.memsize`
    MEMORY_GIGS=$(( $MEMORY_BYTES / 1000000000 ))

    MEMORY_COMPILE_LIMIT=$(( $MEMORY_GIGS / 4 ))
    MEMORY_LINK_LIMIT=$(( $MEMORY_GIGS / 12 ))

    NUM_COMPILE=$MEMORY_COMPILE_LIMIT
    NUM_LINK=$MEMORY_LINK_LIMIT

    MACOS_SYSROOT="-DDEFAULT_SYSROOT=$(xcrun --show-sdk-path)"
else
    NUM_HWTHREADS=`nproc`

    MEMORY_KILOS=`grep MemTotal /proc/meminfo | awk '{print $2}'`
    MEMORY_GIGS=$(( $MEMORY_KILOS / 1000000 ))

    MEMORY_COMPILE_LIMIT=$(( $MEMORY_GIGS / 4 ))
    MEMORY_LINK_LIMIT=$(( $MEMORY_GIGS / 12 ))

    NUM_COMPILE=$MEMORY_COMPILE_LIMIT
    NUM_LINK=$MEMORY_LINK_LIMIT

    USE_GOLD="-DLLVM_USE_LINKER=gold"
fi

cd $LLVM_TEMP || exit
rm -f ${LLVM_TEMP}/CMakeCache.txt

# lldb busted on MacOS
# libcxx requires libcxxabi
cmake \
      -G "$BUILDTOOL" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_PARALLEL_LINK_JOBS=$NUM_LINK \
      -DLLVM_PARALLEL_COMPILE_JOBS=$NUM_COMPILE \
      -DLLVM_TARGETS_TO_BUILD=$MYARCH \
      -DLLVM_ENABLE_RUNTIMES="libcxxabi;libcxx;libunwind" \
      -DLLVM_ENABLE_PROJECTS="lld;mlir;clang;flang;openmp;pstl;polly" \
      -DFLANG_RUNTIME_F128_MATH_LIB=libquadmath \
      -DPYTHON_EXECUTABLE=`which python` \
      -DCMAKE_C_COMPILER=$CC \
      -DCMAKE_CXX_COMPILER=$CXX \
      -DLLVM_LIT_ARGS=-v \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_INSTALL_PREFIX=$LLVM_HOME/latest \
      $USE_GOLD \
      $MACOS_SYSROOT \
      $LLVM_HOME/git/llvm


cmake --build .
cmake --install .

