# Overview

[LLVM](http://llvm.org/) is an open-source compiler project and is an alternative to [[GCC]] for C/C++.  [Clang](http://clang.llvm.org/) is the C/C++ front-end for LLVM.

# Documentation

* http://clang.llvm.org/docs/UsersManual.html

# Notes

* http://llvm.org/releases/3.3/docs/GettingStarted.html#compiling-the-llvm-suite-source-code
* http://polly.llvm.org/get_started.html
* http://lldb.llvm.org/build.html

# Building

```sh
#!/bin/bash -ex

export LLVM_TOP=$HOME/LLVM

mkdir -p $LLVM_TOP

cd $LLVM_TOP
git clone http://llvm.org/git/llvm.git
cd $LLVM_TOP/llvm
git config branch.master.rebase true

cd $LLVM_TOP/llvm/tools
git clone http://llvm.org/git/clang.git

cd $LLVM_TOP/llvm/projects
git clone http://llvm.org/git/compiler-rt.git

cd $LLVM_TOP/llvm/projects
git clone http://llvm.org/git/openmp.git

cd $LLVM_TOP/llvm/projects
git clone http://llvm.org/git/libcxx.git
git clone http://llvm.org/git/libcxxabi.git

cd $LLVM_TOP/llvm/projects
git clone http://llvm.org/git/test-suite.git

cd $LLVM_TOP/llvm
mkdir -p $LLVM_TOP/llvm/build
cd $LLVM_TOP/llvm/build
cmake ..  -G "Unix Makefiles" \
    -DCMAKE_INSTALL_PREFIX=$HOME/LLVM/install \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=X86
make -j12
make -j12 install
```