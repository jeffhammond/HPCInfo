#!/bin/bash -xe

# Persistent location for source and binaries
GCC_HOME=$HOME/GCC
mkdir -p $GCC_HOME

# Download/update the source
cd $GCC_HOME
if [ -d $GCC_HOME/git ] ; then
  cd $GCC_HOME/git
  git pull
else
  git clone --depth 10 https://github.com/gcc-mirror/gcc.git git
fi

# Download the dependencies
# If wget fails, curl may be better (needs manual edit)
cd $GCC_HOME/git
./contrib/download_prerequisites

# Create the temporary build directory in a fast local filesystem
mkdir -p /tmp/gcc-build
cd /tmp/gcc-build
$GCC_HOME/git/configure \
--program-suffix=-7 \
--disable-multilib \
--enable-threads=posix \
--enable-checking=release \
--with-system-zlib \
--enable-__cxa_atexit \
--enable-languages=c,c++,fortran \
--with-tune=native \
--enable-bootstrap \
--enable-lto \
--enable-gold=yes \
--enable-ld=yes \
--prefix=$GCC_HOME/HEAD
make -j15
make install || sudo make install
