#!/bin/bash -xe
cd $HOME/Work/GCC
if [ -d $HOME/Work/GCC/git ] ; then
  cd $HOME/Work/GCC/git
  git pull
else
  git clone --depth 10 https://github.com/gcc-mirror/gcc.git git
fi
mkdir -p /tmp/gcc-build
cd /tmp/gcc-build
$HOME/Work/GCC/git/configure \
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
--prefix=/opt/gcc/master
make -j15
sudo make install
