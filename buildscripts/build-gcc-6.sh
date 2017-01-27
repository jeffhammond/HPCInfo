#!/bin/bash -xe
mkdir -p /tmp/gcc-6
cd /tmp/gcc-6
$HOME/Work/GCC/gcc-6.2.0/configure \
--program-suffix=-6.2 \
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
--prefix=/opt/gcc/6.2.0
make -j7
sudo make install
