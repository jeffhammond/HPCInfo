#!/bin/bash -xe
mkdir -p /tmp/gcc-5
cd /tmp/gcc-5
$HOME/Work/GCC/gcc-5.4.0/configure \
--program-suffix=-5.4 \
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
--prefix=/opt/gcc/5.4.0
make -j7
sudo make install
