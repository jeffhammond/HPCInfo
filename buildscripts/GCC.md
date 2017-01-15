# Overview

See http://gcc.gnu.org/

# Compiling from Source

## Documentation

* http://gcc.gnu.org/wiki/InstallingGCC
* http://gcc.gnu.org/install/configure.html
* http://gcc.gnu.org/onlinedocs/gcc/i386-and-x86-64-Options.html

## Settings

```sh
export GCC_BUILD=/tmp/gcc-$VERSION
export GCC_DIR=/opt/gcc/gcc-$GCC_VERSION
export CPU=native
```

## Other

* If you want to use these instructions to build on another system, your best bet is to set `CPU=generic~ or `CPU=native` unless you are sure you know what your CPU architecture is.
* Building in `/tmp` will be very fast when Linux implements this as a ramdisk (i.e. tmpfs) but you will need a lot of space.  I found that 4 GB was not enough to build GCC.

## Dependencies

http://gcc.gnu.org/wiki/InstallingGCC will tell you to just run <tt>./contrib/download_prerequisites</tt> instead of installing GMP, MPFR and MPC yourself.  I choose to do it the hard way in part because I can run these builds in parallel with the Cloog and ISL ones, thereby reducing the overall build time.

You can build GMP then MPFR then MPC at the same time you are building Cloog then ISL, i.e. the dependency graph looks like this:
```
GMP ---- MPFR ---- MPC ----|
  \                        |-----GCC
   \----ISL ---- Cloog ----|
```

### GMP 4.3.2
```sh
cd $GCC_BUILD && \
wget ftp://gcc.gnu.org/pub/gcc/infrastructure/gmp-4.3.2.tar.bz2 && \
tar -xjf gmp-4.3.2.tar.bz2 && \
cd gmp-4.3.2 && \
mkdir build && \
cd build && \
../configure --prefix=$GCC_DIR && \
make -j10 && \
make check && \
make install ; \
cd $GCC_BUILD ; \
```

### MPFR 2.4.2
```sh
cd $GCC_BUILD && \
wget ftp://gcc.gnu.org/pub/gcc/infrastructure/mpfr-2.4.2.tar.bz2 && \
tar -xjf mpfr-2.4.2.tar.bz2 && \
cd mpfr-2.4.2 && \
mkdir build && \
cd build && \
../configure --prefix=$GCC_DIR --with-gmp=$GCC_DIR && \
make -j10 && \
make check && \
make install ; \
cd $GCC_BUILD ; \
```

### MPC 0.8.1
```sh
cd $GCC_BUILD && \
wget ftp://gcc.gnu.org/pub/gcc/infrastructure/mpc-0.8.1.tar.gz && \
tar -xzf mpc-0.8.1.tar.gz && \
cd mpc-0.8.1 && \
mkdir build && \
cd build && \
../configure --prefix=$GCC_DIR --with-gmp=$GCC_DIR --with-mpfr=$GCC_DIR && \
make -j10 && \
make check && \
make install ; \
cd $GCC_BUILD ; \
```

### ISL 0.11.1
```sh
cd $GCC_BUILD && \
wget ftp://gcc.gnu.org/pub/gcc/infrastructure/isl-0.11.1.tar.bz2 && \
tar -xjf isl-0.11.1.tar.bz2 && \
cd isl-0.11.1 && \
mkdir build && \
cd build && \
../configure --prefix=$GCC_DIR --with-gmp-prefix=$GCC_DIR --with-gcc-arch=$CPU && \
make -j10 && \
make check && \
make install ; \
cd $GCC_BUILD ; \
```

### Cloog 0.18.0
```sh
cd $GCC_BUILD && \
wget ftp://gcc.gnu.org/pub/gcc/infrastructure/cloog-0.18.0.tar.gz && \
tar -xzf cloog-0.18.0.tar.gz && \
cd cloog-0.18.0 && \
mkdir build && \
cd build && \
../configure --prefix=$GCC_DIR --with-gmp-prefix=$GCC_DIR --with-isl-prefix=$GCC_DIR --with-gcc-arch=$CPU && \
make -j10 && \
make check && \
make install ; \
cd $GCC_BUILD ; \
```

## GCC 4.8.1

Before this step, you need to set the following (e.g. in <tt>~/.bashrc</tt>):
```sh
export LD_LIBRARY_PATH=$GCC_DIR/lib:$LD_LIBRARY_PATH
```
Without this, you cannot configure GCC because the shared libraries for the dependencies will not be found.

```sh
cd $GCC_BUILD && \
wget ftp://gcc.gnu.org/pub/gcc/releases/gcc-4.8.1/gcc-4.8.1.tar.bz2 && \
tar -xjf gcc-4.8.1.tar.bz2 && \
cd gcc-4.8.1 && \
mkdir build && \
cd build  && \
../configure \
--enable-threads=posix \
--enable-checking=release \
--with-system-zlib \
--enable-__cxa_atexit \
--enable-languages=c,c++,fortran \
--with-tune=$CPU \
--enable-bootstrap \
--enable-lto \
--with-gmp=$GCC_DIR \
--with-mpfr=$GCC_DIR \
--with-mpc=$GCC_DIR \
--with-cloog=$GCC_DIR \
--with-isl=$GCC_DIR --disable-isl-version-check \
--prefix=$GCC_DIR && \
make -j10 && \
make install ; \
cd $GCC_BUILD
```

## Total Automation

### Compact

This script was generously provided by Rob Latham, who knows a lot more about Bash than I do.

This is yet to be verified...
```sh
#!/bin/bash +x

JNUM=16

FTP_HOST=ftp://gcc.gnu.org/pub/gcc

GMP_VERSION=4.3.2
MPFR_VERSION=2.4.2
MPC_VERSION=0.8.1
ISL_VERSION=0.15
CLOOG_VERSION=0.18.0
GCC_VERSION=6.2.0

CPU=native

GCC_DIR=$HOME/work/tmp/GCC/gcc-$GCC_VERSION
GCC_BUILD=${GCC_DIR}/tmp

mkdir -p ${GCC_BUILD}

# process_lib: download, configure, build, install one of the gcc prerequisite
# libraries
# usage: process_lib <library> <version> <suffix> <path> <doodad> <configure_args>
#

process_lib() {
    cd ${GCC_BUILD}
    TOOL=$1
    TDIR=${TOOL}-${2}
    FILE=${TDIR}.tar.${3}
    INSTALLED=${GCC_DIR}/$5
    if [ -d ${TDIR} ] ; then
	echo ${TDIR} already exists! Using existing copy.
    else
	if [ -f ${FILE} ] ; then
	    echo ${FILE} already exists! Using existing copy.
	else
            # curl is better than wget on some machines (i.e. proxy)
	    #wget ${FTP_HOST}/$4/${FILE}
	    curl ${FTP_HOST}/$4/${FILE} -o ${FILE}
	fi
	echo Unpacking ${FILE}
	tar -xaf ${FILE}
    fi
    if [ -f ${INSTALLED} ] ; then
	echo ${INSTALLED} already exists! Skipping build.
    else
	cd ${GCC_BUILD}/${TDIR}
	mkdir build ; cd build

	../configure --prefix=${GCC_DIR} $6 && make -j ${MAKE_JNUM} && make install
	if [ "x$?" != "x0" ] ; then
	    echo FAILURE 1
	    exit
	fi
    fi
}

process_lib gmp $GMP_VERSION bz2 infrastructure lib/libgmp.a
process_lib mpfr $MPFR_VERSION bz2 infrastructure lib/libmpfr.a "--with-gmp=$GCC_DIR"
process_lib mpc $MPC_VERSION gz infrastructure lib/libmpc.a "--with-gmp=$GCC_DIR"
process_lib isl $ISL_VERSION bz2 infrastructure lib/libisl.a "--with-gmp-prefix=$GCC_DIR \
--with-gcc-arch=$CPU"
process_lib cloog $CLOOG_VERSION gz infrastructure lib/libcloog-isl.a "--with-gmp-prefix=$GCC_DIR \
--with-isl-prefix=$GCC_DIR --with-gcc-arch=$CPU"


process_lib gcc $GCC_VERSION bz2 releases/gcc-$GCC_VERSION /bin/gcc "
  --enable-threads=posix \
  --enable-checking=release \
  --with-system-zlib \
  --enable-__cxa_atexit \
  --enable-languages=c,c++,fortran \
  --with-tune=$CPU \
  --enable-bootstrap \
  --enable-lto \
  --with-gmp=$GCC_DIR \
  --with-mpfr=$GCC_DIR \
  --with-mpc=$GCC_DIR \
  --with-cloog=$GCC_DIR \
  --with-isl=$GCC_DIR --disable-isl-version-check
```

### Snapshot
```sh
#!/bin/bash +x

JNUM=30

FTP_HOST=ftp://gcc.gnu.org/pub/gcc
GCC_VERSION=4.9-20140316

CPU=power7

GCC_BUILD=/tmp/$USER
GCC_DIR=$HOME/GCC/$HOST/${GCC_VERSION}

mkdir -p ${GCC_BUILD}

cd ${GCC_BUILD}
TOOL=gcc
TDIR=${TOOL}-${GCC_VERSION}
FILE=${TDIR}.tar.bz2
INSTALLED=${GCC_DIR}/bin/gcc
if [ -d ${TDIR} ] ; then
  echo ${TDIR} already exists! Using existing copy.
else
  if [ -f ${FILE} ] ; then
    echo ${FILE} already exists! Using existing copy.
  else
    wget ${FTP_HOST}/snapshots/${GCC_VERSION}/${FILE}
  fi
  echo Unpacking ${FILE}                                                                                                                                              
  tar -xjf ${FILE}
fi
if [ -f ${INSTALLED} ] ; then
  echo ${INSTALLED} already exists! Skipping build.
else
  cd ${GCC_BUILD}/${TDIR}
  ./contrib/download_prerequisites
  # see http://gcc.gnu.org/wiki/FAQ#gnu_stubs-32.h
  # for info about --disable-multilib
  # I removed --disable-isl-version-check for the 4.9 build
  mkdir build ; cd build  && \
  ../configure \
  --disable-multilib \
  --enable-threads=posix \
  --enable-checking=release \
  --with-system-zlib \
  --enable-__cxa_atexit \
  --enable-languages=c,c++,fortran \
  --with-tune=$CPU \
  --enable-bootstrap \
  --enable-lto \
  --prefix=$GCC_DIR && \
  make -j$JNUM && make install
  if [ "x$?" != "x0" ] ; then
    echo FAILURE 1
    exit
  fi
fi
```

## Latest 

```sh
#!/bin/bash -x

JNUM=12

FTP_HOST=ftp://gcc.gnu.org/pub/gcc

GMP_VERSION=4.3.2
MPFR_VERSION=2.4.2
MPC_VERSION=0.8.1
ISL_VERSION=0.14
#ISL_VERSION=0.15
CLOOG_VERSION=0.18.0
GCC_VERSION=5.4.0
#GCC_VERSION=6.2.0

CPU=native

#GCC_DIR=$HOME/Work/GCC/gcc-$GCC_VERSION
GCC_DIR=/opt/gcc/$GCC_VERSION
GCC_BUILD=/tmp/gcc-$GCC_VERSION

mkdir -p ${GCC_BUILD}


# process_lib: download, configure, build, install one of the gcc prerequisite
# libraries
# usage: process_lib <library> <version> <suffix> <path> <doodad> <configure_args>
#

process_lib() {
    cd ${GCC_BUILD}
    TOOL=$1
    TDIR=${TOOL}-${2}
    FILE=${TDIR}.tar.${3}
    INSTALLED=${GCC_DIR}/$5
    if [ -d ${TDIR} ] ; then
        echo ${TDIR} already exists! Using existing copy.
    else
        if [ -f ${FILE} ] ; then
            echo ${FILE} already exists! Using existing copy.
        else
            #wget ${FTP_HOST}/$4/${FILE}
            curl ${FTP_HOST}/$4/${FILE} -o  ${FILE}
        fi
        echo Unpacking ${FILE}
        tar -xaf ${FILE}
    fi
    if [ -f ${INSTALLED} ] ; then
        echo ${INSTALLED} already exists! Skipping build.
    else
        cd ${GCC_BUILD}/${TDIR}
        if [ -f ./contrib/download_prerequisites ] ; then
            ./contrib/download_prerequisites
        fi
        mkdir build ; cd build
        
        ../configure --prefix=${GCC_DIR} $6 && make -j ${MAKE_JNUM} && make install
        if [ "x$?" != "x0" ] ; then
            echo FAILURE 1
            exit
        fi
    fi
}

#process_lib gmp $GMP_VERSION bz2 infrastructure lib/libgmp.a
#process_lib mpfr $MPFR_VERSION bz2 infrastructure lib/libmpfr.a "--with-gmp=$GCC_DIR --enable-shared --enable-static"
#process_lib mpc $MPC_VERSION gz infrastructure lib/libmpc.a "--with-gmp=$GCC_DIR --enable-shared --enable-static"
#process_lib isl $ISL_VERSION bz2 infrastructure lib/libisl.a "--with-gmp-prefix=$GCC_DIR \
#--with-gcc-arch=$CPU --enable-shared --enable-static"
#process_lib cloog $CLOOG_VERSION gz infrastructure lib/libcloog-isl.a "--with-gmp-prefix=$GCC_DIR \
#--with-isl-prefix=$GCC_DIR --with-gcc-arch=$CPU --enable-shared --enable-static"

#process_lib gcc $GCC_VERSION bz2 releases/gcc-$GCC_VERSION /bin/gcc "
#  --enable-shared --enable-static \
#  --enable-threads=posix \
#  --enable-checking=release \
#  --with-system-zlib \
#  --enable-__cxa_atexit \
#  --enable-languages=c,c++,fortran \
#  --with-tune=$CPU \
#  --enable-bootstrap \
#  --enable-lto \
#  --with-gmp=$GCC_DIR \
#  --with-mpfr=$GCC_DIR \
#  --with-mpc=$GCC_DIR \
#  --with-cloog=$GCC_DIR \
#  --with-isl=$GCC_DIR --disable-isl-version-check \
#  --disable-multilib
#"

process_lib gcc $GCC_VERSION bz2 releases/gcc-$GCC_VERSION /bin/gcc "
  --enable-shared --enable-static \
  --enable-threads=posix \
  --enable-checking=release \
  --with-system-zlib \
  --enable-__cxa_atexit \
  --enable-languages=c,c++,fortran \
  --with-tune=$CPU \
  --enable-bootstrap \
  --enable-lto \
  --disable-multilib
"
```

# Repo build

```sh
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
```
