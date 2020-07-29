#!/bin/bash -xe

#GCC_BASE=/opt/gcc
GCC_BASE=$HOME/GCC/
#GCC_TEMP=/tmp
GCC_TEMP=/localdisk/${USER}

#MAKE_JNUM="-j40"
MAKE_JNUM="-j8"

FTP_HOST=ftp://gcc.gnu.org/pub/gcc

CPU=native

# process_lib: download, configure, build, install one of the gcc prerequisite
# libraries
# usage: process_lib <library> <version> <suffix> <path> <doodad> <configure_args>
process_lib() {
    GCC_DIR=${GCC_BASE}/${GCC_VERSION}
    GCC_BUILD=${GCC_TEMP}/gcc-${GCC_VERSION}
    mkdir -p ${GCC_BUILD}
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
            wget ${FTP_HOST}/$4/${FILE}
            #curl ${FTP_HOST}/$4/${FILE} -o  ${FILE}
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
        mkdir -p build ; cd build
        ../configure --prefix=${GCC_DIR} $6 && make ${MAKE_JNUM} && make install
        if [ "x$?" != "x0" ] ; then
            echo FAILURE 1
            exit
        fi
    fi
}

for v in 10.2.0 9.3.0 8.4.0 7.5.0 ; do
    GCC_VERSION=$v
    # There is a better way to do this...
    if [ ${GCC_VERSION:0:1} -eq 1 ] ; then
        GCC_SUFFIX=-${GCC_VERSION:0:2}
    else
        GCC_SUFFIX=-${GCC_VERSION:0:1}
    fi
    process_lib gcc ${GCC_VERSION} gz releases/gcc-${GCC_VERSION} /bin/gcc "
      --program-suffix=${GCC_SUFFIX} \
      --enable-shared --enable-static \
      --enable-threads=posix \
      --enable-checking=release \
      --with-system-zlib \
      --enable-__cxa_atexit \
      --enable-languages=c,c++,fortran \
      --with-tune=$CPU \
      --enable-bootstrap \
      --enable-lto \
      --enable-gold=yes \
      --enable-ld=yes \
      --disable-multilib
    "
done
