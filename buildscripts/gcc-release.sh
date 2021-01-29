#!/bin/bash

for arg in "$@" ; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ] ; then
        HELP=1
    fi
done

if [ "$#" = "0" ] || [ "$HELP" = "1" ] ; then
    echo "Provide the versions you would like to build."
    echo "Valid versions are X.Y and X.Y.Z where"
    echo "X > 4, Y > 0, and Z = 0"
    echo "This script can support X = 4 but chooses not to."
fi

GCC_BASE=/opt/gcc
#GCC_BASE=$HOME/Work/GCC/

MAKE_JNUM="-j8"

FTP_HOST=ftp://gcc.gnu.org/pub/gcc

CPU=native

# determine if the version is valid
check_version() {
    V=$1

    IFS='.'
    read -ra S <<< "${V}"
    # very important - leaving this set breaks things
    unset IFS

    MAJOR=${S[0]}
    MINOR=${S[1]}
    PATCH=${S[2]}

    # patchlevel is optional
    if [ -z "$PATCH" ] ; then
        PATCH=0
    fi

    if ! [[ "$MAJOR" =~ ^[0-9]+$ ]] ; then
        echo "Nonsensical choice of $MAJOR"
        exit 1
    fi
    if ! [[ "$MINOR" =~ ^[0-9]+$ ]] ; then
        echo "Nonsensical choice of $MINOR"
        exit 2
    fi
    if ! [[ "$PATCH" =~ ^[0-9]+$ ]] ; then
        echo "Nonsensical choice of $PATCH"
        exit 2
    fi

    if [ $MAJOR -le 4 ] ; then
        echo "Sorry, dinosaur, I don't care about GCC ${MAJOR} anymore."
        exit 4
    fi

    if [ $MINOR -lt 1 ] ; then
        echo "No supported releases are minor version $MINOR"
        exit 3
    fi

    if [ $PATCH != 0 ] ; then
        echo "All supported releases are patchlevel 0 (not $PATCH)"
        exit $PATCH
    fi

    echo "GCC $MAJOR.$MINOR.$PATCH is probably a valid choice..."
}

# process_lib: download, configure, build, install one of the gcc prerequisite
# libraries
# usage: process_lib <library> <version> <suffix> <path> <doodad> <configure_args>
process_lib() {
    GCC_DIR=$GCC_BASE/${2}
    GCC_BUILD=/tmp/gcc-${2}
    mkdir -p ${GCC_BUILD}
    cd ${GCC_BUILD}
    TOOL=$1
    TDIR=${TOOL}-${2}
    FILE=${TDIR}.tar.${3}
    INSTALLED=${GCC_DIR}/${5}
    if [ -d ${TDIR} ] ; then
        echo ${TDIR} already exists! Using existing copy.
    else
        if [ -f ${FILE} ] ; then
            echo ${FILE} already exists! Using existing copy.
        else
            #wget ${FTP_HOST}/${4}/${FILE}
            curl ${FTP_HOST}/${4}/${FILE} -o  ${FILE}
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
        ../configure --prefix=${GCC_DIR} ${6} && make ${MAKE_JNUM} && make install
        if [ "x$?" != "x0" ] ; then
            echo FAILURE 1
            exit
        fi
    fi
}

for v in "$@" ; do
    GCC_VERSION=$v
    check_version $GCC_VERSION
    process_lib gcc ${GCC_VERSION} gz releases/gcc-${GCC_VERSION} /bin/gcc "
      --program-suffix=-${GCC_VERSION:0:1} \
      --enable-shared --enable-static \
      --enable-threads=posix \
      --enable-checking=release \
      --with-system-zlib \
      --enable-__cxa_atexit \
      --enable-languages=c,c++,fortran \
      --enable-bootstrap \
      --enable-lto \
      --enable-gold=yes \
      --enable-ld=yes \
      --disable-multilib
    "
done
