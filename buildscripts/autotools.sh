#!/bin/bash -v

set -ex

TEMP=/tmp/
PREFIX=/usr/local
SUDO=sudo

MAKE_JNUM="-j8"

M4_VERSION=1.4.19
LIBTOOL_VERSION=2.4.6
AUTOCONF_VERSION=2.71
AUTOMAKE_VERSION=1.16.5

mkdir -p ${TEMP}
mkdir -p ${PREFIX}

cd ${TEMP}
TOOL=m4
TDIR=${TOOL}-${M4_VERSION}
FILE=${TDIR}.tar.gz
if [ -f ${FILE} ] ; then
  echo ${FILE} already exists! Using existing copy.
else
  wget http://ftp.gnu.org/gnu/${TOOL}/${FILE}
fi
if [ -d ${TDIR} ] ; then
  echo ${TDIR} already exists! Using existing copy.
else
  echo Unpacking ${FILE}
  tar -xaf ${FILE}
fi
cd ${TEMP}/${TDIR}
./configure --prefix=${PREFIX} && make ${MAKE_JNUM} && ${SUDO} make install
if [ "x$?" != "x0" ] ; then
  echo FAILURE 1
  exit
fi

cd ${TEMP}
TOOL=libtool
TDIR=${TOOL}-${LIBTOOL_VERSION}
FILE=${TDIR}.tar.gz
if [ ! -f ${FILE} ] ; then
  wget http://ftp.gnu.org/gnu/${TOOL}/${FILE}
else
  echo ${FILE} already exists! Using existing copy.
fi
if [ ! -d ${TDIR} ] ; then
  echo Unpacking ${FILE}
  tar -xaf ${FILE}
else
  echo ${TDIR} already exists! Using existing copy.
fi
cd ${TEMP}/${TDIR}
./configure --prefix=${PREFIX} M4=${PREFIX}/bin/m4 && make ${MAKE_JNUM} && ${SUDO} make install
if [ "x$?" != "x0" ] ; then
  echo FAILURE 2
  exit
fi

cd ${TEMP}
TOOL=autoconf
TDIR=${TOOL}-${AUTOCONF_VERSION}
FILE=${TDIR}.tar.gz
if [ ! -f ${FILE} ] ; then
  wget http://ftp.gnu.org/gnu/${TOOL}/${FILE}
else
  echo ${FILE} already exists! Using existing copy.
fi
if [ ! -d ${TDIR} ] ; then
  echo Unpacking ${FILE}
  tar -xaf ${FILE}
else
  echo ${TDIR} already exists! Using existing copy.
fi
cd ${TEMP}/${TDIR}
./configure --prefix=${PREFIX} M4=${PREFIX}/bin/m4 && make ${MAKE_JNUM} && ${SUDO} make install
if [ "x$?" != "x0" ] ; then
  echo FAILURE 3
  exit
fi

cd ${TEMP}
TOOL=automake
TDIR=${TOOL}-${AUTOMAKE_VERSION}
FILE=${TDIR}.tar.gz
if [ ! -f ${FILE} ] ; then
  wget http://ftp.gnu.org/gnu/${TOOL}/${FILE}
else
  echo ${FILE} already exists! Using existing copy.
fi
if [ ! -d ${TDIR} ] ; then
  echo Unpacking ${FILE}
  tar -xaf ${FILE}
else
  echo ${TDIR} already exists! Using existing copy.
fi
cd ${TEMP}/${TDIR}
./configure --prefix=${PREFIX} M4=${PREFIX}/bin/m4 && make ${MAKE_JNUM} && ${SUDO} make install
if [ "x$?" != "x0" ] ; then
  echo FAILURE 4
  exit
fi

#rm -f autoconf-${AUTOCONF_VERSION}.tar.gz
#rm -f automake-${AUTOMAKE_VERSION}.tar.gz
#rm -f libtool-${LIBTOOL_VERSION}.tar.gz
#rm -f m4-${M4_VERSION}.tar.gz
#rm -rf autoconf-${AUTOCONF_VERSION}
#rm -rf automake-${AUTOMAKE_VERSION}
#rm -rf libtool-${LIBTOOL_VERSION}
#rm -rf m4-${M4_VERSION}
