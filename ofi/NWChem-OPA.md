# Building

Set this however you like.  It will be used by all of the remaining commands.
```sh
export NWCHEM_ROOT=$HOME/NWCHEM
mkdir $NWCHEM_ROOT
```

## Autotools

In order to build from Git repo sources, you may need to upgrade Autotools (m4, autoconf, automake, libtool) in order to generate `configure` scripts.  If possible, upgrade to the latest Autotools using your package manager.  If you need install from source, see [this page](https://github.com/jeffhammond/HPCInfo/wiki/Autotools).

As far as I know, the following versions are sufficient.  If they are not, it is likely because MPICH needed to move to a later version to work around a bug.
```sh
M4_VERSION=1.4.17
LIBTOOL_VERSION=2.4.4
AUTOCONF_VERSION=2.69
AUTOMAKE_VERSION=1.15
```

## OFI

If Git is installed, do this:
```sh
cd $NWCHEM_ROOT
git clone --depth 10 https://github.com/ofiwg/libfabric.git
```
If Git is not installed, do this:
```sh
cd $NWCHEM_ROOT
wget https://github.com/ofiwg/libfabric/archive/master.zip
unzip master.zip
mv libfabric-master libfabric
```

For Intel Omni Path, we use `--enable-psm2`.  For Intel True Scale, use `--enable-psm` instead.  For other networks, we need to do something else.  See `../configure --help` for details.
```sh
cd $NWCHEM_ROOT/libfabric/
./autogen.sh # if this fails, upgrade Autotools
mkdir $NWCHEM_ROOT/libfabric/build
cd $NWCHEM_ROOT/libfabric/build
../configure CC=icc CXX=icpc --enable-psm2 --disable-udp --disable-sockets --disable-rxm --prefix=$NWCHEM_ROOT/deps
make -j8 install
```

## MPICH

If Git is installed, do this:
```sh
cd $NWCHEM_ROOT/
git clone -b 'ch4/stable' --depth 10 http://git.mpich.org/mpich-dev.git mpich-ch4
```
If Git is not installed, go to http://git.mpich.org/mpich-dev.git/shortlog/refs/heads/ch4/stable and download the latest snapshot (e.g. http://git.mpich.org/mpich-dev.git/snapshot/e9201c991a92c59019c594a022e3b0b27dda7d95.tar.gz) and unpack that into  ` $NWCHEM_ROOT/mpich-ch4 directory.`

Now build MPICH:
```sh
cd $NWCHEM_ROOT/mpich-ch4/
./autogen.sh # if this fails, upgrade Autotools
mkdir $NWCHEM_ROOT/mpich-ch4/build
cd $NWCHEM_ROOT/mpich-ch4/build
../configure CC=icc CXX=icpc FC=ifort F77=ifort \
             --with-ofi=$NWCHEM_ROOT/deps --with-libfabric=$NWCHEM_ROOT/deps \
             --with-device=ch4:ofi --with-ch4-netmod-ofi-args=no-data \
             --prefix=$NWCHEM_ROOT/deps 
make -j8 install
```

## Casper

If Git is installed, do this:
```sh
cd $NWCHEM_ROOT
git clone http://git.mpich.org/soft/dev/casper.git
```
If Git is not installed, do this:
```sh
cd $NWCHEM_ROOT
wget http://git.mpich.org/soft/dev/casper.git/snapshot/master.tar.gz
tar -xzf master.tar.gz
mv casper-master* casper
```

Now build Casper:
```sh
cd $NWCHEM_ROOT/casper/
./autogen.sh # if this fails, upgrade Autotools
mkdir $NWCHEM_ROOT/casper/build
cd $NWCHEM_ROOT/casper/build
../configure CC=$NWCHEM_ROOT/deps/bin/mpicc --prefix=$NWCHEM_ROOT/deps 
make -j8 install
```

## ARMCI-MPI

If Git is installed, do this:
```sh
cd $NWCHEM_ROOT/
git clone -b 'mpi3rma' --depth 10 http://git.mpich.org/armci-mpi.git
```
If Git is not installed, go to http://git.mpich.org/armci-mpi.git/ and download the latest snapshot (e.g. http://git.mpich.org/armci-mpi.git/snapshot/964033675c986639f7c6fe877809cf65fbfc9410.tar.gz) and unpack that into  ` $NWCHEM_ROOT/armci-mpi directory.`

Now build ARMCI-MPI:
```sh
cd $NWCHEM_ROOT/armci-mpi/
./autogen.sh # if this fails, upgrade Autotools
mkdir $NWCHEM_ROOT/armci-mpi/build
cd $NWCHEM_ROOT/armci-mpi/build
../configure MPICC=$NWCHEM_ROOT/deps/bin/mpicc MPIEXEC=$NWCHEM_ROOT/deps/bin/mpirun \
             --enable-win-allocate --enable-explicit-progress \
             --prefix=$NWCHEM_ROOT/deps
make -j8 install
```

At this point, it is prudent to verify ARMCI-MPI is working:
```sh
make check MPIEXEC="$NWCHEM_ROOT/deps/bin/mpirun -n 2"
```

You can try a second time with Casper active, but this might require you to run ARMCI-MPI tests manually to get the right number of MPI processes.  Skip this step if you are confused.
```sh
export CSP_NG=1
export LD_PRELOAD=$NWCHEM_ROOT/deps/lib/libcasper.so
make check MPIEXEC="$NWCHEM_ROOT/deps/bin/mpirun -n 4 -genv CSP_NG 1 -genv LD_PRELOAD $NWCHEM_ROOT/deps/lib/libcasper.so"
```

## NWChem

Download NWChem from http://www.nwchem-sw.org/index.php/Download.  If PNNL's website is down, you can download https://github.com/jeffhammond/nwchem/archive/master.zip instead.

However you get NWChem, please set `$NWCHEM_TOP` to its location.

```sh
export NWCHEM_TARGET=LINUX64
export NWCHEM_MODULES=all
export NWCHEM_TOP=${NWCHEM_ROOT}/nwchem

export USE_MPI=y
export ARMCI_NETWORK=ARMCI
export EXTERNAL_ARMCI_PATH=${NWCHEM_ROOT}/deps

# required for NWPW but not otherwise
export USE_MPIF=y
export USE_MPIF4=y

MPI_DIR=${NWCHEM_ROOT}/deps
export MPI_LIB="${MPI_DIR}/lib"
export MPI_INCLUDE="${MPI_DIR}/include"

# the following are not necessary if you use CC=mpicc, but that isn't not recommended
MPICH_LIBS="-lmpifort -lmpi"
SYS_LIBS="-ldl -lrt -lpthread -static-intel"
export LIBMPI="-L${MPI_DIR}/lib -Wl,-rpath -Wl,${MPI_DIR}/lib ${MPICH_LIBS} ${SYS_LIBS}"

export CC=icc
export FC=ifort
export F77=ifort
export BLASOPT="-mkl=parallel -qopenmp"
export USE_OPENMP=T
```

Now you can try to compiler NWChem...
```sh
cd $NWCHEM_TOP/src
make nwchem_config
make -j8
```

This may fail in one of two places.  If it fails in GA configure, do this:
```sh
cd $NWCHEM_TOP/src/tools/build
../ga-5-4/configure --prefix=$NWCHEM_TOP/src/tools/install --with-tcgmsg --with-mpi \
                    CC=$NWCHEM_ROOT/deps/bin/mpicc MPICC=$NWCHEM_ROOT/deps/bin/mpicc \
                    CXX=$NWCHEM_ROOT/deps/bin/mpicxx MPICXX=$NWCHEM_ROOT/deps/bin/mpicxx \
                    F77=$NWCHEM_ROOT/deps/bin/mpifort MPIF77=$NWCHEM_ROOT/deps/bin/mpifort \
                    --with-armci=$NWCHEM_ROOT/deps \
                    --enable-peigs --enable-underscoring --disable-mpi-tests \
                    --without-scalapack --without-lapack --with-blas8=$BLASOPT
make -j8 install
```
If this fails, email Jeff to debug.

If NWChem fails to link, email to Jeff to debug.  It is likely something trivial to fix but not easy to enumerate in advance.

# Running jobs

Casper requires one or more cores per node for progress.  Set `CSP_NG=1` or a larger number.  NWChem will use `$(($PPN-$CSP_NG))` processes per node.
```sh
mpirun -n $(($NODES*$PPN)) -genv CSP_NG 1 \
       -genv LD_PRELOAD $NWCHEM_ROOT/deps/lib/libcasper.so \
       $NWCHEM_TOP/bin/LINUX64/nwchem input.nw
```

To see if/how Casper is helping, run this to compare to NWChem without Casper *on the same number of cores*:
```sh
mpirun -n $(($NODES*$(($PPN-1)))) -genv CSP_ASYNC_CONFIG off \
       -genv LD_PRELOAD $NWCHEM_ROOT/deps/lib/libcasper.so \
       $NWCHEM_TOP/bin/LINUX64/nwchem input.nw
```
