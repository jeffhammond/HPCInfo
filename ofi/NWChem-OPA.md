# Building

Set this however you like.  It will be used by all of the remaining commands.
```
export NWCHEM_ROOT=$HOME/NWCHEM
```

## Autotools

In order to build from Git repo sources, you may need to upgrade Autotools (m4, autoconf, automake, libtool) in order to generate `configure` scripts.  If possible, upgrade to the latest Autotools using your package manager.  If you need install from source, see [this page](https://github.com/jeffhammond/HPCInfo/wiki/Autotools).

As far as I know, the following versions are sufficient.  If they are not, it is likely because MPICH needed to move to a later version to work around a bug.
```
M4_VERSION=1.4.17
LIBTOOL_VERSION=2.4.4
AUTOCONF_VERSION=2.69
AUTOMAKE_VERSION=1.15
```

## OFI

If Git is installed, do this:
```
cd $NWCHEM_ROOT
git clone https://github.com/ofiwg/libfabric.git
```
If Git is not installed, do this:
```
cd $NWCHEM_ROOT
wget https://github.com/ofiwg/libfabric/archive/master.zip
unzip master.zip
mv libfabric-master libfabric
```

For Intel Omni Path, we use `--enable-psm2`.  For other networks, we need to do something else.  See `../configure --help` for details.
```
cd $NWCHEM_ROOT/libfabric/
./autogen.sh # if this fails, upgrade Autotools
mkdir $NWCHEM_ROOT/libfabric/build
cd $NWCHEM_ROOT/libfabric/build
../configure CC=icc CXX=icpc --enable-psm2 --disable-udp --disable-sockets --disable-rxm --prefix=$NWCHEM_ROOT/deps
make -j8 install
```

## MPICH

If Git is installed, do this:
```
cd $NWCHEM_ROOT/
git clone -b 'ch4/stable' --depth 10 http://git.mpich.org/mpich-dev.git mpich-ch4
```
If Git is not installed, go to http://git.mpich.org/mpich-dev.git/shortlog/refs/heads/ch4/stable and download the latest snapshot (e.g. http://git.mpich.org/mpich-dev.git/snapshot/e9201c991a92c59019c594a022e3b0b27dda7d95.tar.gz) and unpack that into  ` $NWCHEM_ROOT/mpich-ch4 directory.`

Now build MPICH:
```
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
```
cd $NWCHEM_ROOT
git clone http://git.mpich.org/soft/dev/casper.git
```
If Git is not installed, do this:
```
cd $NWCHEM_ROOT
wget http://git.mpich.org/soft/dev/casper.git/snapshot/master.tar.gz
tar -xzf master.tar.gz
mv casper-master* casper
```

Now build Casper:
```
cd $NWCHEM_ROOT/casper/
./autogen.sh # if this fails, upgrade Autotools
mkdir $NWCHEM_ROOT/casper/buid
cd $NWCHEM_ROOT/casper/build
../configure CC=$NWCHEM_ROOT/deps/bin/mpicc --prefix=$NWCHEM_ROOT/deps 
make -j8 install
```

## ARMCI-MPI

If Git is installed, do this:
```
cd $NWCHEM_ROOT/
git clone -b 'mpi3rma' --depth 10 http://git.mpich.org/armci-mpi.git
```
If Git is not installed, go to http://git.mpich.org/armci-mpi.git/ and download the latest snapshot (e.g. http://git.mpich.org/armci-mpi.git/snapshot/964033675c986639f7c6fe877809cf65fbfc9410.tar.gz) and unpack that into  ` $NWCHEM_ROOT/armci-mpi directory.`

Now build ARMCI-MPI:
```
cd $NWCHEM_ROOT/armci-mpi/
./autogen.sh # if this fails, upgrade Autotools
mkdir $NWCHEM_ROOT/armci-mpi/buid
cd $NWCHEM_ROOT/armci-mpi/build
../configure MPICC=$NWCHEM_ROOT/deps/bin/mpicc MPIEXEC=$NWCHEM_ROOT/deps/bin/mpirun \
             --enable-win-allocate --enable-explicit-progress \
             --prefix=$NWCHEM_ROOT/deps
make -j8 install
```

At this point, it is prudent to verify ARMCI-MPI is working:
```
make check
```

You can try a second time with Casper active, but this might require you to run ARMCI-MPI tests manually to get the right number of MPI processes.  Skip this step if you are confused.
```
export CSP_NG=1
export LD_PRELOAD=$NWCHEM_ROOT/deps/lib/libcasper.so
```

## NWChem



# Running jobs
