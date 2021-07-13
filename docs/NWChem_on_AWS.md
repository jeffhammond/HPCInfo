# NWChem on AWS

## MPI

```
mkdir ~/MPI
cd ~/MPI
wget https://github.com/open-mpi/ompi/archive/refs/tags/v4.1.1.tar.gz
tar -xaf v4.1.1.tar.gz
```

### NVHPC

```
cd ~/MPI/ompi-4.1.1
mkdir build-nvhpc
../configure --build=aarch64-redhat-linux-gnu --host=aarch64-redhat-linux-gnu --program-prefix= --disable-dependency-tracking --prefix=~/MPI/nvhpc-ompi-4.1.1 --with-sge --without-verbs --disable-builtin-atomics --with-libfabric=/opt/amazon/efa --with-hwloc=external --with-libevent=external FC=nvfortran CC=gcc CXX=g++
```
  
### GCC

Just use what is in `/opt/amazon/openmpi`.

### ARM

```
cd ~/MPI/ompi-4.1.1
mkdir build-arm
../configure --build=aarch64-redhat-linux-gnu --host=aarch64-redhat-linux-gnu --program-prefix= --disable-dependency-tracking --prefix=~/MPI/arm-ompi-4.1.1 --with-sge --without-verbs --disable-builtin-atomics --with-libfabric=/opt/amazon/efa --with-hwloc=external --with-libevent=external FC=armflang CC=armclang CXX=armclang++
```

## NWChem

```
mkdir ~/NWCHEM
git clone https://github.com/nwchemgit/nwchem.git ~/NWCHEM/gcc
cp -r gcc arm &
cp -r gcc nvhpc &
wait
```

```
cd ~/NWCHEM
wget https://github.com/jeffhammond/HPCInfo/blob/master/docs/nwchem/setup-nvhpc.sh
wget https://github.com/jeffhammond/HPCInfo/blob/master/docs/nwchem/setup-gcc.sh
wget https://github.com/jeffhammond/HPCInfo/blob/master/docs/nwchem/setup-arm.sh
```

### NVHPC

```
cd ~/NWCHEM/nvhpc/src/tools
./get-tools-github
MPICC=~/MPI/arm-ompi-4.1.1 ./install-armci-mpi
source ~/NWCHEM/setup-nvhpc.sh
cd $NWCHEM_TOP/src
make nwchem_config
make -j32
```

### GCC

```
cd ~/NWCHEM/gcc/src/tools
./get-tools-github
MPICC=/opt/amazon/openmpi/bin/mpicc ./install-armci-mpi
source ~/NWCHEM/setup-gcc.sh
cd $NWCHEM_TOP/src
make nwchem_config
make -j32
```

### ARM

```
cd ~/NWCHEM/arm/src/tools
./get-tools-github
MPICC=~/MPI/arm-ompi-4.1.1 ./install-armci-mpi
source ~/NWCHEM/setup-arm.sh
cd $NWCHEM_TOP/src
make nwchem_config
make -j32
```
