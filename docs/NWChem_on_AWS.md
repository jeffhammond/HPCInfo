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

### NVHPC

```
cd ~/NWCHEM/nvhpc/src
./get-tools-github
MPICC=~/MPI/arm-ompi-4.1.1 ./install-armci-mpi
```

### GCC

```
cd ~/NWCHEM/gcc/src
./get-tools-github
MPICC=/opt/amazon/openmpi/bin/mpicc ./install-armci-mpi
```

### ARM

```
cd ~/NWCHEM/arm/src
./get-tools-github
MPICC=~/MPI/arm-ompi-4.1.1 ./install-armci-mpi
```
