http://hpctoolkit.org/man/hpctoolkit.html

# Building HPCToolkit
```
../configure --prefix=/opt/hpctoolkit/ \
CC=gcc CXX=g++ \
MPICXX=/opt/mpich/dev/gcc/default/bin/mpicxx \
MPICC=/opt/mpich/dev/gcc/default/bin/mpicc \
MPIF77=/opt/mpich/dev/gcc/default/bin/mpifort \
--with-externals=/home/jrhammon/Work/HPCToolkit/hpctoolkit-externals-install  \
--with-papi=/usr
```

The option `--enable-mpi-wrapper` breaks the build for a trivial reason that I do not care to address right now.

# Instrumenting and Profiling NWChem

After the binary exists:
```
/opt/hpctoolkit/bin/hpcstruct $NWCHEM_TOP/bin/LINUX64/nwchem
```

Running the job:
```
/opt/mpich/dev/intel/default/bin/mpiexec -n 36 -env OMP_NUM_THREADS 1 \
/opt/hpctoolkit/bin/hpcrun $NWCHEM_TOP/bin/LINUX64/nwchem $IN.nw | \
tee $IN.out
```

After the job is executed:
```
/opt/hpctoolkit/bin/hpcprof -I $NWCHEM_TOP/src/'*' \
-S $NWCHEM_TOP/bin/LINUX64/nwchem.hpcstruct \
./hpctoolkit-nwchem-measurements
```