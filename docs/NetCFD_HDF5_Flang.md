# HDF5

```
git clone --recursive https://github.com/HDFGroup/hdf5.git
cd hdf5 && ./autogen.sh && mkdir build && cd build
../configure --prefix=$HOME/FORTRAN/install-netcdf \
  CC=/opt/llvm/latest/bin/clang \
  CXX=/opt/llvm/latest/bin/clang++ \
  FC=/opt/llvm/latest/bin/flang-new \
  --enable-fortran
make -j32 install
```

This fails and you have to disable the test utilities related to `REAL(2)` and `REAL(3)` 
by commenting out lines 32, 48, and 175-246 of `hdf5/build/fortran/test/tf_gen.F90`.

# NetCFD-C

```
git clone --recursive https://github.com/Unidata/netcdf-c.git
cd netcdf-c && ./bootstrap && mkdir build && cd build
../configure --prefix=$HOME/FORTRAN/install-netcdf \
  CC=/opt/llvm/latest/bin/clang \
  CXX=/opt/llvm/latest/bin/clang++ \
  LDFLAGS="-L$HOME/FORTRAN/install-netcdf/lib -Wl,-rpath=$HOME/FORTRAN/install-netcdf/lib -lhdf5" \
  LIBS="-L$HOME/FORTRAN/install-netcdf/lib -Wl,-rpath=$HOME/FORTRAN/install-netcdf/lib -lhdf5" \
  CPPFLAGS="-I$HOME/FORTRAN/install-netcdf/include"
make -j32 install
```

# NetCFD-Fortran

```
git clone --recursive https://github.com/Unidata/netcdf-fortran.git
cd netcdf-fortran && autoreconf -fiv && mkdir build && cd build
../configure --prefix=$HOME/FORTRAN/install-netcdf \
  CC=/opt/llvm/latest/bin/clang \
  CXX=/opt/llvm/latest/bin/clang++ \
  LDFLAGS="-L$HOME/FORTRAN/install-netcdf/lib -Wl,-rpath=$HOME/FORTRAN/install-netcdf/lib -lhdf5" \
  LIBS="-L$HOME/FORTRAN/install-netcdf/lib -Wl,-rpath=$HOME/FORTRAN/install-netcdf/lib -lhdf5" \
  CPPFLAGS="-I$HOME/FORTRAN/install-netcdf/include"
make -j32 install
```
