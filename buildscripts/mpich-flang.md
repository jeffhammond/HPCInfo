# Building MPICH with Flang

Issues:
  - There are excessive warnings related to nullability.
  - The C++ bindings cause problems (case statements have duplicate labels).
  - `mpi_f08.mod` isn't built.

```
../configure CC=/opt/llvm/latest/bin/clang FC=/opt/llvm/latest/bin/flang-new CXX=/opt/llvm/latest/bin/clang++ --enable-fortran=all --prefix=/opt/mpich/flang --disable-cxx CFLAGS="-Wno-nullability-completeness"
make -j4
```
