# Specification

https://www.khronos.org/registry/cl/specs/

# Random

## OpenCL-CLHPP testing

```sh
# get it
git clone --recursive https://github.com/KhronosGroup/OpenCL-CLHPP.git && \
cd OpenCL-CLHPP && \
mkdir build && \
cd build
# test it
git clean -dfx ; cmake .. \
  -DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-9 \
  -DCMAKE_CXX_FLAGS="-fmax-errors=1" \
  -DOPENCL_INCLUDE_DIR=/opt/intel/oneapi/compiler/2021.1-beta08/linux/include/sycl/ \
  -DOPENCL_LIB_DIR=/opt/intel/oneapi/compiler/2021.1-beta08/linux/lib/ && make 
```
