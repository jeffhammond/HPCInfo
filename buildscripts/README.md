# Other Builds of Interest

## POCL

Apple has decided to break everything they can w.r.t. MacOS developer experience.
Thanks to https://stackoverflow.com/a/55797977/2189128 for telling me how to find `stdlib.h`.

```sh
export  SDKROOT=`xcrun --show-sdk-path` 
$ cmake -DWITH_LLVM_CONFIG=/usr/local/Cellar/llvm/8.0.0_1/bin/llvm-config \
        -DCMAKE_CXX_COMPILER=/usr/local/Cellar/llvm/8.0.0_1/bin/clang++ \
        -DCMAKE_C_COMPILER=/usr/local/Cellar/llvm/8.0.0_1/bin/clang \
        -DCMAKE_INSTALL_PREFIX=/opt/pocl  .. && make
```
