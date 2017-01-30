# Run command

On a machine with 8 cores and HyperThreading enabled, do this:

## OpenMP
```
OMP_NUM_THREADS=8 OMP_PROC_BIND=TRUE OMP_PLACES={0:8:1} OMP_DISPLAY_ENV=TRUE ./cxx11-counter.${COMPILER}
```

## KMP (Intel and LLVM)
```
OMP_NUM_THREADS=8 KMP_AFFINITY=scatter,verbose,granularity=fine KMP_PLACE_THREADS=1s,8c,t1 ./cxx11-counter.${COMPILER}
```
