# Run command

On a machine with 8 cores and HyperThreading enabled, do this:
```
OMP_NUM_THREADS=8 OMP_PROC_BIND=TRUE OMP_PLACES={0:16:2} OMP_DISPLAY_ENV=TRUE ./cxx11-counter.${COMPILER}
```
