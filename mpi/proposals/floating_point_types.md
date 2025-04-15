# Context

https://github.com/mpi-forum/mpi-issues/issues/66

# Function

We have this already:
```c
int MPI_Type_create_f90_real(int p, int r, MPI_Datatype *newtype)
```

```c
int MPI_Type_create_floating_point(int storage_bits,
                                   int precision_bits,
                                   int exponent_bits,
                                   int base, 
                                   int * alignment,
                                   MPI_Datatype *newtype)
```

If not `MPI_DATATYPE_NULL`, the returned type behaves as a predefined
datatype; it is already committed and cannot be freed.
