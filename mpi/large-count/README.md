# Large-count support in datatypes

MPI-4 added this function:
```c
int MPI_Type_get_envelope_c(MPI_Datatype datatype, MPI_Count *num_integers,
             MPI_Count *num_addresses, MPI_Count *num_large_counts,
             MPI_Count *num_datatypes, int *combiner)
```
This function not only allows the user to query `MPI_Count` arguments in datatypes,
which is essential, but also supports large-count vectors of arguments.

I found this interesting, since it's hard to imagine any reasonable
person creating an MPI datatype that has such a large array of arguments.

I wrote a program to generate a C `struct` that has a huge number of members.
This is artificial since a vector would work in this case, but it's a good
way to establish a lower bound.

The compilation time with GCC is - not surprisingly - linear in the number of
`struct` members and it appears that merely compiling a file with a `struct` that
exceeds the `INT_MAX` limit takes at least an hour, and probably more.
This file is also likely to be tens of gigabytes in size.

```sh
$ gcc -O3 gigastruct.c -o gen && time ./gen $((1024*1024*100))  && ll temp.c && time gcc -Os temp.c && time ./a.out

real	0m15.372s
user	0m8.808s
sys	0m1.334s
-rw-r--r-- 1 jehammond 1.6G Dec 16 04:44 temp.c

real	4m53.584s
user	4m30.625s
sys	0m22.525s
sizeof=104857600

real	0m0.001s
user	0m0.000s
sys	0m0.001s
```
