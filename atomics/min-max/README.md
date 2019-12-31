# Performance data

```
$ ./f90-min-max.gcc
 me,nt,chunk,start,stop=           1           4         256         256         512
 me,nt,chunk,start,stop=           0           4         256           0         256
 me,nt,chunk,start,stop=           2           4         256         512         768
 me,nt,chunk,start,stop=           3           4         256         768        1024
i32: min=         0 took    0.9585430 seconds
i64: min=         0 took    0.8650730 seconds
r32: min= 0.000E+00 took    0.8553090 seconds
r64: min= 0.000E+00 took    0.8563550 seconds
i32: max=        36 took    0.9827830 seconds
i64: max=        36 took    0.8484380 seconds
r32: max= 0.360E+02 took    0.8711210 seconds
r64: max= 0.360E+02 took    0.8752950 seconds

$ ./f90-min-max.intel
 me,nt,chunk,start,stop=           2           4         256         512         768
 me,nt,chunk,start,stop=           0           4         256           0         256
 me,nt,chunk,start,stop=           1           4         256         256         512
 me,nt,chunk,start,stop=           3           4         256         768        1024
i32: min=         0 took    0.0437741 seconds
i64: min=         0 took    0.0403578 seconds
r32: min= 0.000E+00 took    0.0354600 seconds
r64: min= 0.000E+00 took    0.0471640 seconds
i32: max=        36 took    0.0381951 seconds
i64: max=        36 took    0.0489380 seconds
r32: max= 0.360E+02 took    0.0465162 seconds
r64: max= 0.360E+02 took    0.0417261 seconds

$ ./gcc-min-max.gcc
me,nt,chunk,start,stop=  1   4  256  256  512
me,nt,chunk,start,stop=  0   4  256    0  256
me,nt,chunk,start,stop=  2   4  256  512  768
me,nt,chunk,start,stop=  3   4  256  768 1024
i32: min=         0 took    0.0115400 seconds
i64: min=         0 took    0.0112960 seconds
r32: min=     0.000 took    0.0132670 seconds
r64: min=     0.000 took    0.0133660 seconds
i32: max=        36 took    0.0120030 seconds
i64: max=        36 took    0.0110880 seconds
r32: max=    36.000 took    0.0139010 seconds
r64: max=    36.000 took    0.0112640 seconds

$ ./gcc-min-max.intel
me,nt,chunk,start,stop=  1   4  256  256  512
me,nt,chunk,start,stop=  2   4  256  512  768
me,nt,chunk,start,stop=  0   4  256    0  256
me,nt,chunk,start,stop=  3   4  256  768 1024
i32: min=         0 took    0.0138681 seconds
i64: min=         0 took    0.0118332 seconds
r32: min=     0.000 took    0.0137339 seconds
r64: min=     0.000 took    0.0137799 seconds
i32: max=        36 took    0.0113640 seconds
i64: max=        36 took    0.0122621 seconds
r32: max=    36.000 took    0.0129068 seconds
r64: max=    36.000 took    0.0130620 seconds

$ ./c99-omp-min-max.gcc
me,nt,chunk,start,stop=  0   4  256    0  256
me,nt,chunk,start,stop=  1   4  256  256  512
me,nt,chunk,start,stop=  2   4  256  512  768
me,nt,chunk,start,stop=  3   4  256  768 1024
i32: min=         0 took    0.4976790 seconds
i64: min=         0 took    0.5132170 seconds
r32: min=     0.000 took    0.7187450 seconds
r64: min=     0.000 took    0.6636660 seconds
i32: max=        36 took    0.5007790 seconds
i64: max=        36 took    0.4833000 seconds
r32: max=    36.000 took    0.6805630 seconds
r64: max=    36.000 took    0.6845220 seconds

$ ./c99-omp-min-max.intel
me,nt,chunk,start,stop=  1   4  256  256  512
me,nt,chunk,start,stop=  2   4  256  512  768
me,nt,chunk,start,stop=  3   4  256  768 1024
me,nt,chunk,start,stop=  0   4  256    0  256
i32: min=         0 took    1.8755560 seconds
i64: min=         0 took    1.7328641 seconds
r32: min=     0.000 took    1.5479810 seconds
r64: min=     0.000 took    2.2241991 seconds
i32: max=        36 took    1.9648800 seconds
i64: max=        36 took    1.8810740 seconds
r32: max=    36.000 took    1.6893439 seconds
r64: max=    36.000 took    1.7421799 seconds
```
