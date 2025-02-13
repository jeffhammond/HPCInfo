# Linux

```
gcc -fPIC -shared extern2.c -o libxxx.so && gfortran extern.F90 libxxx.so -o extern && ./extern
MPIR_F08_MPI_IN_PLACE=0 &MPIR_F08_MPI_IN_PLACE=0x56f23c500014 &MPIR_F08_MPI_IN_PLACE=95598393950228
 LOC(MPI_IN_PLACE)=       95598393950228
 LOC(buf)=      140733589387360
sendbuf=0x56f23c500014, sendbuf=95598393950228
sendbuf is MPI_IN_PLACE? yes
recvbuf=0x7fff179a7860, recvbuf=140733589387360
*count=1, *datatype=2, *op=3, *comm=4
         911
```

# MacOS

```
gcc -fPIC -shared extern2.c -o libxxx.so && gfortran -c extern.F90 && ld extern.o libxxx.so -L/opt/homebrew/Cellar/gcc/13.2.0/lib/gcc/current/  -lgfortran -commons use_dylibs -o extern && ./extern
MPIR_F08_MPI_IN_PLACE=0 &MPIR_F08_MPI_IN_PLACE=0x104d50000 &MPIR_F08_MPI_IN_PLACE=4376035328
 LOC(MPI_IN_PLACE)=           4376035328
 LOC(buf)=           6091272800
sendbuf=0x104d50000, sendbuf=4376035328
sendbuf is MPI_IN_PLACE? yes
recvbuf=0x16b117260, recvbuf=6091272800
*count=1, *datatype=2, *op=3, *comm=4
         911
```
