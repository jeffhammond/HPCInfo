ifdef FRONTEND
    CC=xlc_r
    CFLAGS=-g -O5 -qhot -qipa=list="transpose.x.lst" -qlanglvl=stdc99 -qreport
    OMPFLAGS=-qsmp=omp -DOMP
    LIBS=safemalloc.o
endif
ifdef BGP
    CC=mpixlc_r
    CFLAGS=-g -O5 -qhot -qipa=list="transpose.x.lst" -qtune=450 -qarch=450d -qlanglvl=stdc99 -qreport -DMPI -DHPM
    OMPFLAGS=-qsmp=omp -DOMP
    HPM=-L/soft/apps/UPC/lib -lhpm
    LIBS=$(HPM) safemalloc.o
endif
ifdef LAPTOP
    CC=gcc
    CFLAGS=-g -O3 -std=c99 -fopenmp -ftree-vectorize -msse -msse2 -msse3 -ffast-math
#    CC=icc
#    CFLAGS=-g -std=c99 -O2 -openmp
    OMPFLAGS=-DOMP
    LIBS=safemalloc.o
endif

all: transpose.x transpose-omp.x

safemalloc.o: safemalloc.c safemalloc.h
	$(CC) $(CFLAGS) -c safemalloc.c -o safemalloc.o

transpose.x: transpose.c safemalloc.o
	$(CC) $(CFLAGS) transpose.c $(LIBS) -o transpose.x

transpose-omp.x: transpose.c safemalloc.o
	$(CC) $(CFLAGS) $(OMPFLAGS) transpose.c $(LIBS) -o transpose-omp.x

ifdef BGP
submit: all
		qsub --mode smp -n 1 -t 55 --env OMP_NUM_THREADS=1 ./transpose.x 1 550 1
		qsub --mode smp -n 1 -t 55 --env OMP_NUM_THREADS=4 ./transpose-omp.x 1 550 1
		qsub --mode smp -n 1 -t 55 --env OMP_NUM_THREADS=1 ./transpose.x 16 3072 16
		qsub --mode smp -n 1 -t 55 --env OMP_NUM_THREADS=4 ./transpose-omp.x 16 3072 16
else
submit: all
	OMP_NUM_THREADS=1 ./transpose.x     1> transpose.out     2> transpose.err
	OMP_NUM_THREADS=2 ./transpose-omp.x 1> transpose-omp.out 2> transpose-omp.err
endif

clean:
	rm -f *.o 

realclean: clean
	rm -f *.x *.lst
