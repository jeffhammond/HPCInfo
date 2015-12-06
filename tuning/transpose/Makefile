#CC=gcc
#CFLAGS=-g -O3 -std=c99 -ftree-vectorize -msse -msse2 -msse3 -ffast-math
#OMPFLAGS=-fopenmp
CC=icc
CFLAGS=-std=c99 -O3
OMPFLAGS=-qopenmp

all: transpose.x transpose-omp.x

safemalloc.o: safemalloc.c safemalloc.h
	$(CC) $(CFLAGS) -c safemalloc.c -o safemalloc.o

transpose.x: transpose.c safemalloc.o
	$(CC) $(CFLAGS) transpose.c $(LIBS) safemalloc.o -o transpose.x

transpose-omp.x: transpose.c safemalloc.o
	$(CC) $(CFLAGS) $(OMPFLAGS) transpose.c $(LIBS) safemalloc.o -o transpose-omp.x

submit: all
	OMP_NUM_THREADS=1 ./transpose.x     1> transpose.out     2> transpose.err
	OMP_NUM_THREADS=2 ./transpose-omp.x 1> transpose-omp.out 2> transpose-omp.err

clean:
	rm -f *.o 
	rm -f *.x
