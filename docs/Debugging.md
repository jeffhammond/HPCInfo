# GDB with MPI

GDB is hard to use with MPI because of the interactivity.  It is not always possible to use the well-known `xterm` trick described [here](https://www.open-mpi.org/faq/?category=debugging).

The better option is to use non-interactive GDB.  There is a short example [here](http://matetelki.com/blog/?p=456), which shows:
```sh
gdb EXECUTABLE \
-ex "set width 1000" \
-ex "thread apply all bt" \
-ex run \
-ex bt \
-ex "set confirm off" \
-ex quit
```

TODO

# GDB non-interactive

```sh
gdb \
-ex "set width 1000" \
-ex "thread apply all bt" \
-ex run \
-ex bt \
-ex "set confirm off" \
-ex quit \
--args ./build/pennant test/leblancbig/leblancbig.pnt
```

# LLDB

I've found that lldb works a lot better on Mac, which is not too surprising, since that is what Apple ships.  I recall that I need to compile with "-Wl,-pie" to get the best debugging experience.

The commands for LLDB are a bit different from GDB but they are intuitive (http://lldb.llvm.org/tutorial.html).

I'm not sure if it helps, but this is what I observed just now...

I built my code with these options:
```
$ make CC=/opt/mpich/dev/gcc/default/bin/mpicc CFLAGS="-g -O2 -Wall -Wl,-pie"
```

MPICH is included as a shared library:
```
$ otool -L win_fence.x
win_fence.x:
/opt/mpich/dev/gcc/default/lib/libmpi.0.dylib (compatibility version 1.0.0, current version 1.0.0)
/opt/mpich/dev/gcc/default/lib/libpmpi.0.dylib (compatibility version 1.0.0, current version 1.0.0)
/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1226.10.1)
/usr/local/lib/gcc/6/libgcc_s.1.dylib (compatibility version 1.0.0, current version 1.0.0)
```

MPICH is built with GCC 6.1.0 with standard options (and debug symbols):
```
$ /opt/mpich/dev/gcc/default/bin/mpicc -show
gcc-6 -g -Wl,-flat_namespace -Wl,-commons,use_dylibs -I/opt/mpich/dev/gcc/default/include -L/opt/mpich/dev/gcc/default/lib -lmpi -lpmpi
```

```
$ /opt/mpich/dev/gcc/default/bin/mpichversion
MPICH Version:    3.3a1
MPICH Release date: unreleased development copy
MPICH Device:    ch3:nemesis
MPICH configure: CC=gcc-6 CXX=g++-6 FC=gfortran-6 F77=gfortran-6 --enable-cxx --enable-fortran --enable-threads=runtime --enable-g=dbg --with-pm=hydra --prefix=/opt/mpich/dev/gcc/default --enable-wrapper-rpath --disable-static --enable-shared
MPICH CC: gcc-6    -g -O2
MPICH CXX: g++-6   -g
MPICH F77: gfortran-6   -g
MPICH FC: gfortran-6   -g
MPICH Custom Information:
```

The following shows the debug symbols including line numbers into MPICH source:
```
$ lldb ./win_fence.x
(lldb) target create "./win_fence.x"
Current executable set to './win_fence.x' (x86_64).
(lldb) breakpoint set -n MPIR_Barrier_impl
Breakpoint 1: where = libpmpi.0.dylib`MPIR_Barrier_impl + 1 at barrier.c:306, address = 0x000000000000af71
(lldb) run
Process 87575 launched: './win_fence.x' (x86_64)
SUCCESS!
Process 87575 stopped
* thread #1: tid = 0x2c927c, 0x0000000100144f71 libpmpi.0.dylib`MPIR_Barrier_impl(comm_ptr=0x0000000100391c60, errflag=0x00007fff5fbfef5c) + 1 at barrier.c:306, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
    frame #0: 0x0000000100144f71 libpmpi.0.dylib`MPIR_Barrier_impl(comm_ptr=0x0000000100391c60, errflag=0x00007fff5fbfef5c) + 1 at barrier.c:306
   303 int MPIR_Barrier_impl(MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag)
   304 {
   305     int mpi_errno = MPI_SUCCESS;
-> 306     if (comm_ptr->coll_fns != NULL && comm_ptr->coll_fns->Barrier != NULL)
   307     {
   308 /* --BEGIN USEREXTENSION-- */
   309 mpi_errno = comm_ptr->coll_fns->Barrier(comm_ptr, errflag);
(lldb) bt
* thread #1: tid = 0x2c927c, 0x0000000100144f71 libpmpi.0.dylib`MPIR_Barrier_impl(comm_ptr=0x0000000100391c60, errflag=0x00007fff5fbfef5c) + 1 at barrier.c:306, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
  * frame #0: 0x0000000100144f71 libpmpi.0.dylib`MPIR_Barrier_impl(comm_ptr=0x0000000100391c60, errflag=0x00007fff5fbfef5c) + 1 at barrier.c:306
    frame #1: 0x0000000100276788 libpmpi.0.dylib`MPID_Win_free(win_ptr=0x00007fff5fbfef88) + 328 at mpidi_rma.c:181
    frame #2: 0x00000001000655f1 libmpi.0.dylib`MPI_Win_free(win=0x00007fff5fbfeff0) + 513 at win_free.c:119
    frame #3: 0x0000000100000e20 win_fence.x`main(argc=<unavailable>, argv=<unavailable>) + 336 at win_fence.c:64
    frame #4: 0x00007fff8afd45ad libdyld.dylib`start + 1
```

# LLDB non-interactive

This runs non-interactively and does a backtrace on failure, but if the job doesn't fail, the backtrace fails with `error: invalid thread`, in which case you have to kill it.  I think this is mostly user error.
```
mpicc -g3 -Wl,-pie bad.cc
mpirun -n 2 \
lldb ./a.out \
--one-line 'run' \
--one-line-on-crash 'bt' \
--one-line 'quit'
```
