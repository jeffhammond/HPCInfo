Welcome to the HPCInfo wiki by [Jeff Hammond](http://jeffhammond.github.io/).

I need to move most of the content to http://jeffhammond.github.io/HPCInfo/ and make a proper website but for now we are just cloning https://wiki.alcf.anl.gov/parts/index.php and slowly converting from MediaWiki to Markdown.  I apologize for broken links.  Not all the content can be trivially migrated from the ALCF wiki site.

One can obtain an offline version of this Wiki with ``git clone git@github.com:jeffhammond/HPCInfo.wiki.git HPCInfo.wiki``.

## Programmer Resources

### Parallelism

Parallelism is easier to find than most people think.  While [[Data Parallelism]] can be difficult, it is often the case that serial codes can be made to run faster using [[Task Parallelism]].

[Algorithms for Scalable Synchronization on Shared-Memory Multiprocessors](http://www.cs.rochester.edu/research/synchronization/pseudocode/ss.html)

### Performance Tuning

https://github.com/jeffhammond/HPCInfo/tree/master/tuning/transpose is a tutorial I've used in ALCF workshops in the past.

http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/gemm/index.html

### External Links

* [Fortran information](http://fortran90.org/)
* [How To Write Fast Numerical Code: A Small Introduction](http://spiral.ece.cmu.edu:8080/pub-spiral/abstract.jsp?id=100)
* [Is Parallel Programming Hard, And, If So, What Can You Do About It?](http://kernel.org/pub/linux/kernel/people/paulmck/perfbook/perfbook.html)
* [Fast Barrier for x86 Platforms](http://www.spiral.net/software/barrier.html)
* [Concurrency Kit](http://concurrencykit.org/index.html)
* [How to Use the restrict Qualifier in C](http://dsc.sun.com/solaris/articles/cc_restrict.html)
* [Demystifying The Restrict Keyword](http://cellperformance.beyond3d.com/articles/2006/05/demystifying-the-restrict-keyword.html)
* [Restricted Pointers in C](http://www.lysator.liu.se/c/restrict.html)
* [Kaz's links for low-level hackers](http://www.mcs.anl.gov/~kazutomo/links.html)
* [Agner Fog's Software optimization resources](http://www.agner.org/optimize/)
* [The GotoBLAS/BLIS Approach to Optimizing Matrix-Matrix Multiplication](http://wiki.cs.utexas.edu/rvdg/HowToOptimizeGemm/)
* LLNL's [Introduction to Parallel Computing](https://computing.llnl.gov/tutorials/parallel_comp/ )
* [John McCalpin's blog](http://blogs.utexas.edu/jdm4372/)
* [Roscoe Bartlett's links on C++ Software Engineering](http://web.ornl.gov/~8vt/readingList.html)

## Software

### Applications

[NWChem](NWChem.mediawiki) is a massively parallel quantum chemistry code that supports a wide variety of methods.

[MPQC](MPQC.mediawiki) is a massively parallel quantum chemistry code that uses portable software (MPI and Pthreads) and supports a limited range of methods (DFT and MP2).  MPQC also supports explicitly correlated ("R12") methods.

[Dalton](Dalton-2.0.md) is a legacy quantum chemistry code that has rich functionality for molecular properties.

[LAMMPS](LAMMPS.mediawiki) is a widely-used, general purpose molecular dynamics code that runs on supercomputers.

[Coupled cluster](Coupled-cluster.md) methods are of great interest to me.  I started writing this page for someone who wanted to learn about them.

### Libraries

[Elemental](Elemental.md) is a modern, parallel, dense linear algebra library.

### Compilers

[LLVM](LLVM.md)

[Preprocessor Macros](Preprocessor-Macros.md)

### Build Systems

[Autotools](../buildscripts/Autotools.md)

### Performance Tools and Debuggers

* My favorite tool is the Kernighan debugger ("The most effective debugging tool is still careful thought, coupled with judiciously placed print statements." - Brian Kernighan in "Unix for Beginners"), not because I like print statements, but because it is too easy to avoid careful thought when using fancy tools.
* I find there is still nothing better than GDB and Valgrind for debugging tricky errors.
* For performance measurement, I use [TAU](http://tau.uoregon.edu), gprof and HPM (relevant only on Blue Gene).

## Programming Models and Runtimes

### The standard models

[MPI](../mpi) is far and away the most popular parallel programming and is used in more than 99% of the parallel applications that run on modern supercomputers and clusters.

[Shared memory](../posix/shm) (between processes) of the POSIX variety is not widely used by application programmers but is widely used in system software.

While Charm++ is not widely used in the way that MPI is, NAMD is arguably the most widely used code in all of open science, hence one can argue that Charm++ is used by the tens of thousands of people running NAMD.

### PGAS models

Note that I conflate PGAS with one-sided despite understanding the difference between the two.  If a one-sided model takes remote addresses as an argument, that's close enough to PGAS for my purposes.

[GA/ARMCI](../ga-armci) and [SHMEM](../shmem) are library implementations of the PGAS programming model.

[UPC](../upc) and [CAF](../coarray-f) are C-based and Fortran-based PGAS programming languages.

[Implementing a Symmetric Heap](../mpi/advanced) is related to implementing PGAS models efficiently.

[MPI3-RMA](../mpi/rma) provides all of the communication features required by ''most'' PGAS models.

### Threading models

#### Library-Based Models

[Pthreads](../posix/threads) is common for systems programming and for codes like MADNESS and MPQC that require a more dynamic threading model than [OpenMP](../openmp) provides.  Using Pthreads from C and C++ is straightforward but not so much from other languages (e.g. Fortran).

Intel Thread Building Blocks (TBB), provides dynamic and static parallelism for C++, along with a number of helpful utility features, such as multidimensional iterators.  TBB is available as OSS from Intel and it is quite portable and runs on a number of non-x86 platforms, including Blue Gene systems.

[OpenCL](../opencl) is an industry-standard interface for GPUs.

#### Language Extensions/Directives

[OpenMP](../openmp) is probably the most common threading model for scientific applications.  It is primarily a fork-join model and well-suited for data parallelism.  Implementing more complex parallel motifs can be more challenging.

[OpenACC](../openacc) is a directive-based model that resembles OpenMP, but explicitly targets accelerators.  At least some of the features of OpenACC will be part of OpenMP 4.

### Machine-specific models

[DCMF](../dcmf) was for Blue Gene/P.

[PAMI](../pami) is for [Blue Gene/Q](https://wiki.alcf.anl.gov/parts/index.php/Blue_Gene/Q) and IBM PERCS.

[DMAPP](../dmapp) is for [Cray](Cray.md) systems.

[CUDA](../cuda) is the best way to get performance out of an NVIDIA GPU.  Do not let anyone tell you otherwise.

## Supercomputers

Blue Gene systems are developed by IBM.

The [K computer](K-computer.mediawiki) was developed by Fujitsu.

Recent [Cray](Cray.md) supercomputers include the XT, XE, XK, and XC series.

[Allocations](Allocations.md) is my page on how to get access to supercomputers (for free).

[Mac](Mac.md) is not a supercomputer by any means but a lot of people use it for development.
