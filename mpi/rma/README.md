# Key Features

* Atomics, which were not present in MPI-2 at all.
* Better synchronization: Local and remote completion are separated in MPI-3.
* Request-based completion: Some applications (e.g. NWChem) can utilize non-bulk synchronization (TODO: post the code).
* Dynamic window allocation: Collective window allocation is a show-stopper for some MPI clients (e.g. GASNet).
* Symmetric window allocation: ```MPI_Win_create``` took user memory.  ```MPI_Win_allocate``` can allocate symmetric memory internally when it is prudent and possible to do so.
* Memory model: MPI-2 RMA was not able to take full advantage of cache-coherent systems.  MPI-3 provides the '''unified''' model to address this.

Jeff Squyres' blog has [another perspective](http://blogs.cisco.com/performance/the-new-mpi-3-remote-memory-access-one-sided-interface/) that may contain more/other features.

# Mapping to other PGAS runtimes

In terms of how MPI-3 RMA maps to other PGAS runtimes, below are some approximate equivalences.  I can't assert they are all exactly right but they should be close enough for discussion purposes.

## SHMEM

See [SHMEM](https://github.com/jeffhammond/HPCInfo/blob/master/mpi/rma/SHMEM.md).

This has been implemented in [OSHMPI v1](https://github.com/jeffhammond/oshmpi) - [this paper](https://github.com/jeffhammond/oshmpi/blob/master/docs/iwosh-paper.pdf) explains the design - and [OSHMPI v2](https://github.com/pmodels/oshmpi) - [this page](https://www.mcs.anl.gov/project/oshmpi/) explains the design.

## ARMCI

http://wiki.mpich.org/armci-mpi/index.php/Main_Page and content linked therefrom address everything.

## Global Arrays

Unlike ARMCI, GA uses opaque data handles and thus all of the problems w.r.t. mapping ARMCI to MPI-RMA disappear.

http://www.mcs.anl.gov/research/projects/mpi/usingmpi2/ outlines an implementation of Global Arrays over MPI-2 RMA.  The primary issue seen there is the lack of remote atomic operations, which were added in MPI-3 RMA.

## GASNet

MPI-3 RMA addresses most/all of the criticisms in [http://www.cs.berkeley.edu/~bonachea/upc/bonachea-duell-mpi.pdf Bonachea and Duell].

MPI-3 does not provide active-messages but they have been implementable since MPI-2.1 using, e.g., Pthreads and Send-Recv (see https://github.com/jeffhammond/HPCInfo/tree/master/mpi/active-messages for a crude example).  This - or intermittent polling - is how the GASNet MPI conduit, Charm++ and MADNESS all implement active messages over MPI 2-sided.

## UPC

The UPC runtime doesn't require active-messages although they are used in the BUPC runtime, which is why GASNet provides them.  The lack of any need for active-messages makes MPI-3 RMA a good match for UPC, at least in theory.  It is possible that one-sided memory allocation is an issue but with dynamic windows and the aid of a compiler that can insert intermittent polling, this should not be a serious issue.

## CAF

Because CAF does memory allocation collectively (and symmetrically), MPI-3 RMA is a good candidate for a CAF runtime.
