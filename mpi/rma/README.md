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

### Issues

* There is not a portable way to export the stack for MPI-3 RMA but one can theoretically do it collectively with ```MPI_Win_create```.  We assume that an implementation of SHMEM over MPI-3 RMA that would support communication to remote stack memory would have compiler support or non-standard extensions.  It is possible to ''cheat'' on some systems when it is known that remote registration is not ''required'' (i.e. RMA sits on top of active-message primitives or an RDMA interface that can accept remote virtual addresses).

* Instead of a static, pre-allocated symmetric heap (as shown below), one can allocate the symmetric heap dynamically using dynamic windows and attaching (via ```MPI_Win_attach```) memory allocated symmetrically using ```mmap(MAP_FIXED,..)``` or using ```MPI_Win_create```, but both will require allocation and perhaps deallocation to be collective, which is inconsistent with Cray SHMEM (at least).  There are more complicated ways to implement the symmetric heap in a manner consistent with vendor implementations of SHMEM that are limited only by the creativity of the implementer, not the semantics of MPI-3.

* The optimal implementation of SHMEM depends on the nature of the MPI implementation, in particular whether or not relaxing ordering of RMA has an performance benefits.  Some comments in the code allude to this.

* OpenSHMEM does not have a finalize routine (Cray SHMEM does), which is a serious problem if ```MPI_Finalize``` is needed to cleanup network resources, etc.  It should be possible to use [atexit](http://linux.die.net/man/3/atexit) to implement a proper collective cleanup routine but this violates some SHMEM users' expectations that SHMEM termination will always be non-collective.  Alternatively, if it is reasonable to terminate MPI with ```MPI_Abort(MPI_COMM_SELF,0)```, then that is an alternative way to achieve the desired behavior (assuming ```MPI_Abort``` is properly acting only on the provided communicator).

### Global State

This is ```shmem-globals.h```
```
#ifndef SHMEM_GLOBALS_H
#define SHMEM_GLOBALS_H

#include <stdint.>
#include <stddef.h>
#include <mpi.h>

const size_t default_symm_heap_size = 1e9;
/* actual size of symm heap */
size_t symm_heap_size;
/* pre-allocate symm heap and track only offset */
ptrdiff_t symm_heap_offset;

/* the symmetric heap is one window */
extern MPI_Win symm_heap;
/* this is my base address for the symm heap */
void * mybase;
/* in the general case, baseptrs are not symmetric;
 * address translation requires O(nproc) array */
void * allbase[];

/* cache world size and rank; need TLS is MPI procs = OS threads */
int world_size, world_rank;

/* declaration of internal functions */
MPI_Aint translate_remote_address_to_sheap_disp(int pe, void * address);

#endif
```

### Initialization
```
#include "shmem-globals.h"

MPI_Win symm_heap;

/* OpenSHMEM doesn't define shmem_init; that's in Cray SHMEM... */
void start_pes(int npes);
{
  if (!MPI_Initialized() )
    MPI_Init_thread(..);

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  /* we could workaround this with a new communicator... */
  if (world_size!=npes)
    MPI_Abort(MPI_COMM_WORLD, 1);

  /* query the environment for the symm heap size provided by user */
  char * c = getenv("SHMEM_SYMMETRIC_HEAP_SIZE");
  symm_heap_size = (c!=NULL) ? atol(c) : default_symm_heap_size;

  /* this overrides the default ordering semantics to enable 
   * the implementation to provide better performance (in theory). 
   * see comment in shmem_fence() for an alternative strategy. */
  MPI_Info shmem_info;
  MPI_Info_create(&shmem_info);
  /* see MPI-3 11.7.2 - this disables all ordering for accumulates, 
   * which are used for word-atomic put/get */
  MPI_Info_set(shmem_info, "accumulate_ordering", "");
  /* see MPI-3 11.2.1 - this indicates we are only going 
   * to use MPI_NO_OP or MPI_REPLACE */
  MPI_Info_set(shmem_info, "accumulate_ops", "same_op_no_op,");

  /* win_allocate is able to use symmetric memory when available in HW */
  MPI_Win_allocate( (MPI_Aint)symm_heap_size, sizeof(long), shmem_info,
                   MPI_COMM_WORLD, &mybase, &symm_heap);

  /* sheap malloc increments this */
  symm_heap_offset = 0;

  /* for remote address translation...
  MPI_Alloc_mem( (MPI_Aint)world_size, MPI_INFO_NULL, &allbase);
  MPI_Allgather(&mybase, sizeof(void*), MPI_BYTE, 
                allbase, sizeof(void*), MPI_BYTE, MPI_COMM_WORLD);

  /* expose data to all PEs for shared access */
  MPI_Win_lock_all(symm_heap);

  return;
}
```

### Utility Functions

```
#include "shmem-globals.h"

int _num_pes(void)
{
  return world_size;
}

int shmem_n_pes(void)
{
  return world_size;
}

￼int _my_pe(void)
{
  return world_rank;
}

int shmem_my_pe(void)
{
  return world_rank;
}
```

### Address Translation

```
#include "shmem-globals.h"

/* this should be a macro or static-inline for performance */
MPI_Aint translate_remote_address_to_sheap_disp(int pe, void * address)
{
  MPI_Aint disp;
 
  /* ignore possible issues related to pointer arithmetic */
  disp = address - allbase[pe];

  /* verify remote access is within symm_heap 
   * no-debug mode should disable this check */
  if (disp<0 || disp>symm_heap_size)
    MPI_Abort(MPI_COMM_WORLD, 1);

  return disp;
}
```

=== Symmetric Heap Allocation ===

Implementing a proper memory allocator is not trivial.  Instead, for illustrative purposes we use a stack allocator that only moves one way and abort when the memory is exhausted.

```
#include "shmem-globals.h"

void *shmalloc(size_t size)
{
  void * ptr = mybase + symm_heap_offset;
  symm_heap_offset += size;
  if (symm_heap_offset>symm_heap_size)
    MPI_Abort(MPI_COMM_WORLD, 1);
  return ptr;
}
```

### Put and Get
```
#include "shmem-globals.h"

void shmem_long_put(long *target, const long *source, size_t nelems, int pe)
{
  /* instead of chunking messages too large for MPI, we just abort */
  if (nelems>(size_t)INT32_MAX)
    MPI_Abort(MPI_COMM_WORLD, 1);

  /* convert the remote address to an offset in the symm_heap */
  MPI_Aint disp = translate_remote_address_to_sheap_disp(target);

  /* enqueues the Put - implementation may or may not start immediately */
  MPI_Put( (const void*)source, nelems, MPI_LONG, pe, disp, nelems, MPI_LONG, symm_heap);

  /* this causes local completion of the Put */
  MPI_Win_flush_local(pe, symm_heap);

  return;
}
```

### Atomics

SHMEM atomics are blocking.

```
#include "shmem-globals.h"

long shmem_long_swap(long *target, long value, int pe)
{
  long result;

  MPI_Aint disp = translate_remote_address_to_sheap_disp(target);

  MPI_Fetch_and_op(&value, &result, MPI_LONG, pe, disp, MPI_REPLACE, symm_heap);

  MPI_Win_flush(pe, symm_heap);

  return result;
}

long shmem_long_cswap(long *target, long cond, long value, int pe)
{
  long result;

  MPI_Aint disp = translate_remote_address_to_sheap_disp(target);

  MPI_Compare_and_swap(&value, &cond, &result, MPI_LONG, pe, disp, symm_heap);

  MPI_Win_flush(pe, symm_heap);

  return result;
}
```

### Synchronization
```
#include "shmem-globals.h"

/* if instead of using shmem_info above to turn off ordering,
   we left ordering enabled, then this call would be a no-op */
void shmem_fence(void)
{
  MPI_Win_flush_all(symm_heap);
  return;
}

void shmem_quiet(void)
{
  MPI_Win_flush_all(symm_heap);
  return;
}

void shmem_barrier_all(void)
{
  MPI_Win_flush_all(symm_heap);
  MPI_Barrier(MPI_COMM_WORLD);
  return;
}
```

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
