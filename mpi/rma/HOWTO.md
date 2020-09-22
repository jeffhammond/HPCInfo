The purpose of this document is to explain the right way to use MPI-3 RMA (henceforth, RMA).
RMA is rather complicated and provides multiple window types and
multiple synchronization motifs.

Short version:
  1. Use `MPI_Win_allocate` whenever possible.
  2. Seriously, change your application memory management to satisfy (1).
  3. Use passive target shared synchronization.
  4. Lock (unlock) your windows immediately after constructing (before destructing) them.
  
This document is organized as follows:
  1. Window selection
  2. Synchronization motifs
  3. RMA operations
  4. Shared memory windows
  5. Library design (maybe)
  
Section 2 assumes that you know what RMA operations are, but if you have somehow
discovered this page without first learning about RMA operations, you may wish
to skim Section 3 before reading Section 2.

# Window Types

There are four types of windows in RMA:

Shorthand|Function name|Buffer|Related functions
---|---|---|---
Created|``MPI_Win_create``|User|
Dynamic|``MPI_Win_create_dynamic``|No|`MPI_Win_{attach,detach}`
Allocated|``MPI_Win_allocate``|Library|
Shared|``MPI_Win_allocate_shared``|Library|`MPI_Win_shared_query`

Created windows were part of MPI-2; the other three are new in MPI-3.

Created windows take a buffer argument from the user at each calling rank
and make those buffers amenable to RMA communication.
The size of the buffers can be different at each calling rank, and may be zero.

Dynamic windows are constructed without any associated buffers.
Those are associated and disassociated using `MPI_Win_{attach,detach}`.
We will provide more detail on this later.

Allocated and shared windows allocate the buffer in the MPI library
and return the address of it to the user.
This makes it possible for the MPI library to use interprocess shared memory
in the case of shared windows.
Allocated windows can use interprocess shared memory, network-registered memory, or both.

All MPI window constructors and destructors are collective.
If you need non-collective RMA allocation, there are two methods:
  1. Allocate a dynamic window and use `MPI_Win_{attach,detach}`, which are non-collective, 
     to associate buffers with that window object.
  2. Create a slab of memory with an allocated window and suballocate that to the application.
     An example of this can be found in OSHMPI [here](https://github.com/jeffhammond/oshmpi/blob/master/src/shmem-internals.c#L338).  If you are using C++, both placement new and [`std::pmr`](https://en.cppreference.com/w/cpp/memory/memory_resource) are handy here.
     
Whenever possible, use allocated windows, as these provide the best performance.
If you use RMA for interprocess shared memory, you need to use shared windows.
It should be possible to use an allocated window for this but the MPI Forum
didn't standardize that
-- see [this](https://github.com/mpi-forum/mpi-forum-historic/issues/397)
and [this](https://github.com/mpi-forum/mpi-issues/issues/23) for details --
so you can't get access to the shared memory associated with an allocated window
if it's there.

If you absolute cannot use allocated or shared windows, created windows 
are the next best option, because this allow for the network registration
associated with RDMA (e.g. Mellanox and Cray networks often support this).

Finally, and only if absolutely necessary, you can use dynamic windows.
Dynamics windows make it very hard on the MPI library to use RDMA features
unless the network supports virtual address translation
(the IBM PERCS network supported this, as one example).

If dynamic windows are so terrible, why are they there?
Dynamic windows are required for some use cases, including the one
where non-collective memory allocation and deallocation is required.
This was noted in the [paper by Bonachea and Duell](https://gasnet.lbl.gov/pubs/bonachea-duell-mpi.pdf)
in the context of Unified Parallel C (UPC).

For both created and dynamic windows, it is nearly impossible
-- exceptions are noted [here](https://github.com/mpi-forum/mpi-issues/issues/23) --
to use interprocess shared memory, which deprives your application of a nearly
universal optimization that applies to every use case involving more than one process per node.
The Argonne [Casper](https://www.mcs.anl.gov/project/casper/) project
only supports allocated windows, because it currently relies on interprocess shared memory
internally.

If you are trying to use the Partitioned Global Address Space (PGAS) model,
your usage is likely consistent with allocated windows, which is what both
OpenSHMEM-over-MPI\* ([OSHMPI](https://github.com/jeffhammond/oshmpi)) and
[ARMCI-MPI](https://github.com/pmodels/armci-mpi) do.
Fortran coarrays are allocated and deallocated colletives, so they too align
with allocated windows, which is likely how [OpenCoarrays](http://www.opencoarrays.org/)
does things (I have not looked in the source in a long time and don't know for sure).

\* OSHMPI also uses created windows, but only because OpenSHMEM supports
operations on global variables, which live in the
[data segment](https://en.wikipedia.org/wiki/Data_segment)
of a binary, which is allocated before the start of `main()`.

# Synchronization Motifs

There are three-ish types of windows in RMA:

Shorthand|Function names
---|---
BSP|``MPI_Win_fence``
PSCW|``MPI_Win_{post,start,complete,wait}``
Passive target|``MPI_Win_(un)lock(_all)``, ``MPI_Win_flush(_local)(_all)``

Side note: bulk synchronous programming is not the same as the
[Bulk Synchronous Parallel (BSP)](https://en.wikipedia.org/wiki/Bulk_synchronous_parallel)
programming model of Valiant, and anyone who uses the term BSP to describe
MPI applications that call collectives too often is not serious about computer science terminology.
If you are reviewing papers that make this error, please correct the authors, no matter
how famous they are or how large their egos may be.  They are wrong.

The only synchronization model worth using is passive target.
The use cases where BSP makes sense can use passive target.
Most uses of PSCW should use point-to-point (send-recv) instead.
In the unlikely event you invent a reason to use BSP or PSCW,
just forget about it, because the mental overhead of carrying this
around in your brain just is not worth.
Forget you ever learned about BSP or PSCW and focus your valuable
time on using passive target properly and you will be happier.

The MPI-2 passive target model was limited to `MPI_Win_(un)lock`.
Locks could be shared or exclusive.
There are limited uses for exclusive locks and I recommend you forget about
those too -- if you want mutual exclusion, do it some other way.
If we limit ourselves to shared locks, we can then focus exclusively
on ``MPI_Win_(un)lock_all``.
The template you should try to use is below:

```c
// allocate + lock_all are always together
MPI_Win_allocate(size, disp_unit, info, comm, &baseptr, &win); 
MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
```

```c
// unlock_all and free are always together
MPI_Win_unlock_all(win);
MPI_Win_free(&win); 
```

You don't have to use an allocated window here, but just get
in the habit of using passive target shared lock synchronization
for the entire lifetime of your window.

## Flushing

Once your window is ready passive target shared lock synchronization,
you will use ``MPI_Win_flush(_local)(_all)`` to synchronize operations.
The flush operations act on all of the outstanding operations on a window.
The local flushes only cause local completion, which means you can reuse
the buffers associated with the arguments.
Flushes that are not local are remote, and cause remote completion,
which means that data will be accessible at the target.

If you find that flush is too heavy a hammer for your use case, you can
use request-based RMA operations and bring about local completion
by testing on the request.
Whether this is more efficient than flushes is not obvious and likely
both implementation and network dependent.

It is natural to wonder why you can only do local completion with requests --
it was [discussed](https://lists.mpi-forum.org/pipermail/mpiwg-rma/2013-October/003124.html)
but the MPI Forum couldn't find compelling use cases.
Feel free to communicate with the MPI Forum via
GitHub [issues](https://github.com/mpi-forum/mpi-issues/issues)
if you believe this needs revisiting.

```c
int buffer = 42;
MPI_Put(&buffer, 1, MPI_INT, 0 /* target rank */, 0 /* disp */, 1, MPI_INT, win);
// Put has been initiated

MPI_Win_flush_local(win);
// buffer is now safe to modify
buffer = 0;

MPI_Win_flush(win);
// 42 is now present in the (Private*) window at the target
// * more on this private window business in Section 4

MPI_Get(&buffer, 1, MPI_INT, 0 /* target rank */, 0 /* disp */, 1, MPI_INT, win);
// for Get, local and remote completion are the same
// whether it matters which one you use depends on what else is happening (see Section 5)
#ifdef LOCAL
MPI_Win_flush_local(win);
#else
MPI_Win_flush(win);
#endif
// buffer now contains the value 42 obtained from the window at the target
```
# Operations

# Shared memory windows

Public vs private.  `MPI_Win_sync`

# Library Design

