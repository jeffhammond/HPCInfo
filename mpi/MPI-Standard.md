# Overview

This is Jeff's TODO list for the MPI Forum, as opposed to any implementation.  For that, see [[MPICH]].

## MPI Forum

* [Meeting details](http://meetings.mpi-forum.org/Meeting_details.php)
* [Working groups](https://svn.mpi-forum.org/trac/mpi-forum-web/wiki)
* [Mailing lists](http://lists.mpi-forum.org/)
* [Tickets I reported](https://svn.mpi-forum.org/trac/mpi-forum-web/query?status=!closed&reporter=jhammond)
* [Active tickets I follow](https://svn.mpi-forum.org/trac/mpi-forum-web/query?status=!closed&cc=~jhammond)

[Hybrid Working Group](https://svn.mpi-forum.org/trac/mpi-forum-web/wiki/MPI3Hybrid)

# Asynchrony

## RMA

* [RMA <tt>same_disp</tt> key (#369)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/369)
* [RMA needs new assertions for passive-target epochs (#396)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/396)
* [extend the use of <tt>MPI_WIN_SHARED_QUERY</tt> to all windows (#397)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/397)
* [request-based remote completion for RMA (#398)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/398)
* [generalize <tt>same_op_no_op</tt> and allow user to specify all ops to be used (#399)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/399)

## Generalized Requests and Active Messages

* [MPICH-style generalized requests](https://svn.mpi-forum.org/trac/mpi-forum-web/wiki/Proposal)
* Active messages were DOA in MPI 3.0 due to priority of RMA.  Now that RMA is done, maybe we can achieve something useful here.

# Datatypes and Reductions

## Better Reductions

* [Extend predefined MPI_Op's to user defined datatypes composed of a single, predefined type (#34)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/34)
* [<tt>MPI_Accumulate</tt>-style Behavior For Predefined Reduction Operations (#338)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/338) - closely related to #34
* [<tt>MPI_Reduce_local</tt> symmetry w.r.t. <tt>MPI_IN_PLACE</tt> (#353)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/353) - withdrawn (pointless)
* [receive and reduce - new p2p function (#407)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/407) - See https://svn.mpi-forum.org/trac/mpi-forum-web/wiki/PtpWikiPage/notes-2014-02-03

## New Built-in Types

* [New quad precision predefined data types (#318)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/318)
* [Longer types for use with <tt>MPI_MINLOC</tt> and <tt>MPI_MAXLOC</tt> (#319)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/319)
* [<tt>MPI_{MAX,MIN}LOC</tt> needs to work with more pairs (#342)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/342) (redundant with #319)

## Datatype Functions

* [Clarify what functions can be called on uncommitted datatypes (#356)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/356)
* [MPI datatype info keys (#382)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/382)

# Other

## Info Objects

* [Support for integer info keys (#370)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/370) - Better to have general type support.

## Comm_split types

* [Add <tt>Comm_split_typed</tt> keys other than SHARED (#372)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/372) - See notes in my printed copy of MPI-3.0 as well.
* [<tt> MPI_COMM_TYPE_NEIGHBORHOOD</tt> (#297)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/297).  Closely related to 372.

## Thread Support

* [Trivial text fix in <tt>MPI_Init_thread</tt> (#352)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/352)
* [Always thread-safe query functions (#357)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/357) - This is still being debated in the Hybrid WG.
* [Require <tt>MPI_THREAD_MULTIPLE</tt> (#371)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/371) - Unlikely to pass any time soon.
* [Generalize thread-per-comm to thread-per-object (#373)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/373) - Seems unlikely to pass any time soon.

## Subsetting on Communicators

* [Communicator info keys (#381)](https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/381) - Performance-oriented disabling of wildcards on communicators.

## Memory Allocation

* <tt>MPI_Alloc_mem</tt> could use some info arguments (e.g. "allocate at this address if possible").
