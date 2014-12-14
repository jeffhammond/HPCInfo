# External Resources

## The MPI Standard

The MPI standard is a formal specification document that is backed by the [http://mpi-forum.org/ MPI Forum], but no official standardization body (e.g. [http://www.iso.org/ISO] or [http://www.ansi.org/ ANSI].

Please see [http://mpi-forum.org/docs/docs.html this page] for all of the MPI standardization documents, including the latest version of the standard, [http://mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf MPI 3.0].

## Tutorials

The [https://computing.llnl.gov/tutorials/mpi/ LLNL tutorial] is excellent.  The Internet has too many MPI-related tutorials to list them all here.  Search engines can help you find some of them.

## Books

I personally recommend [http://www.mcs.anl.gov/research/projects/mpi/usingmpi/ Using MPI] and [http://www.mcs.anl.gov/research/projects/mpi/usingmpi2/index.html Using MPI-2] as a means for learning both the basic and advanced features of MPI.

I recall that [http://www.cs.usfca.edu/~peter/ppmpi/ Peter Pacheco's book] was good, but I lost my copy many years ago and can't certify that my recollection is accurate.

## Profiling

External:
* http://mpip.sourceforge.net/

Internal:
* https://svn.mcs.anl.gov/repos/mpi/mpich2/trunk/src/util/mem/trmem.c

# Basic MPI

These are some very simple programs that start with a basic "Hello, world!" program, demonstrate broadcast and reduce, then bring these together to do the (in)famous Monte Carlo computation of Pi.

See ```mpi/basic``` in the repo.

# Neighbor exchange

See ```mpi/intermediate``` in the repo.

# Hybrid MPI

See [https://www.ieeetcsc.org/activities/blog/challenges_for_interoperability_of_runtime_systems_in_scientific_applications Challenges for Interoperability of Runtime Systems in Scientific Applications] for some commentary on using MPI and threads, among other things.

## MPI and threads

### MPI and OpenMP

The most common usage of OpenMP is fork-join, in which case <tt>MPI_THREAD_FUNNELED</tt> is sufficient.

### MPI and Pthreads

If multiple threads make MPI calls, <tt>MPI_THREAD_SERIALIZED</tt> or  <tt>MPI_THREAD_MULTIPLE</tt> is required, the former if the user code implements mutual exclusion between different thread's access to MPI and the latter if the MPI implementation is expected to do this.

### Example code

This is a simple example that prints out information about MPI, Pthreads and OpenMP used together.

You should submit this job with different values of the environment variables <tt>POSIX_NUM_THREADS</tt> and <tt>OMP_NUM_THREADS</tt>.  On Blue Gene/Q, this test will output where the threads are executing.  On other systems, the hardware affinity information is null.

See ```mpi/with-threads``` in the repo.

## MPI and PGAS languages

The challenge of interoperability between MPI and [[Main_Page#PGAS_models|PGAS languages]] such as [[CAF]] and [[UPC]] is the subject of ongoing research.

One challenge is the mapping between MPI processes, which are not required to be OS processes but are in most implementations, and CAF images, UPC threads or SHMEM processing elements, respectively.  The canonical usage is for a 1-to-1 mapping, although Dinan and coworkers have discussed alternative models for MPI-UPC interoperability in their recent paper (need citation).

# Intermediate MPI

TODO: Subcommunicators...

TODO: Datatypes...

# One-sided MPI

See ```mpi/one-sided``` in the repo.

# Active Messages

See ```mpi/active-messages``` in the repo.

# Advanced MPI

TODO: Generalized requests...

# MPI and C++

https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/340 fixes important issues with the use of C++ datatypes in MPI.

See http://blogs.cisco.com/performance/the-mpi-c-bindings-what-happened-and-why/ and the follow-up for details on what has changed in MPI-3 with respect to C++ support in MPI.

# Performance Considerations

## Asynchronous Progress

Asynchronous progress is essential for performance when using one-sided communication.  Get and Accumulate operations may wait for the target process to make an MPI call before communication can begin.  Additionally, nonblocking send and receive may not match until this happens, thus delaying communication when using the rendezvous protocol (for larger messages).

### MPICH

* Set <tt>MPICH_ASYNC_PROGRESS=1</tt> in your execution environment.

This will spawn a thread for every process and thus you should have one or more spare cores to use this option effectively.

### Blue Gene/P

* Set <tt>DCMF_INTERRUPTS=1</tt> in your execution environment.

This will increase the latency of MPI-1 operations by a noticeable amount but it is essential for asynchronous progress in nonblocking send and receive and especially one-sided communication.

### Blue Gene/Q

* Use the non-legacy variant of MPI.
* Set <tt>PAMID_THREAD_MULTIPLE=1</tt> in your execution environment.

See [[Mira MPI Documentation#Official MPI variants|Blue Gene/Q MPI variants]] for more information.

Unlike most other platforms, asynchronous progress is not going to reduce performance of synchronous communication significantly and may increase the overall performance of MPI by a significant amount.

### MVAPICH2

I cannot find any mention of these options in the latest documentation so I don't know if they are still accurate.

* Recompile with <tt>--enable-async-progress</tt>.
* Set <tt>MPICH_ASYNC_PROGRESS=1</tt> in your execution environment.

### Cray

This should apply to XE, XK and XC systems, but please verify with the Cray documentation for your system.

* Set <tt>MPICH_NEMESIS_ASYNC_PROGRESS=1</tt>. 
* Set <tt>MPICH_MAX_THREAD_SAFETY=multiple</tt>.
* Launch your job with <tt>aprun -n nproc -r 1 ./a.out</tt> to reserve a hardware thread for the communication thread.

See e.g. [https://cug.org/proceedings/attendee_program_cug2012/includes/files/pap115-file2.pdf Toward MPI Asynchronous Progress on Cray XE Systems] for more information.

### OpenMPI

* Use <tt>MPI_THREAD_MULTIPLE</tt> (this may not work reliably on every platform).
* See documentation for details.  I know far less about OpenMPI as I do about MPICH and its derivatives.

## Blue Gene/P

See http://www.alcf.anl.gov/user-guides/bgp-tuning-mpi-bgp.

## Blue Gene/Q

See https://www.alcf.anl.gov/user-guides/tuning-mpi-bgq.

# Implementation Details

TODO: Describe eager versus rendezvous protocols...

## MPICH2

### Building from Source

See http://wiki.mpich.org/mpich/index.php/Git.

### Implementation Details

See [http://wiki.mcs.anl.gov/mpich2/index.php/Category:Design_Documents MPICH2 Design Documents].

## MVAPICH2

### Important Usage Information

See [[MVAPICH2 and processor affinity]] for important information about using hybrid (i.e. MPI+Threads) programming models with MVAPICH.
