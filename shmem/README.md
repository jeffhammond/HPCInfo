## Implementation

I wrote OpenSHMEM over MPI-3 [OSHMPI](https://github.com/jeffhammond/oshmpi) to produce an ultra-portable implementation of SHMEM for distributed-memory systems.  It runs anywhere that has Linux and MPI-3 (MPICH, MVAPICH and CrayMPI all support this now).

## Compatibility

On Cray XE6, [OpenSHMEM](http://openshmem.org/) 1.0 was not supported when I ran the tests provided in this repo.  The Cray SHMEM semantics are a bit different, and frankly better than OpenSHMEM 1.0.  A future OpenSHMEM specification is likely to include similar improvements.

This header fixes the compatibility issues by implementing Cray SHMEM calls in terms of OpenSHMEM ones, should one use this code on a non-Cray system that has OpenSHMEM.

## Examples

See ./shmem in this repo.

## Benchmarks

* http://mvapich.cse.ohio-state.edu/benchmarks/ (scroll down or search for SHMEM)
* https://svn.mcs.anl.gov/repos/performance/benchmarks/randomaccess/hpcc/shmem
* http://hpcrl.cse.ohio-state.edu/wiki/index.php/UTS
* http://sourceforge.net/p/uts-benchmark/wiki/Home/
* https://github.com/perarnau/uts
* http://www.csm.ornl.gov/essc/c/uts-preview-ornl/

## Relationship to MPI-3 RMA

See [[MPI3-RMA#SHMEM]] or the OSHMPI source noted above.
