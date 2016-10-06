all: 
	#$(MAKE) -C assembly
	$(MAKE) -C atomics
	$(MAKE) -C c++11
	$(MAKE) -C c11
	$(MAKE) -C c++98
	$(MAKE) -C c99
	#$(MAKE) -C chapel
	$(MAKE) -C cilk
	$(MAKE) -C coarray-f
	#$(MAKE) -C cuda
	#$(MAKE) -C dcmf
	#$(MAKE) -C dlang
	#$(MAKE) -C dmapp
	$(MAKE) -C fortran
	$(MAKE) -C ga-armci
	#$(MAKE) -C gpi
	#$(MAKE) -C hugetlb
	$(MAKE) -C lapack
	#$(MAKE) -C linux-cma
	$(MAKE) -C mpi
	#$(MAKE) -C ofi
	#$(MAKE) -C openacc
	$(MAKE) -C openmp
	#$(MAKE) -C pami
	$(MAKE) -C posix
	#$(MAKE) -C shmem
	$(MAKE) -C tbb
	#$(MAKE) -C timing
	#$(MAKE) -C topology
	#$(MAKE) -C tuning
	#$(MAKE) -C upc

clean: 
	#$(MAKE) -C assembly clean
	$(MAKE) -C atomics clean
	$(MAKE) -C c++11 clean
	$(MAKE) -C c11 clean
	$(MAKE) -C c++98 clean
	$(MAKE) -C c99 clean
	#$(MAKE) -C chapel clean
	$(MAKE) -C cilk clean
	$(MAKE) -C coarray-f clean
	#$(MAKE) -C cuda clean
	#$(MAKE) -C dcmf clean
	#$(MAKE) -C dlang clean
	#$(MAKE) -C dmapp clean
	$(MAKE) -C fortran clean
	$(MAKE) -C ga-armci clean
	#$(MAKE) -C gpi clean
	#$(MAKE) -C hugetlb clean
	$(MAKE) -C lapack clean
	#$(MAKE) -C linux-cma clean
	$(MAKE) -C mpi clean
	#$(MAKE) -C ofi clean
	#$(MAKE) -C openacc clean
	$(MAKE) -C openmp clean
	#$(MAKE) -C pami clean
	$(MAKE) -C posix clean
	#$(MAKE) -C shmem clean
	$(MAKE) -C tbb clean
	#$(MAKE) -C timing clean
	#$(MAKE) -C topology clean
	#$(MAKE) -C tuning clean
	#$(MAKE) -C upc clean



