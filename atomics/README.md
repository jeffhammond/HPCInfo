# Information

## Websites

* [C/C++11 mappings to processors](https://www.cl.cam.ac.uk/~pes20/cpp/cpp0xmappings.html)
* [Reasoning about C11 Program Transformations](http://plv.mpi-sws.org/c11comp/)
* [C11 atomic variables and the kernel](https://lwn.net/Articles/586838/)
* Preshing's [Weak vs. Strong Memory Models](http://preshing.com/20120930/weak-vs-strong-memory-models/)

## PDFs

* [The C11 and C++11 Concurrency Model](http://www.sigplan.org/Awards/Dissertation/2015_batty.pdf) ([Mark Batty](https://www.cl.cam.ac.uk/~mjb220/)'s dissertation)
* [A Tutorial Introduction to the ARM and POWER Relaxed Memory Models](https://www.cl.cam.ac.uk/~pes20/ppc-supplemental/test7.pdf)
* [Taming the complexities of the C11 and OpenCL memory models](http://multicore.doc.ic.ac.uk/opencl/openclmm_paper.pdf)

# Details

## Interfaces

* C11 atomics: 
   * [API docs](http://en.cppreference.com/w/c/atomic)
   * [Clang docs](https://clang.llvm.org/docs/LanguageExtensions.html#c11-atomic-operations)
* C++11 atomics:
   * [API docs](http://en.cppreference.com/w/cpp/atomic)
* GCC `__sync` intrinsics:
   * [GCC docs](https://gcc.gnu.org/onlinedocs/gcc-7.2.0/gcc/_005f_005fsync-Builtins.html)
* GCC `__atomic` intrinsics:
   * [GCC docs](https://gcc.gnu.org/onlinedocs/gcc-7.2.0/gcc/_005f_005fatomic-Builtins.html)
* Clang `__c11` intrinsics:
   * [Clang docs](https://clang.llvm.org/docs/LanguageExtensions.html#c11-atomic-builtins)
* OpenPA: 
   * [Home page](http://www.mcs.anl.gov/project/openpa-open-portable-atomics)
   * [Trac](https://trac.mpich.org/projects/openpa/wiki)
   * [Jeff's GitHub repo](https://github.com/jeffhammond/OpenPA)
