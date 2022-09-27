# Overview

Preprocessor macros allow a user to write portable code in the face of non-portable features by enabling platform-specific code only when it is supported.

# External Resources

See [Pre-defined C/C++ Compiler Macros](http://sourceforge.net/p/predef/wiki/Home/) for an incredibly thorough list of preprocessor macros for various compilers, systems, etc.

# Manual Extraction

With GCC, you can get the predefined macros using <tt>gcc -dM -E - < /dev/null</tt>.

# Languages

Language|Macro|Details
---|---|---
C89|``__STDC__``|ANSI X3.159-1989
C90|``__STDC_VERSION__``| ISO/IEC 9899:1990
C94|``__STDC_VERSION__ = 199409L``|ISO/IEC 9899-1:1994
C99|``__STDC_VERSION__ = 199901L``|ISO/IEC 9899:1999
C11|``__STDC_VERSION__ = 201112L``|ISO/IEC 9899:2011
C20|``__STDC_VERSION__ > 201710L``|
C++|``__cplusplus``|
C++98|``__cplusplus = 199711L``|ISO/IEC 14882:1998
C++11|``__cplusplus = 201103L``|ISO/IEC 14882:2011
C++14|``__cplusplus = 201402L``|ISO/IEC 14882:2014
C++17|``__cplusplus = 201703L``|ISO/IEC 14882:2017
C++20|``__cplusplus > 201703L``|

See https://en.cppreference.com/w/cpp/feature_test for fine-grain feature testing in C++.

# Language Extensions for Parallelism

Language|Macro|Details
---|---|---
UPC|``__UPC__``|UPC Identification
UPC|``__UPC_DYNAMIC_THREADS__``|The integer constant 1 in the dynamic THREADS translation environment.
UPC|``__UPC_STATIC_THREADS__``|The integer constant 1 in the static THREADS translation environment.</td>
OpenMP|``_OPENMP``|OpenMP
OpenMP|``_OPENMP = 200505``|OpenMP 2.5
OpenMP|``_OPENMP = 200805``|OpenMP 3.0
OpenMP|``_OPENMP = 201107``|OpenMP 3.1
OpenMP|``_OPENMP = 201307``|OpenMP 4.0
OpenMP|``_OPENMP = 201511``|OpenMP 4.5
OpenMP|``_OPENMP = 201811``|OpenMP 5.0
Cilk|`__cilk = 200`|Cilk++

# Compilers

## GCC

Macro|Purpose
---|---
``__GNUC__``|Identification
``__GNUC__``|Version
``__GNUC_MINOR__``|Revision
``__GNUC_PATCHLEVEL__``|Patch (introduced in version 3.0)

## Clang/LLVM

Macro|Purpose
---|---
``__llvm__``|Identification of LLVM
``__clang__``|Identification of Clang
``__clang_major__``|Clang Version
``__clang_minor__``|Clang Revision
``__clang_patchlevel__``|Clang Patch

## Intel

Macro|Purpose
---|---
``__INTEL_COMPILER ``|Identification
``__INTEL_COMPILER ``|Version (Format: VRP, where V = Version, R = Revision, P = Patch)
``__INTEL_OFFLOAD ``|True if Intel compiler supports offload (on by default)

## Cray

Macro|Purpose
---|---
``_CRAYC ``|Identification
``_RELEASE ``|Version
``_RELEASE_MINOR ``|Revision

## IBM

Macro|Purpose
---|---
``__xlc__ ``|Identification
``__xlC__ ``|Identification
``__IBMC__ ``|Identification
``__IBMCPP__ ``|Identification
``__IBMC__ ``|Version (Format: VRP, where V = Version, R = Revision, P = Patch)
``__IBMCPP__ ``|Version (Format: VRP, where V = Version, R = Revision, P = Patch)

## NVHPC

Use ``nvc++ -stdpar -cuda -acc -mp -target=multicore -dM -E /dev/null | grep NV`` to find more...

Macro|Purpose
---|---
``__NVCOMPILER ``|Identification
``__NVCOMPILER_MAJOR__ ``|Version
``__NVCOMPILER_MINOR__ ``|Revision
``__NVCOMPILER_PATCHLEVEL__ ``|Patch
``__NVCOMPILER_CUDA__``, ``_NVHPC_CUDA``|CUDA support enabled
``_NVHPC_STDPAR_GPU``, ``_NVHPC_STDPAR_CUDA``| StdPar for GPU with CUDA (i.e. ``-stdpar=gpu``)
``_NVHPC_STDPAR_MULTICORE``, ``_NVHPC_STDPAR_OMP``| StdPar for CPU with OpenMP (i.e. ``-stdpar=multicore``)
``__NVCOMPILER_CUDA_ARCH__``|CUDA arch (e.g. 860 for ``sm_86``)

Note that the following PGI macros are also supported, for historical reasons.

## PGI

Macro|Purpose
---|---
``__PGI ``|Identification
``__PGIC__ ``|Version
``__PGIC_MINOR__ ``|Revision
``__PGIC_PATCHLEVEL__ ``|Patch

# Programming Models

## MPI

Macro|Purpose
---|---
``MPI_VERSION ``|Version
``MPI_SUBVERSION ``|Revision
``MPICH``|Identification of MPICH
``MPICH_NUMVERSION ``|MPICH Version for numerical comparison (See ``mpi.h`` for details)
``MPICH_VERSION ``|MPICH Version as string
``OPEN_MPI``|Identification of OpenMPI
``OMPI_MAJOR_VERSION ``|OpenMPI Version
``OMPI_MINOR_VERSION ``|OpenMPI Revision
``OMPI_RELEASE_VERSION ``|OpenMPI Patch Level

## CUDA

Macro|Purpose
---|---
``__CUDACC__``|Identification
``__CUDACC_VER__``|Toolkit Version - added in CUDA 7.5, removed in CUDA 8.0 (Format: XXYYZZ, where XX = Major Version Number, YY = Minor Version Number, ZZ = Build Number)
``__CUDACC_VER_MAJOR__``|Toolkit Major Version - added in CUDA 7.5
``__CUDACC_VER_MINOR__``|Toolkit Minor Version - added in CUDA 7.5
``__CUDACC_VER_BUILD__``|Toolkit Build Version - added in CUDA 7.5
``CUDA_VERSION``|API Version - defined in `<cuda.h>` (Format: XXYY, where XX = Major Version Number, YY = Minor Version Number)

# Operating Systems

Some of these were obtained from the site linked above and are assumed to be correct.

OS|Macro|Details
---|---|---
Linux|``__linux__``|
Mac OSX|``__APPLE__``| This may not be the _best_ way to do this since this macro may be defined in other operating systems produced by Apple.
Mac OSX|``__MACH__``| Distinguishes Darwin/Mach kernel from iOS.
BSD|``__FreeBSD__``, ``__NetBSD__``, ``__OpenBSD__``, ``__bsdi__``, ``__DragonFly__``|Unverified.
AIX|``_AIX``| Unverified due to lack of system access.
Blue Gene CNK|``__bg__``| See below for details.

TODO: Test for POSIX support?

# Architectures

## Cray

Manually extracted from Cray compilers (i.e. verified):

Macro|Meaning
---|---
`__CRAYXT`|Defined on the Cray XT architecture.
`__CRAYXE`|Defined on the Cray XE architecture.
`__CRAYXC`|Defined on the Cray XC architecture.
`__CRAYXT_COMPUTE_LINUX_TARGET`|Defined when Cray is using Linux, as opposed to Catamount.

From [Cray documentation](http://docs.cray.com/books/S-2179-52/html-S-2179-52/zfixeddt715h8f.html) (i.e. reputable):

Macro|Meaning
---|---
`__CRAY`|Defined as 1 on UNICOS/mp systems.
`_CRAYIEEE`|Defined as 1 if the targeted CPU type uses IEEE floating-point format.

## Blue Gene

From [IBM Blue Gene/P compiler documentation](http://publib.boulder.ibm.com/infocenter/compbgpl/v9v111/index.jsp?topic=/com.ibm.bg9111.doc/bgusing/bg_platform_related.htm) and [IBM Blue Gene/Q compiler documentation](http://pic.dhe.ibm.com/infocenter/compbg/v121v141/index.jsp?topic=%2Fcom.ibm.xlcpp121.bg.doc%2Fcompiler_ref%2Fmacros_platform.html) (i.e. reliable):

Macro|Meaning
---|---
`__bg__`|Indicates that this is a Blue Gene platform.
`__blrts`, `__blrts__`|Indicates that the target architecture is Blue Gene/L.
`__bgp__`|Indicates that the architecture is Blue Gene/P.
`__bgq__`|Indicates that the architecture is Blue Gene/Q.
`__TOS_BLRTS__`|Indicates that the target architecture is Blue Gene/L.
`__TOS_BGP__`|Indicates that the target architecture is Blue Gene/P.
`__TOS_BGQ__`|Indicates that the target architecture is Blue Gene/Q.
`__VECTOR4DOUBLE__`|Indicates the support of vector data types on Blue Gene/Q.

The ``__bg__``, ``__bgp__``, and ``__bgq__`` have been verified on both Blue Gene/P and Blue Gene/Q systems and are the recommended one in this authors opinion.

# Processor Features

## Intel/AMD

Suffix|Macro|Description
---|---|---
SSE|``__SSE__``|Introduced with Pentium 3.
SSE2|``__SSE2__``|Introduced with Willamette (Pentium 4).
SSE3|``__SSE3__``|Introduced with Prescott.
SSE4.1|``__SSE4_1__``|Introduced with Penryn.
SSE4.2|``__SSE4_2__``|Introduced with Penryn.
SSSE3|``__SSSE3__``|Introduced with Merom.
AVX|``__AVX__``|Introduced with Sandy Bridge.
AVX2|``__AVX2__``|Introduced with Haswell.
FMA|``__FMA__``|Fused-multiple-add instructions.
FMA4|``__FMA4__``|See https://en.wikipedia.org/wiki/FMA_instruction_set.

## AVX-512

Suffix|Macro|Description
---|---|---
F|``__AVX512F__``|Foundation
CD|``__AVX512CD__``|Conflict Detection Instructions (CDI)
ER|``__AVX512ER__``|Exponential and Reciprocal Instructions (ERI) 
PF|``__AVX512PF__``|Prefetch Instructions (PFI)
DQ|``__AVX512DQ__``|Doubleword and Quadword Instructions (DQ)
BW|``__AVX512BW__``|Byte and Word Instructions (BW)
VL|``__AVX512VL__``|Vector Length Extensions (VL)
IFMA|``__AVX512IFMA__``|Integer Fused Multiply Add (IFMA)
VBMI|``__AVX512VBMI__``|Vector Byte Manipulation Instructions (VBMI)
VBMI2|``__AVX512VBMI2__``|Vector Byte Manipulation Instructions 2 (VBMI2)
BITALG|``__AVX512BITALG__``|Vector Bit Algorithms
VNNI|``__AVX512VNNI__ ``|Vector Neural Network Instructions
VNNIW|``__AVX5124VNNIW__ ``|Vector instructions for deep learning enhanced word variable precision
FMAPS|``__AVX5124FMAPS__ ``|Vector instructions for deep learning floating-point single precision
VPOPCNT|``__AVX512VPOPCNTDQ__ ``|Vectorized Hamming weight
VP2INTERSECT|``__AVX512VP2INTERSECT__``|Compute intersection between doublewords/quadwords to a pair of mask registers


The primary source for these is the [Intel® Architecture Instruction Set Extensions Programming Reference](https://software.intel.com/en-us/isa-extensions).

The intrinsics for AVX-512 are nicely documented in the [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide).

Wikipedia's articles [AVX-512](https://en.wikipedia.org/wiki/AVX-512) is a useful unofficial source.

LLVM source code (http://llvm.org/viewvc/llvm-project/cfe/trunk/test/Preprocessor/predefined-arch-macros.c?view=diff&r1=262200&r2=262201&pathrev=262201) is the basis for two of the entries in the above table.

[Cauldron14_AVX-512_Vector_ISA_Kirill_Yukhin_20140711.pdf](https://github.com/m-a-d-n-e-s-s/madness/files/460538/Cauldron14_AVX-512_Vector_ISA_Kirill_Yukhin_20140711.pdf) and [Intel® AVX-512 Instructions and Their
Use in the Implementation of Math Functions](http://arith22.gforge.inria.fr/slides/s1-cornea.pdf) may be of interest.
