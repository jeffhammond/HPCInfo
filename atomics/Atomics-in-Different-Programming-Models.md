# Atomic Support in Various Programming Models

This is stale.  The content here is from April 2013.  I will try to update it at some point.

## MPI-3

The one-sided remote memory access (RMA) functions in MPI-3 are:
* `MPI_Fetch_and_op`, which supports "any of the predefined operations for `MPI_Reduce`, as well as `MPI_NO_OP` or `MPI_REPLACE`.  "The datatype argument must be a predefined datatype."
* `MPI_Compare_and_swap`, which supports "C integer, Fortran integer, Logical, Multi-language types, or Byte as specified in (MPI-3) Section 5.9.2."

## UPC

See [UPC 1.3](https://upc-lang.org/assets/Uploads/spec/upc-lang-spec-1.3.pdf).

The following table presents the required support for operations and operand ￼types:
Operand Type|Accessors|Bit-wise Ops|Numeric Ops
---|---|---|---
Integer|X|X|X
Floating Point|X||X
`UPC_PTS`|X||

￼￼￼￼where
* Supported integer types are UPC_INT, UPC_UINT, UPC_LONG, UPC_ULONG, UPC_INT32, UPC_UINT32, UPC_INT64, and UPC_UINT64.
* Supported floating-point types are UPC_FLOAT and UPC_DOUBLE.
* Supported accessors are UPC_GET, UPC_SET, and UPC_CSWAP.
* Supported bit-wise operations are UPC_AND, UPC_OR, and UPC_XOR.
* Supported numeric operations are UPC_ADD, UPC_SUB, UPC_MULT, UPC_INC, UPC_DEC, UPC_MAX, and UPC_MIN.

## OpenSHMEM

See http://openshmem.org/.

Operations: 
* swap
* compare-and-swap
* fetch-and-add
* fetch-and-increment
* add
* increment

Types:
* C int,long, and long long
* C float and double for swap only

## OpenMP

From http://publib.boulder.ibm.com/infocenter/lnxpcomp/v8v101/index.jsp?topic=%2Fcom.ibm.xlcpp8l.doc%2Fcompiler%2Fref%2Fruprpdir.htm, the operations supported are:
```
+  *  -  /  &  ^  |  <<  >>
x++	 
++x	 
x--	 
--x
```

It would be good to have an authoritative answer for what is part of OpenMP 3.1+...

## Intel TBB

See http://threadingbuildingblocks.org/docs/help/reference/synchronization/atomic_cls.htm.

TBB's `atomic<T>` supports the following operations, where T may be an integral type, enumeration type, or a pointer type. 
* `fetch_and_add`
* `fetch_and_increment`
* `fetch_and_decrement`
* `compare_and_swap`
* `fetch_and_store` (aka swap)

## CUDA

See http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions

## OpenCL

See http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/atomicFunctions.html and http://www.khronos.org/registry/cl/specs/opencl-1.2.pdf#page=279.

## GCC

See 
* http://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Atomic-Builtins.html
* http://gcc.gnu.org/onlinedocs/gcc-4.7.2/gcc/_005f_005fatomic-Builtins.html#g_t_005f_005fatomic-Builtins
* http://gcc.gnu.org/onlinedocs/gcc-4.7.2/gcc/_005f_005fsync-Builtins.html#g_t_005f_005fsync-Builtins
