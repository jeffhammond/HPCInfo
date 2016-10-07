//------------------------------------------------------------------------------
//
// kernel:  vadd
//
// Purpose: Compute the elementwise sum c = a+b
//
// input: a and b float vectors of length count
//
// output: c float vector of length count holding the sum a + b
//
__kernel void vadd( __global int * s)
{
   int i = get_global_id(0);
   atomic_add(s,i);
}
