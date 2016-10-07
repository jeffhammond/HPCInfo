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
#if 0
__kernel void vadd(                             
   __global atomic_float* a,                      
   __global atomic_float* b,                      
   __global atomic_float* c,                      
   const unsigned int count)               
{                                          
   int i = get_global_id(0);               
   if(i < count)  {
       atomic_store(c[i], atomic_load(a[i]) + atomic_load(b[i]));
   }
}                                          
#else
__kernel void vadd(                             
   __global float* a,                      
   __global float* b,                      
   __global float* c,                      
   const unsigned int count)               
{                                          
   int i = get_global_id(0);               
   if(i < count)  {
       c[i] = a[i] + b[i];                 
   }
}                                          
#endif
