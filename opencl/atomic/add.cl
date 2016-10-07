__kernel void AtomicSum( __global int * s)
{
    //int i = get_global_id(0);
    const int i = 1;
    atomic_add(s,i);
}
