#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int main(void)
{
    cl_uint n = 0;
    clGetPlatformIDs(0, 0, &n);
    return 0;
}
