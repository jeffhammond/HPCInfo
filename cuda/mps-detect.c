#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

int main(int argc, char* argv[])
{
    int device_num = ((argc > 1) ? atoi(argv[1]) : 0);
    CUresult rc;
    CUdevice device;
    rc = cuDeviceGet(&device, device_num); assert(rc == CUDA_SUCCESS);
    int attribute;
    rc = cuDeviceGetAttribute(&attribute, CU_DEVICE_ATTRIBUTE_MPS_ENABLED, device); assert(rc == CUDA_SUCCESS);
    printf("CU_DEVICE_ATTRIBUTE_MPS_ENABLED = %d\n", attribute);
    return 0;
}
