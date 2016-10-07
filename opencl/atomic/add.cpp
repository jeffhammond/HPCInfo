//------------------------------------------------------------------------------
//
// Name:       add.cpp
//
// Purpose:    Atomic summation, inspired by:
//             http://simpleopencl.blogspot.com/2013/04/performance-of-atomics-atomics-in.html
//
// HISTORY:    Written by Tim Mattson, June 2011
//             Ported to C++ Wrapper API by Benedict Gaster, September 2011
//             Updated to C++ Wrapper API v1.2 by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//             Rewritten to a different purpose altogther by Jeff Hammond, October 2016
//
//------------------------------------------------------------------------------

#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>

#include "cl.hpp"
#include "util.hpp"

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

int main(int argc, char* argv[])
{
    try {
        cl::Context context(DEVICE);
        cl::Program program(context, util::loadProgram("add.cl"), /* build= */ true);
        cl::CommandQueue queue(context);

        // create host and device data
        int sum=0;
        auto bufferSum = cl::Buffer(context, CL_MEM_READ_WRITE, 1 * sizeof(int));
        // copy host-to-device
        queue.enqueueWriteBuffer(bufferSum, CL_TRUE, 0, 1 * sizeof(int), &sum);

        auto kernel=cl::Kernel(program, "AtomicSum");
        // bind device buffer to first argument in kernel invocation
        kernel.setArg(0,bufferSum);
        // run the kernel
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1024*1024*128), cl::NullRange);
        queue.finish();

        // copy device-to-host
        queue.enqueueReadBuffer(bufferSum,CL_TRUE,0,1 * sizeof(int),&sum);

        std::cout << "Sum: " << sum << std::endl;
    }
    catch (cl::Error err) {
        std::cout << "Exception\n";
        std::cerr << "ERROR: " << err.what() << std::endl;
    }
    return 0;
}
