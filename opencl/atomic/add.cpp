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

#include "cl.hpp"

#include "util.hpp"

#include <vector>

#include <iostream>
#include <fstream>

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

int main(int argc, char* argv[])
{
    try {
    	// Create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context
        cl::Program program(context, util::loadProgram("add.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        int sum=0;
        cl::Buffer bufferSum = cl::Buffer(context, CL_MEM_READ_WRITE, 1 * sizeof(int));
        queue.enqueueWriteBuffer(bufferSum, CL_TRUE, 0, 1 * sizeof(int), &sum);
        cl::Kernel kernel=cl::Kernel(program, "AtomicSum");
        kernel.setArg(0,bufferSum);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1024*1024*128), cl::NullRange);
        queue.finish();
        queue.enqueueReadBuffer(bufferSum,CL_TRUE,0,1 * sizeof(int),&sum);
        std::cout << "Sum: " << sum << "\n";
    }
    catch (cl::Error err) {
        std::cout << "Exception\n";
        std::cerr << "ERROR: " << err.what() << std::endl;
    }
    return 0;
}
