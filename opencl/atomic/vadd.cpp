//------------------------------------------------------------------------------
//
// Name:       vadd_cpp.cpp
// 
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
//                   c = a + b
//
// HISTORY:    Written by Tim Mattson, June 2011
//             Ported to C++ Wrapper API by Benedict Gaster, September 2011
//             Updated to C++ Wrapper API v1.2 by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//             
//------------------------------------------------------------------------------

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

#include "util.hpp" // utility library

#include <vector>
//#include <cstdio>
//#include <cstdlib>
//#include <string>

#include <iostream>
#include <fstream>

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

//char* err_code(cl_int);

//------------------------------------------------------------------------------

const double threshold = 0.001;

int main(int argc, char* argv[])
{
    const int count = (argc>1) ? atoi(argv[1]) : 1024;

    std::vector<float> h_a(count);             // a vector
    std::vector<float> h_b(count);             // b vector
    std::vector<float> h_c(count, 0xdeadbeef); // c = a + b, from compute device

    cl::Buffer d_a;      // device memory used for the input  a vector
    cl::Buffer d_b;      // device memory used for the input  b vector
    cl::Buffer d_c;      // device memory used for the output c vector

    // Fill vectors a and b with random float values
    // FIXME: replace with C++11 RNG
    for(int i = 0; i < count; ++i) {
        h_a[i]  = rand() / (float)RAND_MAX;
        h_b[i]  = rand() / (float)RAND_MAX;
    }

    try {
    	// Create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context

        cl::Program program(context, util::loadProgram("vadd.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor

        auto vadd = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "vadd");

        d_a  = cl::Buffer(context, begin(h_a), end(h_a), true);
        d_b  = cl::Buffer(context, begin(h_b), end(h_b), true);
        d_c  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count);

        util::Timer timer;

        vadd(cl::EnqueueArgs( queue, cl::NDRange(count)), d_a, d_b, d_c, count);

        queue.finish();

        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        std::cout << "The kernels ran in " << rtime << " seconds\n";

        cl::copy(queue, d_c, begin(h_c), end(h_c));

        // Test the results
        int correct = 0;
        for(int i = 0; i < count; ++i) {
            float tmp = (h_a[i] + h_b[i]) - h_c[i];
            if(tmp*tmp < threshold*threshold) {
                correct++;
            } else {
                std::cerr << " tmp=" << tmp << " h_a=" << h_a[i] << " h_b=" << h_b[i] << " h_c=" << h_c[i] << " \n";
            }
        }

        // summarize results
        std::cout << "vector add to find C = A+B: " << correct << " out of " << count << "  results were correct.\n";
    }
    catch (cl::Error err) {
        std::cout << "Exception\n";
        std::cerr << "ERROR: " << err.what() << std::endl;
    }
    return 0;
}
