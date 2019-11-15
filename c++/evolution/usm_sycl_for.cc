#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <vector>
#include <algorithm>
//#include <ranges>
#include "ranges.h"

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

int main(int argc, char* argv[])
{
    int n = (argc>1) ? std::atoi(argv[1]) : 1000;

    auto range = ranges::view::iota(0,n);

    std::vector<float> hA(n,2);
    std::vector<float> hB(n,0);

    sycl::buffer<float> dA( hA.data(), hA.size() );
    sycl::buffer<float> dB( hB.data(), hB.size() );

    sycl::queue q(sycl::default_selector{});

    q.submit([&](sycl::handler& h) {

        auto A = dA.template get_access<sycl::access::mode::read>(h);
        auto B = dB.template get_access<sycl::access::mode::read_write>(h);

        h.parallel_for<class kernel>( sycl::range<1>{n}, [=] (sycl::id<1> it) {
            const int i = it[0];
            B[i] += A[i] * A[i];
        });
    });
    q.wait();

    std::for_each( std::begin(range), std::end(range), [&] (auto i) {
        if (hB[i] != 4.0) std::abort();
    });

    std::cout  << "ALL DONE" << std::endl;

    return 0;
}
