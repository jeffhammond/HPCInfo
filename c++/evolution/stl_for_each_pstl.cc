#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>
namespace exec = std::execution;

//#include <ranges>
#include "ranges.h"

int main(int argc, char* argv[])
{
    int n = (argc>1) ? std::atoi(argv[1]) : 1000;

    auto range = ranges::view::iota(0,n);

    std::vector<float> A(n,2);
    std::vector<float> B(n,0);

    std::for_each( exec::par_unseq, std::begin(range), std::end(range), [&] (auto i) {
        B[i] += A[i] * A[i];
    });

    std::for_each( std::begin(range), std::end(range), [&] (auto i) {
        if (B[i] != 4.0) std::abort();
    });

    std::cout  << "ALL DONE" << std::endl;

    return 0;
}
