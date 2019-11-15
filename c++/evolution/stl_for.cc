#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <vector>
#include <algorithm>
//#include <ranges>
#include "ranges.h"

int main(int argc, char* argv[])
{
    int n = (argc>1) ? std::atoi(argv[1]) : 1000;

    std::vector<float> A(n,2);
    std::vector<float> B(n,0);

    for(int i=0; i<n; i++) {
        B[i] += A[i] * A[i];
    }

    for(int i=0; i<n; i++) {
        if (B[i] != 4.0) std::abort();
    }

    std::cout  << "ALL DONE" << std::endl;

    return 0;
}
