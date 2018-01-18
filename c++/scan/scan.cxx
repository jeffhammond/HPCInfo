#include <iostream>
#include <vector>
#include <numeric> // inclusive_scan
#include <iterator>
#include <functional>

//#include "tbb/tbb.h"

template <class Container, class String>
void print(Container & v, String & name)
{
    std::cout << name << "\n";
    for (auto & i : v) {
        std::cout << i;
        if (&i == &v.back()) {
            std::cout << std::endl;
        } else {
            std::cout << ", ";
        }
    }
}

int main(int argc, char* argv[])
{
    size_t n = (argc>1) ? atol(argv[1]) : 1000;

    std::vector<double> in;
    in.resize(n,0);

    std::vector<double> psum(in);
    std::vector<double> iscan(in);
    std::vector<double> xscan(in);

    // initialize input to the sequence of natural numbers
    std::iota(in.begin(), in.end(), 1);
    print(in,"in");

    std::partial_sum(in.begin(), in.end(), psum.begin());
    print(psum,"psum");

#if defined(_LIBCPP_VERSION)
    std::inclusive_scan(in.begin(), in.end(), iscan.begin());
    print(iscan,"iscan");

    std::exclusive_scan(in.begin(), in.end(), xscan.begin(), 0);
    print(xscan,"xscan");
#else
#warning GCC libstdc++ does not yet support C++17 {in,ex}clusive_scan...
#endif

    return 0;
}
