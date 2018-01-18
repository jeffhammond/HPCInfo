#include <iostream>
#include <vector>
#include <numeric> // inclusive_scan
#include <iterator>
#include <functional>

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1800)
#include <pstl/execution>
#include <pstl/algorithm>
#include <pstl/numeric>
#include <pstl/memory>

#include "tbb/tbb.h"
#endif

#include <chrono>

static inline double wtime(void)
{
    using t = std::chrono::high_resolution_clock;
    const auto c = t::now().time_since_epoch().count();
    const auto n = t::period::num;
    const auto d = t::period::den;
    const double r = static_cast<double>(c)/static_cast<double>(d)*static_cast<double>(n);
    return r;
}

template <class Container, class String>
void print(Container & v, String & name)
{
    if (v.size() > 1000) return;

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

// times to repeat each scan
const int rep = 20;

int main(int argc, char* argv[])
{
    size_t n = (argc>1) ? atol(argv[1]) : 10;

    std::cout << "====================================\n"
              << "C++17 scan test for " << n << " elements" << std::endl;

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1800)
    const char* envvar = std::getenv("TBB_NUM_THREADS");
    const int num_threads = (envvar!=NULL) ? std::atoi(envvar) : tbb::task_scheduler_init::default_num_threads();
    tbb::task_scheduler_init init(num_threads);
#endif

    std::vector<double> in;
    in.resize(n,0);

    std::vector<double> psum(in);
    std::vector<double> iscan(in);
    std::vector<double> xscan(in);

    // initialize input to the sequence of natural numbers
    auto t0 = wtime();
    std::iota(in.begin(), in.end(), 1);
    auto t1 = wtime();
    print(in,"in");
    std::cout << "std::iota = " << t1-t0 << std::endl;

    t0 = wtime();
    for (auto i=0; i<rep; ++i) std::partial_sum(in.begin(), in.end(), psum.begin());
    t1 = wtime();
    print(psum,"psum");
    std::cout << "std::partial_sum = " << t1-t0 << std::endl;

#if defined(_LIBCPP_VERSION)

    t0 = wtime();
    for (auto i=0; i<rep; ++i) std::inclusive_scan(in.begin(), in.end(), iscan.begin());
    t1 = wtime();
    print(iscan,"iscan");
    std::cout << "std::inclusive_scan = " << t1-t0 << std::endl;

    t0 = wtime();
    for (auto i=0; i<rep; ++i) std::exclusive_scan(in.begin(), in.end(), xscan.begin(), 0);
    t1 = wtime();
    print(xscan,"xscan");
    std::cout << "std::exclusive_scan = " << t1-t0 << std::endl;

#elif defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1800)

    t0 = wtime();
    for (auto i=0; i<rep; ++i) std::inclusive_scan(pstl::execution::seq, in.begin(), in.end(), iscan.begin());
    t1 = wtime();
    print(iscan,"iscan");
    std::cout << "std::inclusive_scan(seq) = " << t1-t0 << std::endl;

    t0 = wtime();
    for (auto i=0; i<rep; ++i) std::exclusive_scan(pstl::execution::seq, in.begin(), in.end(), xscan.begin(), 0);
    t1 = wtime();
    print(xscan,"xscan");
    std::cout << "std::exclusive_scan(seq) = " << t1-t0 << std::endl;

    t0 = wtime();
    for (auto i=0; i<rep; ++i) std::inclusive_scan(pstl::execution::unseq, in.begin(), in.end(), iscan.begin());
    t1 = wtime();
    print(iscan,"iscan");
    std::cout << "std::inclusive_scan(unseq) = " << t1-t0 << std::endl;

    t0 = wtime();
    for (auto i=0; i<rep; ++i) std::exclusive_scan(pstl::execution::unseq, in.begin(), in.end(), xscan.begin(), 0);
    t1 = wtime();
    print(xscan,"xscan");
    std::cout << "std::exclusive_scan(unseq) = " << t1-t0 << std::endl;

    t0 = wtime();
    for (auto i=0; i<rep; ++i) std::inclusive_scan(pstl::execution::par_unseq, in.begin(), in.end(), iscan.begin());
    t1 = wtime();
    print(iscan,"iscan");
    std::cout << "std::inclusive_scan(par_unseq) = " << t1-t0 << std::endl;

    t0 = wtime();
    for (auto i=0; i<rep; ++i) std::exclusive_scan(pstl::execution::par_unseq, in.begin(), in.end(), xscan.begin(), 0);
    t1 = wtime();
    print(xscan,"xscan");
    std::cout << "std::exclusive_scan(par_unseq) = " << t1-t0 << std::endl;

#else

#warning GCC libstdc++ does not yet support C++17 {in,ex}clusive_scan...

#endif

    std::cout << "====================================" << std::endl;

    return 0;
}
