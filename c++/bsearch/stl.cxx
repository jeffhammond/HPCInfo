#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <functional>
#include <random>

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
    for (auto const & i : v) {
        std::cout << i;
        if (&i == &v.back()) {
            std::cout << std::endl;
        } else {
            std::cout << ", ";
        }
    }
}

// times to repeat each binary_search
const int rep = 20;

int main(int argc, char* argv[])
{
    size_t n = (argc>1) ? atol(argv[1]) : 10;

    std::cout << "====================================\n"
              << "C++17 binary_search test for " << n << " elements" << std::endl;

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1800)
    const char* envvar = std::getenv("TBB_NUM_THREADS");
    const int num_threads = (envvar!=NULL) ? std::atoi(envvar) : tbb::task_scheduler_init::default_num_threads();
    tbb::task_scheduler_init init(num_threads);
#endif

    std::random_device random_device;
    std::mt19937 random_generator(random_device());
    std::uniform_int_distribution<> random_distribution(0, std::numeric_limits<int>::max());

    std::vector<int> haystack(n,0);
    int64_t needle = 0;
    bool found = false;

    // initialize input to the sequence of natural numbers
    auto t0 = wtime();
    //std::generate(haystack.begin(), haystack.end(), std::rand);
    std::generate(haystack.begin(), haystack.end(), [&](){ return random_distribution(random_generator); });
    auto t1 = wtime();
    print(haystack,"haystack");
    std::cout << "std::generate = " << t1-t0 << std::endl;

#if defined(_LIBCPP_VERSION)

    t0 = wtime();
    for (auto i=0; i<rep; ++i)
        found = std::binary_search(haystack.begin(), haystack.end(), needle);
    t1 = wtime();
    //print(iscan,"iscan");
    std::cout << "std::binary_search = " << t1-t0 << std::endl;

#elif defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1800)

    t0 = wtime();
    for (auto i=0; i<rep; ++i)
        found = std::binary_search(haystack.begin(), haystack.end(), needle);
    t1 = wtime();
    //print(iscan,"iscan");
    std::cout << "std::binary_search(seq) = " << t1-t0 << std::endl;

#else

#warning GCC libstdc++ does not yet support C++17 binary_search

#endif

    std::cout << "====================================" << std::endl;

    return 0;
}
