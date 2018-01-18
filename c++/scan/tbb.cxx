#include <iostream>
#include <vector>
#include <numeric> // partial_sum

#include "tbb/tbb.h"

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
              << "TBB scan test for " << n << " elements" << std::endl;

    const char* envvar = std::getenv("TBB_NUM_THREADS");
    const int num_threads = (envvar!=NULL) ? std::atoi(envvar) : tbb::task_scheduler_init::default_num_threads();
    tbb::task_scheduler_init init(num_threads);

    std::vector<double> in;
    in.resize(n,0);

    std::vector<double> psum(in);
    std::vector<double> tbbscan(in);

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

    t0 = wtime();
    for (auto i=0; i<rep; ++i)  {
        tbb::parallel_scan(tbb::blocked_range<size_t>(0,n),
                           0,
                           [&](const tbb::blocked_range<size_t> & r, double sum, bool is_final_scan) -> double {
                               double temp = sum;
                               for(size_t i=r.begin(); i<r.end(); ++i ) {
                                   temp += in[i];
                                   if( is_final_scan ) {
                                       tbbscan[i] = temp;
                                   }
                               }
                               return temp;
                           },
                           [](double left, double right) {
                               return left + right;
                           }
                          );
    }
    t1 = wtime();
    print(psum,"tbbscan");
    std::cout << "tbb::parallel_scan = " << t1-t0 << std::endl;

    std::cout << "====================================" << std::endl;

    return 0;
}
