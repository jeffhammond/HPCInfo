#include <iostream>
#include <vector>
#include <numeric> // inclusive_scan
#include <iterator>
#include <functional>

#include "tbb/tbb.h"

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

    const char* envvar = std::getenv("TBB_NUM_THREADS");
    const int num_threads = (envvar!=NULL) ? std::atoi(envvar) : tbb::task_scheduler_init::default_num_threads();
    tbb::task_scheduler_init init(num_threads);

    std::vector<double> in;
    in.resize(n,0);

    std::vector<double> psum(in);
    std::vector<double> tbbscan(in);

    // initialize input to the sequence of natural numbers
    std::iota(in.begin(), in.end(), 1);
    print(in,"in");

    std::partial_sum(in.begin(), in.end(), psum.begin());
    print(psum,"psum");

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
    print(psum,"tbbscan");

    return 0;
}
