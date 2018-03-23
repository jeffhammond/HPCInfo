#if defined(__cplusplus) && (__cplusplus >= 201103L)

#include <iostream>
#include <iomanip>

#include <atomic>
#include <vector>

#include <chrono>

#ifdef _OPENMP
# include <omp.h>
#else
# error No OpenMP support!
#endif

#define SEQUENTIAL_CONSISTENCY 0

#if SEQUENTIAL_CONSISTENCY
auto load_model  = std::memory_order_seq_cst;
auto store_model = std::memory_order_seq_cst;
#else
auto load_model  = std::memory_order_acquire;
auto store_model = std::memory_order_release;
#endif

int main(int argc, char * argv[])
{
    int nt = omp_get_max_threads();
    int iterations = (argc>1) ? atoi(argv[1]) : 1000000;

    std::cout << "thread baton benchmark\n";
    std::cout << "num threads  = " << nt << "\n";
    std::cout << "iterations   = " << iterations << "\n";
#if SEQUENTIAL_CONSISTENCY
    std::cout << "memory model = " << "seq_cst";
#else
    std::cout << "memory model = " << "acq-rel";
#endif
    std::cout << std::endl;

    std::vector<std::atomic<int>> flags(nt);
    for (auto & f : flags) { f = -1; }
    flags[nt-1] = 0;

    std::cerr << "BEFORE: ";
    for (auto & f : flags) {
        std::cerr << f << ",";
    }
    std::cerr << std::endl;

    double dtmin(1e9), dtmax(0), dtavg(0);

    #pragma omp parallel reduction(min:dtmin) reduction(max:dtmax) reduction(+:dtavg)
    {
        int me    = omp_get_thread_num();
        int end   = nt-1;
        int left  = (me==0) ? end : (me-1);

        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {

#if DEBUG
            #pragma omp critical
            {
                std::cerr << "BEFORE " << me << ": ";
                for (auto & f : flags) {
                    std::cerr << f << ",";
                }
                std::cerr << std::endl;
            }
#endif

#if 0
            if (me > 0) {
                while (flags[me-1].load(load_model) < i);
                //std::cout << me << ": " << "received flag (iteration " << i << ")" << std::endl;
            }

            if (me < (nt-1) ) {
                flags[me].store(i, store_model);
                //std::cout << me << ": " << "sent flag (iteration " << i << ")" << std::endl;
            }
#else
            while (flags[left].load(load_model) < i);
            //std::cout << me << ": " << "received flag (iteration " << i << ")" << std::endl;

            int val = (me==end) ? (i+1) : i;
            flags[me].store(val, store_model);
            //std::cout << me << ": " << "sent flag (iteration " << i << ")" << std::endl;
#endif

#if DEBUG
            #pragma omp barrier
            #pragma omp critical
            {
                std::cerr << "AFTER " << me << ": ";
                for (auto & f : flags) {
                    std::cerr << f << ",";
                }
                std::cerr << std::endl;
            }
#endif
        }

        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);

#ifdef DEBUG
        #pragma omp critical
        {
            std::cerr << "total time elapsed = " << dt.count() << "\n";
            std::cerr << "time per iteration = " << dt.count()/iterations  << "\n";
        }
#endif

        dtmax = dt.count();
        dtmin = dt.count();
        dtavg = dt.count();
    }

    std::cout << "MAX=" << dtmax/iterations << "\n";
    std::cout << "MIN=" << dtmin/iterations << "\n";
    std::cout << "AVG=" << dtavg/nt/iterations << std::endl;

    std::cerr << "AFTER: ";
    for (auto & f : flags) {
        std::cerr << f << ",";
    }
    std::cerr << std::endl;

    return 0;
}

#else  // C++11
#error You need C++11 for this test!
#endif // C++11
