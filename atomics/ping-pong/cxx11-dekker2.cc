#if defined(__cplusplus) && (__cplusplus >= 201103L)

#include <iostream>
#include <iomanip>

#include <atomic>

#include <chrono>

#ifdef _OPENMP
# include <omp.h>
#else
# error No OpenMP support!
#endif

auto sc_model  = std::memory_order_seq_cst;

int main(int argc, char * argv[])
{
    int nt = omp_get_max_threads();
    if (nt != 2) omp_set_num_threads(2);

    int iterations = (argc>1) ? atoi(argv[1]) : 1000000;

    std::cout << "Dekker benchmark\n";
    std::cout << "num threads  = " << omp_get_max_threads() << "\n";
    std::cout << "iterations   = " << iterations << "\n";
    std::cout << std::endl;

    std::atomic<int> * x = new std::atomic<int>[iterations];
    std::atomic<int> * y = new std::atomic<int>[iterations];
    for (int i=0; i<iterations; ++i) {
        x[i].store(0, std::memory_order_relaxed);
        y[i].store(0, std::memory_order_relaxed);
    }
    std::atomic_thread_fence(sc_model);
    int lcount=0;
    int rcount=0;

    #pragma omp parallel
    {
        int me = omp_get_thread_num();

        /// START TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {
            if (me==0) {
                x[i].store(1, sc_model);
                int temp = y[i].load(sc_model);
                if (temp==0) {
                    lcount++;
                }
            } else {
                y[i].store(1, sc_model);
                int temp = x[i].load(sc_model);
                if (temp==0) {
                    rcount++;
                }
            }
            #pragma omp barrier
        }

        /// STOP TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        /// PRINT TIME
        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);
        #pragma omp critical
        {
            std::cout << "total time elapsed = " << dt.count() << "\n";
            std::cout << "time per iteration = " << dt.count()/iterations  << "\n";
            std::cout << "lcount=" << lcount << ", rcount=" << rcount << std::endl;
        }
    }

    return 0;
}

#else  // C++11
#error You need C++11 for this test!
#endif // C++11
