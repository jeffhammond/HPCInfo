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

#ifdef SEQUENTIAL_CONSISTENCY
auto load_model  = std::memory_order_seq_cst;
auto store_model = std::memory_order_seq_cst;
#else
auto load_model  = std::memory_order_acquire;
auto store_model = std::memory_order_release;
#endif

int main(int argc, char * argv[])
{
    int nt = omp_get_max_threads();
#if 1
    if (nt != 2) omp_set_num_threads(2);
#else
    if (nt < 2)      omp_set_num_threads(2);
    if (nt % 2 != 0) omp_set_num_threads(nt-1);
#endif

    int iterations = (argc>1) ? atoi(argv[1]) : 100;

    std::cout << "thread ping-pong benchmark\n";
    std::cout << "num threads  = " << omp_get_max_threads() << "\n";
    std::cout << "iterations   = " << iterations << "\n";
#ifdef SEQUENTIAL_CONSISTENCY
    std::cout << "memory model = " << "seq_cst" << "\n";
#else
    std::cout << "memory model = " << "acq-rel" << "\n";
#endif
    std::cout << std::endl;

    std::atomic<int> left_ready  = {-1};
    std::atomic<int> right_ready = {-1};

    int left_payload  = 0;
    int right_payload = 0;

    #pragma omp parallel
    {
        int me      = omp_get_thread_num();
        /// 0=left 1=right
        bool parity = (me % 2 == 0);

        int junk = 0;

        /// START TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {

            if (parity) {

                /// send to left
                left_payload = i;
                left_ready.store(i, store_model);

                /// recv from right
                while (i != right_ready.load(load_model));
                //std::cout << i << ": left received " << right_payload << std::endl;
                junk += right_payload;

            } else {

                /// recv from left
                while (i != left_ready.load(load_model));
                //std::cout << i << ": right received " << left_payload << std::endl;
                junk += left_payload;

                ///send to right
                right_payload = i;
                right_ready.store(i, store_model);

            }

        }

        /// STOP TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        /// PRINT TIME
        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);
        #pragma omp critical
        {
            std::cout << "total time elapsed = " << dt.count()  << "\n";
            std::cout << "time per iteration = " << dt.count()/iterations  << "\n";
            std::cout << junk << std::endl;
        }
    }

    return 0;
}

#else  // C++11
#error You need C++11 for this test!
#endif // C++11
