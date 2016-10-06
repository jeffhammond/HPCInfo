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
auto update_model = std::memory_order_seq_cst;
#else
auto update_model = std::memory_order_relaxed;
#endif

int main(int argc, char * argv[])
{
    int iterations = (argc>1) ? atoi(argv[1]) : 10000000;

    std::cout << "thread counter benchmark\n";
    std::cout << "num threads  = " << omp_get_max_threads() << "\n";
    std::cout << "iterations   = " << iterations << "\n";
#ifdef SEQUENTIAL_CONSISTENCY
    std::cout << "memory model = " << "seq_cst";
#else
    std::cout << "memory model = " << "relaxed";
#endif
    std::cout << std::endl;

    std::cout << "1) std::atomic_fetch_add(&counter, 1)\n";

    std::atomic<int> counter = {0};

    #pragma omp parallel
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// START TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {
            std::atomic_fetch_add(&counter, 1);
        }

        /// STOP TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// PRINT TIME
        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);
        #pragma omp critical
        {
            std::cout << "total time elapsed = " << dt.count() << "\n";
            std::cout << "time per iteration = " << dt.count()/iterations  << "\n";
            std::cout << counter << std::endl;
        }
    }

    std::cout << "2) ++counter\n";

    counter = 0;

    #pragma omp parallel
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// START TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {
            ++counter;
        }

        /// STOP TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// PRINT TIME
        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);
        #pragma omp critical
        {
            std::cout << "total time elapsed = " << dt.count() << "\n";
            std::cout << "time per iteration = " << dt.count()/iterations  << "\n";
            std::cout << counter << std::endl;
        }
    }

    std::cout << "3) counter++\n";

    counter = 0;

    #pragma omp parallel
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// START TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {
            counter++;
        }

        /// STOP TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// PRINT TIME
        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);
        #pragma omp critical
        {
            std::cout << "total time elapsed = " << dt.count() << "\n";
            std::cout << "time per iteration = " << dt.count()/iterations  << "\n";
            std::cout << counter << std::endl;
        }
    }

    std::cout << "4) output = counter++\n";

    counter = 0;

    #pragma omp parallel
    {
        int output = -1;

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// START TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {
            output = counter++;
        }

        /// STOP TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// PRINT TIME
        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);
        #pragma omp critical
        {
            std::cout << "total time elapsed = " << dt.count() << "\n";
            std::cout << "time per iteration = " << dt.count()/iterations  << "\n";
            std::cout << counter << std::endl;
            std::cout << output << std::endl;
        }
    }

    std::cout << "5) output = std::atomic_fetch_add(&counter, 1)\n";

    counter = 0;

    #pragma omp parallel
    {
        int output = -1;

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// START TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {
            output = std::atomic_fetch_add(&counter, 1);
        }

        /// STOP TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// PRINT TIME
        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);
        #pragma omp critical
        {
            std::cout << "total time elapsed = " << dt.count() << "\n";
            std::cout << "time per iteration = " << dt.count()/iterations  << "\n";
            std::cout << counter << std::endl;
            std::cout << output << std::endl;
        }
    }

    return 0;
}

#else  // C++11
#error You need C++11 for this test!
#endif // C++11
