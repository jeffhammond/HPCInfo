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

int main(int argc, char * argv[])
{
    /// ensure we use an even number of threads
    {
        int nt = omp_get_max_threads();
        if (nt == 1) {
            std::cout << "You must use more than " << nt << " threads" << std::endl;
            abort();
        }
        if (nt % 2 != 0) omp_set_num_threads(nt-1);
#if 1
        /// temporary limitation
        if (nt != 2) omp_set_num_threads(2);
#endif
    }

    int iterations = (argc>1) ? atoi(argv[1]) : 100;

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
                left_ready.store(i, std::memory_order_release);

                /// recv from right
                while (i != right_ready.load(std::memory_order_acquire));
                //std::cout << i << ": left received " << right_payload << std::endl;
                junk += right_payload;

            } else {

                /// recv from left
                while (i != left_ready.load(std::memory_order_acquire));
                //std::cout << i << ": right received " << left_payload << std::endl;
                junk += left_payload;

                ///send to right
                right_payload = i;
                right_ready.store(i, std::memory_order_release);

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
