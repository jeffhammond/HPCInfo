#if defined(__cplusplus) && (__cplusplus >= 201103L)

#include <iostream>
#include <iomanip>

#include <chrono>
#include <atomic>


#ifdef _OPENMP
# include <omp.h>
# define OMP_PARALLEL            _Pragma("omp parallel")
# define OMP_BARRIER             _Pragma("omp barrier")
# define OMP_CRITICAL            _Pragma("omp critical")
# ifdef SEQUENTIAL_CONSISTENCY
#  define OMP_ATOMIC              _Pragma("omp atomic seq_cst")
#  define OMP_ATOMIC_CAPTURE      _Pragma("omp atomic capture seq_cst")
# else
#  define OMP_ATOMIC              _Pragma("omp atomic")
#  define OMP_ATOMIC_CAPTURE      _Pragma("omp atomic capture")
# endif
#else
# error No OpenMP support!
#endif

/// from https://en.wikipedia.org/wiki/Fetch-and-add#x86_implementation
static inline int fetch_and_add(int* variable, int value)
{
    __asm__ ("lock; xaddl %0, %1"
                        : "+r" (value), "+m" (*variable) // input+output
                        : // No input-only
                        : //"memory"
    );
    return value;
}

static inline int add(int* variable, int value)
{
    __asm__ ("lock; addl %0, %1"
                        : "+r" (value), "+m" (*variable) // input+output
                        : // No input-only
                        : //"memory"
    );
    return value;
}

int main(int argc, char * argv[])
{
    int iterations = (argc>1) ? atoi(argv[1]) : 10000000;

    std::cout << "thread counter benchmark\n";
    std::cout << "num threads  = " << omp_get_max_threads() << "\n";
    std::cout << "iterations   = " << iterations << "\n";
    std::cout << "memory model = " << "x86 (seq_cst)";
    std::cout << std::endl;

    int counter = {0};

    OMP_PARALLEL
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// START TIME
        OMP_BARRIER
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {
            add(&counter, 1);
        }

        /// STOP TIME
        OMP_BARRIER
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// PRINT TIME
        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);
        OMP_CRITICAL
        {
            std::cout << "total time elapsed = " << dt.count() << "\n";
            std::cout << "time per iteration = " << dt.count()/iterations  << "\n";
            std::cout << counter << std::endl;
        }
    }

    counter = 0;

    OMP_PARALLEL
    {
        int output = -1;

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// START TIME
        OMP_BARRIER
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {
            output = fetch_and_add(&counter, 1);
        }

        /// STOP TIME
        OMP_BARRIER
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// PRINT TIME
        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);
        OMP_CRITICAL
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
