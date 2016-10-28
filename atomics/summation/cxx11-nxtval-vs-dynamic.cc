#if defined(__cplusplus) && (__cplusplus >= 201103L)

#include <iostream>
#include <iomanip>

#include <atomic>

#include <cmath>

#ifdef _OPENMP
# include <omp.h>
# define OMP_PARALLEL            _Pragma("omp parallel reduction(+:work)")
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

double foo(int i)
{
    const unsigned k = (i+33)%1000;
    const unsigned n = (k*k)%100000;
    double junk = 0.0;
    for (int j=1; j<n; ++j) {
        junk += std::log(static_cast<double>(j));
    }
    return junk;
}

int main(int argc, char * argv[])
{
    const int iterations = (argc>1) ? atoi(argv[1]) : 100000;

    const int num_threads = omp_get_max_threads();

    std::cout << "thread counter benchmark\n";
    std::cout << "num threads  = " << num_threads << "\n";
    std::cout << "iterations   = " << iterations << "\n";
#ifdef SEQUENTIAL_CONSISTENCY
    std::cout << "memory model = " << "seq_cst";
#else
    std::cout << "memory model = " << "relaxed";
#endif
    std::cout << std::endl;

    std::cout << "1) std::atomic_fetch_add(&counter, 1)\n";

    std::atomic<int> counter = {0};

    int work = 0;
    OMP_PARALLEL
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);

        double junk  = 0.0;

        /// START TIME
        OMP_BARRIER
        auto t0 = omp_get_wtime();

        int count = 0;
        int next  = std::atomic_fetch_add(&counter, 1);
        for (int i=0; i<iterations; ++i) {
            if (next==count) {
                ++work;
                junk += foo(i);
                next = std::atomic_fetch_add(&counter, 1);
            }
            ++count;
        }

        /// STOP TIME
        OMP_BARRIER
        auto t1 = omp_get_wtime();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// PRINT TIME
        auto dt = (t1-t0);
        OMP_CRITICAL
        {
            std::cout << "total time elapsed = " << dt << "\n";
            std::cout << "time per iteration = " << dt/iterations  << "\n";
            std::cout << "junk = " << junk << std::endl;
        }
    }
    std::cout << "work = " << work << std::endl;

    std::cout << "2) #pragma omp atomic acquire\n";

    int omp_counter = 0;

    work = 0;
    OMP_PARALLEL
    {
        double junk  = 0.0;

        /// START TIME
        OMP_BARRIER
        auto t0 = omp_get_wtime();

        int count = 0;
        int next;
        OMP_ATOMIC_CAPTURE
        next = omp_counter++;
        for (int i=0; i<iterations; ++i) {
            if (next==count) {
                ++work;
                junk += foo(i);
                OMP_ATOMIC_CAPTURE
                next = omp_counter++;
            }
            ++count;
        }

        /// STOP TIME
        OMP_BARRIER
        auto t1 = omp_get_wtime();

        /// PRINT TIME
        auto dt = (t1-t0);
        OMP_CRITICAL
        {
            std::cout << "total time elapsed = " << dt << "\n";
            std::cout << "time per iteration = " << dt/iterations  << "\n";
            std::cout << "junk = " << junk << std::endl;
        }
    }
    std::cout << "work = " << work << std::endl;

    work = 0;
    OMP_PARALLEL
    {
        double junk  = 0.0;

        /// START TIME
        OMP_BARRIER
        auto t0 = omp_get_wtime();

        #pragma omp for schedule(dynamic,1)
        for (int i=0; i<iterations; ++i) {
            ++work;
            junk += foo(i);
        }

        /// STOP TIME
        OMP_BARRIER
        auto t1 = omp_get_wtime();

        /// PRINT TIME
        auto dt = (t1-t0);
        OMP_CRITICAL
        {
            std::cout << "total time elapsed = " << dt << "\n";
            std::cout << "time per iteration = " << dt/iterations  << "\n";
            std::cout << "junk = " << junk << std::endl;
        }
    }
    std::cout << "work = " << work << std::endl;

    return 0;
}

#else  // C++11
#error You need C++11 for this test!
#endif // C++11
