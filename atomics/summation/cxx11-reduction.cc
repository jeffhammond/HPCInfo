#if defined(__cplusplus) && (__cplusplus >= 201103L)

#include <iostream>
#include <iomanip>

#include <atomic>
#include "myatomics.h"

#ifdef _OPENMP
# include <omp.h>
# define OMP_PARALLEL            _Pragma("omp parallel reduction(+:tavg) reduction(max:tmax) reduction(min:tmin) reduction(max:tinc)")
# define OMP_BARRIER             _Pragma("omp barrier")
# define OMP_CRITICAL            _Pragma("omp critical")
# define OMP_SINGLE              _Pragma("omp single")
# define OMP_ATOMIC              _Pragma("omp atomic")
# define OMP_ATOMIC_CAPTURE      _Pragma("omp atomic capture")
# define OMP_ATOMIC_SC           _Pragma("omp atomic seq_cst")
# define OMP_ATOMIC_CAPTURE_SC   _Pragma("omp atomic capture seq_cst")
#else
# error No OpenMP support!
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

    double tmin, tmax, tavg, tinc;

    std::atomic<double> counter = {0.0};
    double omp_counter = 0.0;

    std::cout << "1) atomic_sum(&counter, t0)\n";

    counter = 0.0;
    tmin=1.e9, tmax=-1.0, tavg=-1.0, tinc=-1.0;

    OMP_PARALLEL
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// START TIME
        OMP_BARRIER
        double t0 = omp_get_wtime();

        for (int i=0; i<iterations; ++i) {
            atomic_sum_explicit(&counter, t0, std::memory_order_relaxed);
        }

        /// STOP TIME
        double t1 = omp_get_wtime();
        OMP_BARRIER
        double t2 = omp_get_wtime();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        double junk = counter;

        /// PRINT TIME
        double dt = (t1-t0);
        tmin = dt;
        tmax = dt;
        tavg = dt;
        tinc = t2-t0;

        OMP_SINGLE
        {
            std::cerr << "junk = " << junk << std::endl;
        }
    }
    tavg /= omp_get_max_threads();
    std::cout << "total time elapsed (min,max,avg,inc) = " << tmin << ", " << tmax << ", " << tavg << ", " << tinc << " (s)\n";
    tmin *= 1.e9/iterations;
    tmax *= 1.e9/iterations;
    tavg *= 1.e9/iterations;
    tinc *= 1.e9/iterations;
    std::cout << "time per iteration (min,max,avg,inc) = " << tmin << ", " << tmax << ", " << tavg << ", " << tinc << " (ns)\n";

    std::cout << "3) #pragma omp atomic\n";

    omp_counter = 0.0;
    tmin=1.e9, tmax=-1.0, tavg=-1.0, tinc=-1.0;

    OMP_PARALLEL
    {
        /// START TIME
        OMP_BARRIER
        double t0 = omp_get_wtime();

        for (int i=0; i<iterations; ++i) {
            OMP_ATOMIC
            omp_counter += t0;
        }

        /// STOP TIME
        double t1 = omp_get_wtime();
        OMP_BARRIER
        double t2 = omp_get_wtime();

        double junk = omp_counter;

        /// PRINT TIME
        double dt = (t1-t0);
        tmin = dt;
        tmax = dt;
        tavg = dt;
        tinc = t2-t0;

        OMP_SINGLE
        {
            std::cerr << "junk = " << junk << std::endl;
        }
    }
    tavg /= omp_get_max_threads();
    std::cout << "total time elapsed (min,max,avg,inc) = " << tmin << ", " << tmax << ", " << tavg << ", " << tinc << " (s)\n";
    tmin *= 1.e9/iterations;
    tmax *= 1.e9/iterations;
    tavg *= 1.e9/iterations;
    tinc *= 1.e9/iterations;
    std::cout << "time per iteration (min,max,avg,inc) = " << tmin << ", " << tmax << ", " << tavg << ", " << tinc << " (ns)\n";

    std::cout << "5) atomic_sum(&counter, t0, seq_cst)\n";

    counter = 0.0;
    tmin=1.e9, tmax=-1.0, tavg=-1.0, tinc=-1.0;

    OMP_PARALLEL
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// START TIME
        OMP_BARRIER
        double t0 = omp_get_wtime();

        for (int i=0; i<iterations; ++i) {
            atomic_sum_explicit(&counter, t0, std::memory_order_seq_cst);
        }

        /// STOP TIME
        double t1 = omp_get_wtime();
        OMP_BARRIER
        double t2 = omp_get_wtime();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        double junk = counter;

        /// PRINT TIME
        double dt = (t1-t0);
        tmin = dt;
        tmax = dt;
        tavg = dt;
        tinc = t2-t0;

        OMP_SINGLE
        {
            std::cerr << "junk = " << junk << std::endl;
        }
    }
    tavg /= omp_get_max_threads();
    std::cout << "total time elapsed (min,max,avg,inc) = " << tmin << ", " << tmax << ", " << tavg << ", " << tinc << " (s)\n";
    tmin *= 1.e9/iterations;
    tmax *= 1.e9/iterations;
    tavg *= 1.e9/iterations;
    tinc *= 1.e9/iterations;
    std::cout << "time per iteration (min,max,avg,inc) = " << tmin << ", " << tmax << ", " << tavg << ", " << tinc << " (ns)\n";

    std::cout << "7) #pragma omp atomic seq_cst\n";

    omp_counter = 0.0;
    tmin=1.e9, tmax=-1.0, tavg=-1.0, tinc=-1.0;

    OMP_PARALLEL
    {
        /// START TIME
        OMP_BARRIER
        double t0 = omp_get_wtime();

        for (int i=0; i<iterations; ++i) {
            OMP_ATOMIC_SC
            omp_counter += t0;
        }

        /// STOP TIME
        double t1 = omp_get_wtime();
        OMP_BARRIER
        double t2 = omp_get_wtime();

        double junk = omp_counter;

        /// PRINT TIME
        double dt = (t1-t0);
        tmin = dt;
        tmax = dt;
        tavg = dt;
        tinc = t2-t0;

        OMP_SINGLE
        {
            std::cerr << "junk = " << junk << std::endl;
        }
    }
    tavg /= omp_get_max_threads();
    std::cout << "total time elapsed (min,max,avg,inc) = " << tmin << ", " << tmax << ", " << tavg << ", " << tinc << " (s)\n";
    tmin *= 1.e9/iterations;
    tmax *= 1.e9/iterations;
    tavg *= 1.e9/iterations;
    tinc *= 1.e9/iterations;
    std::cout << "time per iteration (min,max,avg,inc) = " << tmin << ", " << tmax << ", " << tavg << ", " << tinc << " (ns)\n";

    return 0;
}

#else  // C++11
#error You need C++11 for this test!
#endif // C++11
