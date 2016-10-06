#ifndef MYATOMICS_H
#define MYATOMICS_H

#if defined(__cplusplus) && (__cplusplus >= 201103L)

#include <atomic>

template <class T>
static inline T atomic_fetch_sum(std::atomic<T> * obj, T arg)
{
    T expected, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak(obj, &expected, desired));
    return expected;
}

template <class T>
static inline T atomic_fetch_sum(volatile std::atomic<T> * obj, T arg)
{
    T expected, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak(obj, &expected, desired));
    return expected;
}

#else

#error You do not have C++11

#endif // C++11

#endif // MYATOMICS_H
