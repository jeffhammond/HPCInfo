#ifndef MYATOMICS_H
#define MYATOMICS_H

#if defined(__cplusplus) && (__cplusplus >= 201103L)

#include <atomic>

template <class T>
static inline void atomic_sum(std::atomic<T> * obj, T arg)
{
    T expected, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak(obj, &expected, desired));
}

template <class T>
static inline void atomic_sum(volatile std::atomic<T> * obj, T arg)
{
    T expected, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak(obj, &expected, desired));
}

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

/// EXPLICIT
//
// These are invalid if order is std::memory_order_release or std::memory_order_acq_rel.
// I attempted to address this below but have no idea what I am doing with templates,
// so I failed.
// Ideally, we can specialize on these two orders and set the right values
// in atomic_compare_exchange_weak_explicit.

template <class T>
static inline void atomic_sum_explicit(std::atomic<T> * obj, T arg, std::memory_order order)
{
    T expected, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak_explicit(obj, &expected, desired, order, order));
}

template <class T>
static inline void atomic_sum_explicit(volatile std::atomic<T> * obj, T arg, std::memory_order order)
{
    T expected, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak_explicit(obj, &expected, desired, order, order));
}

template <class T>
static inline T atomic_fetch_sum_explicit(std::atomic<T> * obj, T arg, std::memory_order order)
{
    T expected, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak_explicit(obj, &expected, desired, order, order));
    return expected;
}

template <class T>
static inline T atomic_fetch_sum_explicit(volatile std::atomic<T> * obj, T arg, std::memory_order order)
{
    T expected, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak_explicit(obj, &expected, desired, order, order));
    return expected;
}

#if 0
/// SPECIALIZATIONS
//
// template< class T >
// bool atomic_compare_exchange_strong_explicit( std::atomic<T>* obj,
//                                               T* expected, T desired,
//                                               std::memory_order succ,
//                                               std::memory_order fail );
//
//  obj         - pointer to the atomic object to test and modify
//  expected    - pointer to the value expected to be found in the atomic object
//  desired     - the value to store in the atomic object if it is as expected
//  succ        - the memory synchronization ordering for the read-modify-write operation
//                if the comparison succeeds. All values are permitted.
//  fail        - the memory synchronization ordering for the load operation
//                if the comparison fails.
//                Cannot be std::memory_order_release or std::memory_order_acq_rel
//                and cannot specify stronger ordering than succ
//

template <class T>
static inline T atomic_fetch_sum_explicit(std::atomic<T> * obj, T arg,
                                          typename std::underlying_type<std::memory_order>::type order)
{
    T expected, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak_explicit(obj, &expected, desired, order));
    return expected;
}

template <class T>
static inline T atomic_fetch_sum_explicit(volatile std::atomic<T> * obj, T arg,
                                          typename std::underlying_type<std::memory_order>::type order)
{
    T expected, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak_explicit(obj, &expected, desired, order));
    return expected;
}
#endif

#else

#error You do not have C++11

#endif // C++11

#endif // MYATOMICS_H
