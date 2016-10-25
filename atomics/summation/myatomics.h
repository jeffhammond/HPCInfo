#ifndef MYATOMICS_H
#define MYATOMICS_H

template <class T>
static inline void atomic_sum_relaxed(T * obj, T arg)
{
    T expected = *obj, desired;
    do {
      desired = expected + arg;
    } while (!__atomic_compare_exchange_n(obj, &expected, desired, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}

template <class T>
static inline void atomic_sum_relaxed(volatile T * obj, T arg)
{
    T expected = *obj, desired;
    do {
      desired = expected + arg;
    } while (!__atomic_compare_exchange_n(obj, &expected, desired, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}

template <class T>
static inline T atomic_fetch_sum_relaxed(T * obj, T arg)
{
    T expected = *obj, desired;
    do {
      desired = expected + arg;
    } while (!__atomic_compare_exchange_n(obj, &expected, desired, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
    return expected;
}

template <class T>
static inline T atomic_fetch_sum_relaxed(volatile T * obj, T arg)
{
    T expected = *obj, desired;
    do {
      desired = expected + arg;
    } while (!__atomic_compare_exchange_n(obj, &expected, desired, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
    return expected;
}

template <class T>
static inline void atomic_sum_explicit(T * obj, T arg, int order)
{
    T expected = *obj, desired;
    do {
      desired = expected + arg;
    } while (!__atomic_compare_exchange_n(obj, &expected, desired, true, order, order));
}

template <class T>
static inline void atomic_sum_explicit(volatile T * obj, T arg, int order)
{
    T expected = *obj, desired;
    do {
      desired = expected + arg;
    } while (!__atomic_compare_exchange_n(obj, &expected, desired, true, order, order));
}

template <class T>
static inline T atomic_fetch_sum_explicit(T * obj, T arg, int order)
{
    T expected = *obj, desired;
    do {
      desired = expected + arg;
    } while (!__atomic_compare_exchange_n(obj, &expected, desired, true, order, order));
    return expected;
}

template <class T>
static inline T atomic_fetch_sum_explicit(volatile T * obj, T arg, int order)
{
    T expected = *obj, desired;
    do {
      desired = expected + arg;
    } while (!__atomic_compare_exchange_n(obj, &expected, desired, true, order, order));
    return expected;
}

#if defined(__cplusplus) && (__cplusplus >= 201103L)

#include <atomic>

template <class T>
static inline void atomic_sum(std::atomic<T> * obj, T arg)
{
    T expected = *obj, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak(obj, &expected, desired));
}

template <class T>
static inline void atomic_sum(volatile std::atomic<T> * obj, T arg)
{
    T expected = *obj, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak(obj, &expected, desired));
}

template <class T>
static inline T atomic_fetch_sum(std::atomic<T> * obj, T arg)
{
    T expected = *obj, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak(obj, &expected, desired));
    return expected;
}

template <class T>
static inline T atomic_fetch_sum(volatile std::atomic<T> * obj, T arg)
{
    T expected = *obj, desired;
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
    T expected = *obj, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak_explicit(obj, &expected, desired, order, order));
}

template <class T>
static inline void atomic_sum_explicit(volatile std::atomic<T> * obj, T arg, std::memory_order order)
{
    T expected = *obj, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak_explicit(obj, &expected, desired, order, order));
}

template <class T>
static inline T atomic_fetch_sum_explicit(std::atomic<T> * obj, T arg, std::memory_order order)
{
    T expected = *obj, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak_explicit(obj, &expected, desired, order, order));
    return expected;
}

template <class T>
static inline T atomic_fetch_sum_explicit(volatile std::atomic<T> * obj, T arg, std::memory_order order)
{
    T expected = *obj, desired;
    do {
      desired = expected + arg;
    } while (!std::atomic_compare_exchange_weak_explicit(obj, &expected, desired, order, order));
    return expected;
}

#else

#error You do not have C++11

#endif // C++11

#endif // MYATOMICS_H
