#include <atomic>
#include <complex>
#include <iostream>

int main()
{
    std::atomic<long long> all;
    std::atomic<double> ad;
    std::atomic< std::complex<double> > acd;
    std::cout << std::atomic_is_lock_free(&all) << "\n";
    std::cout << std::atomic_is_lock_free(&ad) << "\n";
    std::cout << std::atomic_is_lock_free(&acd) << "\n";
}
