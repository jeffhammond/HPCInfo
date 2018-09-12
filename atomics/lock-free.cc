#include <atomic>
#include <complex>
#include <iostream>

int main()
{
    std::atomic<long long> all{0};
    std::atomic<double> ad{0};
    std::atomic< std::complex<double> > acd{0};
    std::cout << std::atomic_is_lock_free(&all) << "\n";
    std::cout << std::atomic_is_lock_free(&ad) << "\n";
    std::cout << std::atomic_is_lock_free(&acd) << "\n";
    all = 1;
    ad  = 1.0;
    acd = std::complex<double>{1.0,0.0};
    all.fetch_add(1);
#if defined(__cplusplus) && !(__cplusplus<=202000L)
    ad.fetch_add(1.0);
    acd.fetch_add(std::complex<double>{1.0,0.0});
#endif
}
