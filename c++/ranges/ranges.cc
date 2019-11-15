#include <iostream>
#include <chrono>

#include <range/v3/core.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/slice.hpp>
#include <range/v3/view/cycle.hpp>
#include <range/v3/view/repeat.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/reverse.hpp>
#include <range/v3/view/sliding.hpp>

#include <boost/range/irange.hpp>

static inline double wtime(void)
{
    using t = std::chrono::high_resolution_clock;
    auto c = t::now().time_since_epoch().count();
    auto n = t::period::num;
    auto d = t::period::den;
    double r = static_cast<double>(c)/static_cast<double>(d)*static_cast<double>(n);
    return r;
}

int main(int argc, char* argv[])
{
    int n = (argc>1) ? std::atoi(argv[1]) : 10;
    {
        std::cout << "standard\n";
        double k(0);
        auto t0 = wtime();
        for (int i=0; i<n; ++i) k+=1.0;
        auto t1 = wtime();
        std::cout << "dt = " << t1-t0 << "\n";
        std::cout << "k = " << k << "\n";
    }
    {
        auto rvi = boost::irange(0,n);

        //std::cout << "boost::irange(0,n)";
        //for (auto const & i : rvi) std::cout << i << ",";
        //std::cout << "\n";

        std::cout << "boost::irange\n";
        double k(0);
        auto t0 = wtime();
        for (auto const & i : rvi) k+=1.0;
        auto t1 = wtime();
        std::cout << "dt = " << t1-t0 << "\n";
        std::cout << "k = " << k << "\n";
    }
    {
        auto rvi = ranges::view::iota(0,n);

        //std::cout << "ranges::view::iota(0,n) = ";
        //for (auto const & i : rvi) std::cout << i << ",";
        //std::cout << "\n";

        std::cout << "ranges::view::iota\n";
        double k(0);
        auto t0 = wtime();
        for (auto const & i : rvi) k+=1.0;
        auto t1 = wtime();
        std::cout << "dt = " << t1-t0 << "\n";
        std::cout << "k = " << k << "\n";
    }
    {
        auto rvi = ranges::view::iota(0) | ranges::view::slice(0,n);

        //std::cout << "ranges::view::iota(0) | ranges::view::slice(0,n) = ";
        //for (auto const & i : rvi) std::cout << i << ",";
        //std::cout << "\n";

        std::cout << "ranges::view::iota | ranges::view::slice\n";
        double k(0);
        auto t0 = wtime();
        for (auto const & i : rvi) k+=1.0;
        auto t1 = wtime();
        std::cout << "dt = " << t1-t0 << "\n";
        std::cout << "k = " << k << "\n";
    }

    return 0;
}
