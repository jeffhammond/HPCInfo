#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <iterator>
#include <vector>
 
template <typename T>
void print(std::vector<T> v)
{
    for (auto && e : v) {
        std::cout << e << ",";
    }
    std::cout << std::endl;
}

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
    size_t n = (argc > 1) ? std::atol(argv[1]) : 10;
    std::cout << "bogosort of " << n << " elements" << std::endl;

    std::vector<size_t> v(n,0);
    std::iota(v.begin(), v.end(),0);
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(v.begin(), v.end(), g);
    }
    std::cout << "Before:\n";
    print(v);

    {
        std::random_device rd;
        std::mt19937 g(rd());

        // warmup
        std::shuffle(v.begin(), v.end(), g);

        size_t i{0};
        double t0 = wtime();
        while ( !std::is_sorted(v.begin(), v.end()) ) {
            std::cout << "Iteration " << i << "\n";
            std::shuffle(v.begin(), v.end(), g);
            i++;
        }
        double t1 = wtime();
        std::cout << "Time: " << t1-t0 << std::endl;
    }

    return 0;
}
