#include <iostream>
#include <algorithm>
#include <execution>
#include <numeric>

int main(void)
{
    std::vector<double> A(1000);

#if 0
    #pragma omp workshare
    std::iota(A.begin(), A.end(), 0.0);
#endif

    #pragma omp parallel for
    std::for_each( std::begin(A), std::end(A), [&] (double x) { std::cout << x; } );


    return 0;
}
