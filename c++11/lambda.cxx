#include <iostream>
#include <vector>
#include <numeric>

int main(int argc, char* argv[])
{
    std::vector<double> v{1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};

    for (auto& x : v) {
        x += 1.;
    }

    for (auto& x : v) {
        std::cout << x << ",";
    }
    std::cout << std::endl;

    return 0;
}
