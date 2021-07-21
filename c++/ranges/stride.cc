#include <iostream>
#include <ranges>
#include <execution>
#include <algorithm>

int main(int argc, char* argv[])
{
    int start = 0;
    int end   = 30;

    auto even = [](int i) { return 0 == i % 2; };
    auto scale = [](int i) { return i*3; };

    auto x = std::views::iota(start, end);
    auto y = std::views::iota(start, end) | std::views::transform(scale);
    auto z = std::views::iota(start, end) | std::ranges::views::filter(even);

    std::cout << std::endl << "contiguous" << std::endl;
    std::for_each( std::begin(x), std::end(x), [] (int i) {
        std::cout << i << "\n";
    });

    std::cout << std::endl << "stride 1" << std::endl;
    std::for_each( std::begin(y), std::end(y), [] (int i) {
        std::cout << i << "\n";
    });

    std::cout << std::endl << "stride 2" << std::endl;
    std::for_each( std::begin(z), std::end(z), [] (int i) {
        std::cout << i << "\n";
    });

    return 0;
}
