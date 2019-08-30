// inspired by https://www.bfilipek.com/2019/03/cpp17indetail-done.html

#include <iostream>
#include <vector>
#include <memory_resource>

int main(int argc, char* argv[])
{
    /*
    char buffer[64] = {};
    std::fill_n(std::begin(buffer), std::size(buffer)-1, '_');
    std::cout << buffer << '\n';
    std::pmr::monotonic_buffer_resource pool{ std::data(buffer), std::size(buffer) };
    */

    size_t n = (argc>1) ? atol(argv[1]) : 64;
    char * buffer = new char[n];
    std::fill_n(buffer, n-1, '_');
    std::cout << "buffer=" << buffer << '\n';

    std::pmr::monotonic_buffer_resource pool{ buffer, n };

    std::pmr::vector<char> vec{&pool};
    for (char ch='a'; ch <= 'z'; ++ch) {
        vec.push_back(ch);
    }

    std::cout << "buffer=" << buffer << '\n';
    //std::cout << "vector=" << vec << '\n';
    std::cout << "vector.size=" << std::size(vec) << ", vector.data=" << std::data(vec) << "\n";
    std::cout << "vector=";
    for (auto && c : vec) {
        std::cout << c;
    }
    std::cout << "\n";
}

