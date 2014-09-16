#include <iostream>
#include <thread>

int main(int argc, char* argv[])
{
    std::cout << "hw concurrency = " << std::thread::hardware_concurrency() << std::endl;
    return 0;
}
