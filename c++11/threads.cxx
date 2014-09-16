#include <iostream>
#include <thread>

void fnoarg(int t)
{
    std::this_thread::sleep_for( std::chrono::seconds(t) );
    std::cout << "fnoarg running on thread " << std::this_thread::get_id() << std::endl;
}

int main(int argc, char* argv[])
{
    int nthreads = std::thread::hardware_concurrency();
    std::cout << "nthreads = " << nthreads << std::endl;

    std::thread * pool = new std::thread[nthreads];
    for (int i=0; i<nthreads; ++i) {
        pool[i] = std::thread(fnoarg,i);
    }
    for (int i=0; i<nthreads; ++i) {
        pool[i].join();
    }
    delete[] pool;

    return 0;
}
