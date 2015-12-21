#include <iostream>
#include "tbb/tbb.h"

int main(int argc, char* argv[])
{
    int n = tbb::task_scheduler_init::default_num_threads();
    tbb::parallel_for(size_t(0), size_t(n), size_t(1),
            [=](size_t i)
            {
                std::cout << i;
            });
    std::cout << std::endl;
    return 0;
}
