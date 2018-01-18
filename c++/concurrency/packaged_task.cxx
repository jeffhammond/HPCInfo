/* The code below was derived from http://en.cppreference.com/w/cpp/thread/packaged_task */

#include <iostream>
#include <cmath>
#include <thread>
#include <future>
#include <functional>

// unique function to avoid disambiguating the std::pow overload set
int f(int x, int y) { return std::pow(x,y); }

void task_lambda()
{
    std::packaged_task<int(int,int)> task([](int a, int b) { return std::pow(a, b); });
    std::future<int> result = task.get_future();

    task(2, 9);

    std::cout << "task_lambda:\t" << result.get() << '\n';
}

void task_bind()
{
    std::packaged_task<int()> task(std::bind(f, 2, 11));
    std::future<int> result = task.get_future();

    task();

    std::cout << "task_bind:\t" << result.get() << '\n';
}

void task_thread()
{
    /* the following does not work */
    //std::packaged_task<int(int,int)> task(f);
 
    std::packaged_task<int(int,int)> task([](int a, int b) { return f(a,b); });
    std::future<int> result = task.get_future();

    std::thread task_td(std::move(task), 2, 10);
    task_td.join();

    std::cout << "task_thread:\t" << result.get() << '\n';
}

int main(int argc, char * argv[])
{
    task_lambda();
    task_bind();
    task_thread();
    return 0;
}
