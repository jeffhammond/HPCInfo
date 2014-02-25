/* The content is licensed under Creative Commons Attribution-Sharealike 3.0 Unported License (CC-BY-SA)
 * and by the GNU Free Documentation License (GFDL) (unversioned, with no invariant sections, front-cover
 * texts, or back-cover texts). That means that you can use this site in almost any way you like, including
 * mirroring, copying, translating, etc. All we would ask is to provide link back to cppreference.com so
 * that people know where to get the most up-to-date content. In addition to that, any modified content
 * should be released under a equivalent license so that everyone could benefit from the modified versions.
 *
 * The code below was derived from http://en.cppreference.com/w/cpp/thread/future
 *
 * All modifications fall under CC-BY-SA.
 *
 */

#include <iostream>
#include <future>
#include <thread>

int main(int argc, char * argv[])
{
    // future from a packaged_task
    std::packaged_task<int()> task([](){ return 7; }); // wrap the function
    std::future<int> f1 = task.get_future();  // get a future
    std::thread(std::move(task)).detach(); // launch on a thread

    // future from an async()
    std::future<int> f2 = std::async(std::launch::async, [](){ return 8; });

    // future from a promise
    std::promise<int> p;
    std::future<int> f3 = p.get_future();
    std::thread( [](std::promise<int> p){ p.set_value_at_thread_exit(9); }, std::move(p) ).detach();

    std::cout << "Waiting..." << std::flush;
    f1.wait();
    f2.wait();
    f3.wait();
    std::cout << "Done!\nResults are: "
              << f1.get() << ' ' << f2.get() << ' ' << f3.get() << '\n';

    return 0;
}
