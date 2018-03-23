#include "CL/sycl.hpp"

template <typename T>
void run()
{
  cl::sycl::queue q(cl::sycl::cpu_selector{});
  const size_t length = 1000;
  cl::sycl::buffer<T> d_A { length };
  q.submit([&](cl::sycl::handler& h) {
    auto A = d_A.template get_access<cl::sycl::access::mode::read_write>(h);
    h.parallel_for<class nstream>(cl::sycl::range<1>{length}, [=] (cl::sycl::item<1> i) {
        A[i] = 1;
    });
  });
}

void foo()
{
#ifdef FLOAT
  run<float>();
#endif
#ifdef DOUBLE
  run<double>();
#endif
}
