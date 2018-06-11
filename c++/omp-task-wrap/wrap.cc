#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <iostream>

template <typename Lambda>
void task(Lambda && body)
{
    #pragma omp task
    {
        body();
    }
}

template <typename Lambda, typename InDepend, typename InoutDepend, typename OutDepend>
void task(Lambda && body, InDepend && iref, InoutDepend && ioref, OutDepend && oref)
{
    #pragma omp task depend(in:iref) depend(inout:ioref) depend(out:oref)
    {
        body();
    }
}

int main(int argc, char* argv[])
{
    int const n = (argc > 1) ? std::atoi(argv[1]) : 100;
    double * A = new double[n];
    double * B = new double[n];
    double * C = new double[n];
    #pragma omp parallel
    #pragma omp single
    {
     /* #pragma omp task depend(out:A[0])
        {
            for (int i=0; i<n; ++i) {
                A[i] = i;
            }
        } */
        auto a = [&] () {
           std::cout << "Task A\n";
           for (int i=0; i<n; ++i) {
              A[i] = i;
           }
        };
        task(a,A[0],nullptr,nullptr);
     /* #pragma omp task depend(in:A[0]) depend(out:B[0])
        {
            for (int i=0; i<n; ++i) {
                B[i] = A[i];
            }
        } */
        auto b = [&] () {
           std::cout << "Task B\n";
           for (int i=0; i<n; ++i) {
               B[i] = A[i];
           }
        };
        task(b,A[0],nullptr,B[0]);
     /* #pragma omp task depend(in:B[0]) depend(out:C[0])
        {
            for (int i=0; i<n; ++i) {
                C[i] = B[i];
            }
        } */
        auto c = [&] () {
           std::cout << "Task C\n";
           for (int i=0; i<n; ++i) {
              C[i] = B[i];
           }
        };
        task(c,B[0],nullptr,C[0]);
    }
    return 0;
}
