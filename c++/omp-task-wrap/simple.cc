#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[])
{
    int const n = (argc > 1) ? std::atoi(argv[1]) : 100;
    double * A = new double[n];
    double * B = new double[n];
    double * C = new double[n];
    #pragma omp parallel
    #pragma omp single
    {
        #pragma omp task depend(out:A[0])
        {
            for (int i=0; i<n; ++i) {
                A[i] = i;
            }
        }
        #pragma omp task depend(in:A[0]) depend(out:B[0])
        {
            for (int i=0; i<n; ++i) {
                B[i] = A[i];
            }
        }
        #pragma omp task depend(in:B[0]) depend(out:C[0])
        {
            for (int i=0; i<n; ++i) {
                C[i] = B[i];
            }
        }
    }
    return 0;
}
