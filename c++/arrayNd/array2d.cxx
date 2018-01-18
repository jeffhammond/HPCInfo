#include <boost/multi_array.hpp>
#include <iostream>

// safety: array dimensions and data tied into one, can't mix them up!
// also: type safety.
void print(const boost::multi_array<double, 2>& array)
{
    for(int i=0; i<array.shape()[0]; i++) {
        for(int j=0; j<array.shape()[1]; j++) {
            std::cout << array[i][j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n";
}


int main(int argc, char **argv)
{
    int m = (argc>1) ? atoi(argv[1]) : 4;
    int n = (argc>2) ? atoi(argv[2]) : 5;

    boost::multi_array<double, 2> A(boost::extents[n][m]);
    for(int i=0; i<n; i++)
        for(int j=0; j<m; j++)
                A[i][j] = i * m + j;

    boost::multi_array<double, 2> B(boost::extents[n][m]);
    for(int i=0; i<n; i++)
        for(int j=0; j<m; j++)
                B[i][j] = i * m + j + 0.5;
    print(A);
    print(B);

    // safe copies, would even work with differing dimensions:
    B = A;
    print(B);

    // RAII: no manual free, no memory leaks
    return 0;
}
