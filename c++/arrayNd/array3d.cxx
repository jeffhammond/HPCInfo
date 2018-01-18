#include <boost/multi_array.hpp>
#include <iostream>

// safety: array dimensions and data tied into one, can't mix them up!
// also: type safety.
void print(const boost::multi_array<double, 3>& array)
{
    for(int i=0; i<array.shape()[0]; i++) {
        for(int j=0; j<array.shape()[1]; j++) {
            for(int k=0; k<array.shape()[2]; k++) {
                std::cout << array[i][j][k] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}


int main(int argc, char **argv)
{
    int m = (argc>1) ? atoi(argv[1]) : 4;
    int n = (argc>2) ? atoi(argv[2]) : 5;
    int o = (argc>3) ? atoi(argv[3]) : 2;

    boost::multi_array<double, 3> A(boost::extents[o][n][m]);
    for(int i=0; i<o; i++)
        for(int j=0; j<n; j++)
            for(int k=0; k<m; k++)
                A[i][j][k] = i * m * n + j * m + k;

    boost::multi_array<double, 3> B(boost::extents[o][n][m]);
    for(int i=0; i<o; i++)
        for(int j=0; j<n; j++)
            for(int k=0; k<m; k++)
                B[i][j][k] = i * m * n + j * m + k + 0.5;
    print(A);
    print(B);

    // safe copies, would even work with differing dimensions:
    B = A;
    print(B);

    // RAII: no manual free, no memory leaks
    return 0;
}
