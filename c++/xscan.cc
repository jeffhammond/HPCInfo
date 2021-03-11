#include <iostream>
#include <vector>
#include <numeric>

int main(void)
{
    const int n = 100;
    std::vector<int> A(n,0);
    std::iota(A.begin(), A.end(), 0);

    std::vector<int> B(n,0);
    B[0] = 0;

    for (int i=1; i<n; ++i) {
        B[i] = B[i-1] + A[i-1];
    }

    std::vector<int> C(n,0);
    std::exclusive_scan( A.cbegin(), A.cend(), C.begin(), 0);

    for (int i=1; i<n; ++i) {
        std::cout << i << "," << A[i] << "," << B[i] << "," << C[i] << std::endl;
    }

    return 0;
}

