double foo(int N, double * B, double * C)
{
    double sum = 0.0;
    #pragma omp target map(to:B,C) map(tofrom:sum)
    #pragma omp teams distribute parallel for reduction(+:sum)
    for (int i=0; i<N; i++){
        sum += B[i] + C[i];
    } 
    return sum;
}
