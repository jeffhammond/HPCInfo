void foo(int n, double x[n])
{
#pragma omp for simd
  for (int i=0; i<n; i++) {
    x[i] *= 2.0;
  }
}
