#define ALIGN_BYTES 32
#define ASSUME_ALIGNED(x) x = __builtin_assume_aligned(x, ALIGN_BYTES)

void fn0(const float *restrict a0, const float *restrict a1,
         float *restrict b, int n)
{
    ASSUME_ALIGNED(a0); ASSUME_ALIGNED(a1); ASSUME_ALIGNED(b);

    for (int i = 0; i < n; ++i)
        b[i] = a0[i] + a1[i];
}

#ifdef OLD
void fn1(const float *restrict *restrict a, float *restrict b, int n)
#else
void fn1(const float *restrict a[restrict], float * restrict b, int n)
#endif
{
    ASSUME_ALIGNED(a[0]); ASSUME_ALIGNED(a[1]); ASSUME_ALIGNED(b);

    for (int i = 0; i < n; ++i)
        b[i] = a[0][i] + a[1][i];
}
