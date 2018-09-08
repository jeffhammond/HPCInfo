#include <cstdlib>
#include <cstdint>
#include <cstddef>

#include <iostream>
#include <iomanip>

#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#define PRAGMA_SIMD _Pragma("omp simd simdlen(16)")
#else
double omp_get_time() { return 0.0; }
#define PRAGMA_SIMD
#endif

int main(int argc, char* argv[])
{
    const int n = (argc>1) ? std::atoi(argv[1]) : 4096;

    uint16_t * ibf16 = new uint16_t[n];
    uint16_t * if16  = new uint16_t[n];
    float    * f32   = new float[n];
    uint16_t * obf16 = new uint16_t[n];
    uint16_t * of16  = new uint16_t[n];

    for (int i=0; i<n; i++) {
        f32[i] = (float)i;
    }

    for (int i=0; i<n; ++i) {
        uint16_t j = i % 16384;
        ibf16[i] = j;
        if16[i]  = j;
        obf16[i] = j;
        of16[i]  = j;
    }

    for (int i=0; i<n; i+=8) {
        __m128i a = _mm_load_si128((__m128i*)&ibf16[i]);
        __m256i b = _mm256_cvtepu16_epi32(a);
        __m256i c = _mm256_slli_epi32(b,16);

        _mm_store_si128((__m128i*)&obf16[i],c);
    }

    for (int i=0; i<n; i++) {
        std::cout << std::setw(10) << i << ": "
                  << std::setw(10) << f1[i]
                  << std::setw(10) << f2[i]
                  << std::setw(10) << f3[i]
                  << std::setw(10) << f4[2*i+1]*UINT16_MAX
                  << std::setw(10) << f5[i]
                  << std::setw(10) << f6[i] << "\n";
    }

    return 0;
}
