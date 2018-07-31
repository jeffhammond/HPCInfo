#include <cstdlib>
#include <cstdint>
#include <cstddef>

#include <iostream>
#include <iomanip>

#include <immintrin.h>

#ifdef _OPENMP
#define PRAGMA_SIMD _Pragma("omp simd simdlen(16)")
#else
#define PRAGMA_SIMD
#endif

typedef union u {
    uint32_t f32;
    uint16_t bf16[2];
} bridge;

int main(int argc, char* argv[])
{
    const int n = (argc>1) ? std::atoi(argv[1]) : 4096;

    uint16_t * bf = new uint16_t[n];
    uint32_t * f1 = new uint32_t[n];
    uint32_t * f2 = new uint32_t[n];
    uint32_t * f3 = new uint32_t[n];
    uint16_t * f4 = new uint16_t[2*n];
    uint32_t * f5 = new uint32_t[n];
    uint32_t * f6 = new uint32_t[n];

    for (int i=0; i<n; i+=16) {
        for (int j=0; j<16; ++j) {
            bf[i+j] = j+1;
        }
    }

    for (int i=0; i<n; ++i) {
        f1[i] = 0;
        f2[i] = 0;
        f3[i] = 0;
        f4[i] = 0; f4[2*i] = 0;
        f5[i] = 0;
        f6[i] = 0;
    }

    uint16_t * pf1 = (uint16_t*)f1;
    pf1++;
    PRAGMA_SIMD
    for (int i=0; i<n; i++) {
        memcpy(pf1,&bf[i],sizeof(uint16_t));
        pf1+=2;
    }

    PRAGMA_SIMD
    for (int i=0; i<n; i++) {
        bridge b = {0};
        b.bf16[1] = bf[i];
        f2[i] = b.f32;
    }

    PRAGMA_SIMD
    for (int i=0; i<n; i++) {
        uint16_t b[2] = {0,bf[i]};
        f3[i] = *(uint32_t*)&b;
    }

    PRAGMA_SIMD
    for (int i=0; i<n; i++) {
        f4[2*i+1] = bf[i];
    }

    uint32_t * pf5 = (uint32_t*)f5;
    PRAGMA_SIMD
    for (int i=0; i<n; i++) {
        pf5[i] = ((uint32_t)bf[i]) << 16;
    }

    for (int i=0; i<n; i+=8) {
        __m128i a = _mm_load_si128((__m128i*)&bf[i]);
        __m256i b = _mm256_cvtepu16_epi32(a);
        __m256i c = _mm256_slli_epi32(b,16);
        _mm256_store_si256((__m256i*)&f6[i],c);
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
