#include <cstdlib>
#include <cstdint>
#include <cstddef>

#include <iostream>
#include <iomanip>

#include <immintrin.h>

#ifdef _OPENMP
#define PRAGMA_SIMD _Pragma("omp simd")
#else
#define PRAGMA_SIMD
#endif

typedef union u {
    uint32_t f32;
    uint16_t bf16[2];
} bridge;

int main(int argc, char* argv[])
{
    const size_t n = (argc>1) ? std::atol(argv[1]) : 4096;

    uint16_t * bf = new uint16_t[n];
    uint32_t * f1 = new uint32_t[n];
    uint32_t * f2 = new uint32_t[n];
    uint32_t * f3 = new uint32_t[n];
    uint32_t * f4 = new uint32_t[n];
    uint32_t * f5 = new uint32_t[n];

    for (size_t i=0; i<n; i+=16) {
        for (size_t j=0; j<16; ++j) {
            bf[i+j] = j+1;
        }
    }

    for (size_t i=0; i<n; ++i) {
        f1[i] = 0;
        f2[i] = 0;
        f3[i] = 0;
        f4[i] = 0;
        f5[i] = 0;
    }

    uint16_t * pf1 = (uint16_t*)f1;
    pf1++;
    PRAGMA_SIMD
    for (size_t i=0; i<n; i++) {
        memcpy(pf1,&bf[i],sizeof(uint16_t));
        pf1+=2;
    }

    PRAGMA_SIMD
    for (size_t i=0; i<n; i++) {
        bridge b = {0};
        b.bf16[1] = bf[i];
        f2[i] = b.f32;
    }

    PRAGMA_SIMD
    for (size_t i=0; i<n; i++) {
        uint16_t b[2] = {0,bf[i]};
        f3[i] = *(uint32_t*)&b;
    }

    // BROKEN
    const __m256i zeros = _mm256_setzero_si256();
    for (size_t i=0; i<n; i+=8) {
        __m256i upper = _mm256_load_si256((__m256i*)&bf[i]);
        __m256i blend = _mm256_unpacklo_epi16(zeros,upper);
        _mm256_store_si256((__m256i*)&f4[i],blend);
    }

    for (size_t i=0; i<n; i+=8) {
        __m128i a = _mm_load_si128((__m128i*)&bf[i]);
        __m256i b = _mm256_cvtepu16_epi32(a);
        __m256i c = _mm256_slli_epi32(b,16);
        _mm256_store_si256((__m256i*)&f5[i],c);
    }

    for (size_t i=0; i<n; i++) {
        std::cout << std::setw(10) << i << ": "
                  << std::setw(10) << f1[i]
                  << std::setw(10) << f2[i]
                  << std::setw(10) << f3[i]
                  << std::setw(10) << f4[i]
                  << std::setw(10) << f5[i] << "\n";
    }

    return 0;
}
