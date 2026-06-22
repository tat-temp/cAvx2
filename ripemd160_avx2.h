#ifndef RIPEMD160_AVX2_H
#define RIPEMD160_AVX2_H

#include <immintrin.h>
#include <cstdint>

namespace ripemd160avx2 {

// Initialyzying Ripemd160
void Initialize(__m256i *state);

// Transform AVX2
void Transform(__m256i *state, uint8_t *blocks[8]);

// Hashing functions
void ripemd160avx2_32(
    unsigned char *i0, unsigned char *i1, unsigned char *i2, unsigned char *i3,
    unsigned char *i4, unsigned char *i5, unsigned char *i6, unsigned char *i7,
    unsigned char *d0, unsigned char *d1, unsigned char *d2, unsigned char *d3,
    unsigned char *d4, unsigned char *d5, unsigned char *d6, unsigned char *d7);

// Specialized single-block RIPEMD-160 of 8 x 32-byte inputs (hot path).
// in[i]  : 32-byte message (a SHA-256 digest); read-only, padding is implicit.
// out[i] : 20-byte digest.
void ripemd160avx2_32_fast(const uint8_t in[8][32], uint8_t out[8][20]);

// Core: RIPEMD-160 over 8 messages whose data words 0..7 are already supplied
// as transposed lane vectors (lane = message). Padding words 8..15 are added
// internally. Lets the fused pubkey hasher feed SHA output straight in.
void ripemd160_8way_words(const __m256i in_w[8], unsigned char out[8][20]);

}  // namespace ripemd160avx2

#endif  // RIPEMD160_AVX2_H
