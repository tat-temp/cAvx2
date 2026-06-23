#ifndef SIMD_FIELD_H
#define SIMD_FIELD_H
// ---------------------------------------------------------------------------
// Tier 3: 4-way (AVX2) secp256k1 field arithmetic.  Radix 2^26, 10 limbs.
//
// One fe4 holds FOUR independent field elements, stored SoA ("limb-major"):
// fe4.l[i] is a YMM whose lane k carries the 26-bit limb i of element k.  The
// 4-way multiply maps every scalar 64-bit op of a radix-2^26 schoolbook to one
// AVX2 intrinsic (vpmuludq / vpaddq / shift / and), so four field multiplies
// run in the four 64-bit lanes at once.
//
// Field prime p = 2^256 - 2^32 - 977, so  2^256 = 2^32 + 977 = R = 0x1000003D1.
// A high limb at 2^260 (limb >=10) folds as 2^260 = 16*R = 2^36 + 15632.
//
// Results are reduced to < 2^256 and congruent (NOT necessarily < p) -- the
// same contract the scalar ModMulK1 keeps.  Bit-exactness vs the scalar field
// is gated by --simdtest (see simd_test.cpp).
// ---------------------------------------------------------------------------
#include <immintrin.h>
#include <cstdint>

struct fe4 { __m256i l[10]; };   // four field elements, one per 64-bit lane

static const uint64_t SF_M26 = 0x3FFFFFFULL;   // 2^26 - 1
static const uint64_t SF_M22 = 0x3FFFFFULL;    // 2^22 - 1 (top limb of a 256-bit value)

#define SF_MUL(x,y) _mm256_mul_epu32((x),(y))
#define SF_ADD(x,y) _mm256_add_epi64((x),(y))
#define SF_SHR(x,n) _mm256_srli_epi64((x),(n))
#define SF_SHL(x,n) _mm256_slli_epi64((x),(n))
#define SF_AND(x,y) _mm256_and_si256((x),(y))

// Extract 26-bit limb i (i=0..9) from a 256-bit little-endian value b[0..3].
static inline uint64_t sf_get_limb(const uint64_t* b, int i) {
    int bit = 26 * i, w = bit >> 6, off = bit & 63;
    uint64_t lo = b[w] >> off;
    if (off > 38 && w < 3) lo |= b[w + 1] << (64 - off);  // 64 - 26 = 38
    return lo & SF_M26;
}

// Pack four canonical 256-bit values (each uint64[4], < 2^256) into an fe4.
// Lane k <-> element k.
static inline fe4 fe4_pack(const uint64_t* a0, const uint64_t* a1,
                           const uint64_t* a2, const uint64_t* a3) {
    fe4 r;
    for (int i = 0; i < 10; i++)
        r.l[i] = _mm256_set_epi64x((long long)sf_get_limb(a3, i),
                                   (long long)sf_get_limb(a2, i),
                                   (long long)sf_get_limb(a1, i),
                                   (long long)sf_get_limb(a0, i));
    return r;
}

// Unpack one lane (k) back to a 256-bit little-endian value out[0..3].
// Assumes the fe4 is normalized (each limb < 2^26) and the value < 2^256.
static inline void fe4_unpack_lane(const fe4& x, int k, uint64_t out[4]) {
    alignas(32) uint64_t lane[10][4];
    for (int i = 0; i < 10; i++) _mm256_store_si256((__m256i*)lane[i], x.l[i]);
    uint64_t w[5] = {0, 0, 0, 0, 0};
    auto add64 = [&](int k, uint64_t v) {            // w += v at limb k, carry up
        while (v && k < 5) {
            uint64_t s = w[k] + v;
            v = (s < w[k]) ? 1ULL : 0ULL;            // carry out
            w[k] = s; k++;
        }
    };
    for (int i = 0; i < 10; i++) {
        uint64_t L = lane[i][k];
        int bit = 26 * i, word = bit >> 6, off = bit & 63;
        add64(word, L << off);
        if (off) add64(word + 1, L >> (64 - off));
    }
    out[0] = w[0]; out[1] = w[1]; out[2] = w[2]; out[3] = w[3];
}

// Carry-normalize limbs [0..n-1] in place; returns the carry out of limb n-1
// (bits >= 26*n).
static inline __m256i sf_carry(__m256i* d, int n, __m256i mask26) {
    __m256i carry = _mm256_setzero_si256();
    for (int k = 0; k < n; k++) {
        d[k] = SF_ADD(d[k], carry);
        carry = SF_SHR(d[k], 26);
        d[k] = SF_AND(d[k], mask26);
    }
    return carry;
}

// out = a * b (mod p), 4-way.  out may alias neither a nor b.
static inline void fe4_mul(fe4& out, const fe4& a, const fe4& b) {
    const __m256i mask26 = _mm256_set1_epi64x(SF_M26);
    const __m256i c15632 = _mm256_set1_epi64x(15632);

    // --- schoolbook convolution: d[0..18], d[19] reserved for the carry ---
    __m256i d[20];
    for (int k = 0; k < 20; k++) d[k] = _mm256_setzero_si256();
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++)
            d[i + j] = SF_ADD(d[i + j], SF_MUL(a.l[i], b.l[j]));

    // normalize the full 512-bit product so every limb < 2^26
    sf_carry(d, 20, mask26);   // carry out of d[19] is 0 (product < 2^512)

    // --- fold the high half (limbs 10..19, weight 2^260 = 16R) into 0..10 ---
    __m256i acc[11];
    for (int k = 0; k < 10; k++) acc[k] = d[k];
    acc[10] = _mm256_setzero_si256();
    for (int m = 0; m < 10; m++) {              // k = 10 + m
        acc[m]     = SF_ADD(acc[m],     SF_MUL(d[10 + m], c15632)); // *15632
        acc[m + 1] = SF_ADD(acc[m + 1], SF_SHL(d[10 + m], 10));     // *2^36 -> limb+1 <<10
    }
    // collapse limb 10 (weight 2^260 = 16R) back into 0..1
    acc[0] = SF_ADD(acc[0], SF_MUL(acc[10], c15632));
    acc[1] = SF_ADD(acc[1], SF_SHL(acc[10], 10));

    // normalize 0..9; fold the carry out of limb 9 (weight 2^260 = 16R)
    __m256i cc = sf_carry(acc, 10, mask26);
    acc[0] = SF_ADD(acc[0], SF_MUL(cc, c15632));
    acc[1] = SF_ADD(acc[1], SF_SHL(cc, 10));
    sf_carry(acc, 10, mask26);                  // carry out now 0 (value < 2^260)

    // --- final 2^256 cleanup: fold limb-9 bits >=256 via R = 2^32 + 977 ---
    const __m256i mask22  = _mm256_set1_epi64x(SF_M22);
    const __m256i c977    = _mm256_set1_epi64x(977);
    for (int pass = 0; pass < 2; pass++) {
        __m256i hi = SF_SHR(acc[9], 22);        // bits 256.. -> coeff of 2^256 = R
        acc[9] = SF_AND(acc[9], mask22);
        acc[0] = SF_ADD(acc[0], SF_MUL(hi, c977)); // hi*977 at limb 0
        acc[1] = SF_ADD(acc[1], SF_SHL(hi, 6));    // hi*2^32 at limb 1 (<<6)
        sf_carry(acc, 10, mask26);
    }
    for (int k = 0; k < 10; k++) out.l[k] = acc[k];
}

// out = a^2 (mod p).  Correctness-first: defer to the multiply (the dedicated
// squaring kernel, ~half the products, is a later optimization).
static inline void fe4_sqr(fe4& out, const fe4& a) { fe4_mul(out, a, a); }

#endif // SIMD_FIELD_H
