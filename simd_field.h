#ifndef SIMD_FIELD_H
#define SIMD_FIELD_H
// ---------------------------------------------------------------------------
// Tier 3: 4-way (AVX2) secp256k1 field arithmetic.  Radix 2^29, 9 limbs.
//
// One fe4 holds FOUR independent field elements, stored SoA ("limb-major"):
// fe4.l[i] is a YMM whose lane k carries the 29-bit limb i of element k.  The
// 4-way multiply maps every scalar 64-bit op of a radix-2^29 schoolbook to one
// AVX2 intrinsic (vpmuludq / vpaddq / shift / and), so four field multiplies
// run in the four 64-bit lanes at once.
//
// Field prime p = 2^256 - 2^32 - 977, so  2^256 = 2^32 + 977 = R = 0x1000003D1.
// A high limb at 2^261 (limb >=9) folds as 2^261 = 32*R = 2^37 + 31264.
//
// Radix-2^29 over 2^26: 81 vpmuludq per multiply instead of 100, and 9 limbs to
// carry instead of 10.  Headroom: a product limb sums <=9 terms of <2^58, so
// <2^61.2 -- fits uint64 with ~2.8 bits to spare.  Every value handed to
// vpmuludq stays < 2^32 *provided inputs are < 2^256* (then limb 8 < 2^24): the
// reduction guarantees that on output and fe4_pack guarantees it on input.
//
// Results are reduced to < 2^256 and congruent (NOT necessarily < p) -- the
// same contract the scalar ModMulK1 keeps.  Bit-exactness vs the scalar field
// is gated by ./simdtest (see simd_test.cpp).
// ---------------------------------------------------------------------------
#include <immintrin.h>
#include <cstdint>

struct fe4 { __m256i l[9]; };   // four field elements, one per 64-bit lane

static const uint64_t SF_M29 = 0x1FFFFFFFULL;  // 2^29 - 1
static const uint64_t SF_M24 = 0xFFFFFFULL;    // 2^24 - 1 (top limb of a 256-bit value)

#define SF_MUL(x,y) _mm256_mul_epu32((x),(y))
#define SF_ADD(x,y) _mm256_add_epi64((x),(y))
#define SF_SHR(x,n) _mm256_srli_epi64((x),(n))
#define SF_SHL(x,n) _mm256_slli_epi64((x),(n))
#define SF_AND(x,y) _mm256_and_si256((x),(y))

// Extract 29-bit limb i (i=0..8) from a 256-bit little-endian value b[0..3].
static inline uint64_t sf_get_limb(const uint64_t* b, int i) {
    int bit = 29 * i, w = bit >> 6, off = bit & 63;
    uint64_t lo = b[w] >> off;
    if (off > 35 && w < 3) lo |= b[w + 1] << (64 - off);  // 64 - 29 = 35
    return lo & SF_M29;
}

// Pack four canonical 256-bit values (each uint64[4], < 2^256) into an fe4.
static inline fe4 fe4_pack(const uint64_t* a0, const uint64_t* a1,
                           const uint64_t* a2, const uint64_t* a3) {
    fe4 r;
    for (int i = 0; i < 9; i++)
        r.l[i] = _mm256_set_epi64x((long long)sf_get_limb(a3, i),
                                   (long long)sf_get_limb(a2, i),
                                   (long long)sf_get_limb(a1, i),
                                   (long long)sf_get_limb(a0, i));
    return r;
}

// Unpack one lane (k) back to a 256-bit little-endian value out[0..3].
// Assumes the fe4 is normalized (each limb < 2^29) and the value < 2^256.
static inline void fe4_unpack_lane(const fe4& x, int k, uint64_t out[4]) {
    alignas(32) uint64_t lane[9][4];
    for (int i = 0; i < 9; i++) _mm256_store_si256((__m256i*)lane[i], x.l[i]);
    uint64_t w[5] = {0, 0, 0, 0, 0};
    auto add64 = [&](int kk, uint64_t v) {           // w += v at limb kk, carry up
        while (v && kk < 5) {
            uint64_t s = w[kk] + v;
            v = (s < w[kk]) ? 1ULL : 0ULL;
            w[kk] = s; kk++;
        }
    };
    for (int i = 0; i < 9; i++) {
        uint64_t L = lane[i][k];
        int bit = 29 * i, word = bit >> 6, off = bit & 63;
        add64(word, L << off);
        if (off) add64(word + 1, L >> (64 - off));
    }
    out[0] = w[0]; out[1] = w[1]; out[2] = w[2]; out[3] = w[3];
}

// Carry-normalize limbs [0..n-1] in place; returns the carry out of limb n-1.
static inline __m256i sf_carry(__m256i* d, int n, __m256i mask) {
    __m256i carry = _mm256_setzero_si256();
    for (int k = 0; k < n; k++) {
        d[k] = SF_ADD(d[k], carry);
        carry = SF_SHR(d[k], 29);
        d[k] = SF_AND(d[k], mask);
    }
    return carry;
}

// Reduce a raw 17-limb convolution d[0..16] (d[17] must be 0) to a normalized
// fe4 < 2^256.  Shared by multiply and square.  Three carry sweeps: normalize
// the product, fold the high half (2^261 = 32R), then fold the leftover carry
// (32R) AND the 2^256 cleanup (limb-8 bits >=256, via R = 2^32+977) together in
// one final settle.  Every value fed to vpmuludq stays < 2^32 for <2^256 inputs.
static inline void sf_reduce(__m256i* d, fe4& out) {
    const __m256i m29     = _mm256_set1_epi64x(SF_M29);
    const __m256i c31264  = _mm256_set1_epi64x(31264);
    const __m256i m24     = _mm256_set1_epi64x(SF_M24);
    const __m256i c977    = _mm256_set1_epi64x(977);

    sf_carry(d, 18, m29);                  // sweep 1: full product normalized (<2^29)

    // fold the high half (limbs 9..17, weight 2^261 = 32R) into acc[0..9]
    __m256i acc[10];
    for (int k = 0; k < 9; k++) acc[k] = d[k];
    acc[9] = _mm256_setzero_si256();
    for (int m = 0; m < 9; m++) {          // k = 9 + m
        acc[m]     = SF_ADD(acc[m],     SF_MUL(d[9 + m], c31264)); // *31264
        acc[m + 1] = SF_ADD(acc[m + 1], SF_SHL(d[9 + m], 8));      // *2^37 -> limb+1 <<8
    }
    acc[0] = SF_ADD(acc[0], SF_MUL(acc[9], c31264));  // collapse limb 9 (32R)
    acc[1] = SF_ADD(acc[1], SF_SHL(acc[9], 8));

    __m256i cc = sf_carry(acc, 9, m29);    // sweep 2: -> cc = carry out of limb 8 (>=2^261)

    // fold cc (32R) and the 2^256 cleanup together, then one settle sweep
    acc[0] = SF_ADD(acc[0], SF_MUL(cc, c31264));
    acc[1] = SF_ADD(acc[1], SF_SHL(cc, 8));
    __m256i hi = SF_SHR(acc[8], 24);       // limb-8 bits >=256 -> coeff of 2^256 = R
    acc[8] = SF_AND(acc[8], m24);
    acc[0] = SF_ADD(acc[0], SF_MUL(hi, c977)); // hi*977 at limb 0
    acc[1] = SF_ADD(acc[1], SF_SHL(hi, 3));    // hi*2^32 at limb 1 (<<3)

    sf_carry(acc, 9, m29);                 // sweep 3: settle; value < 2^256

    for (int k = 0; k < 9; k++) out.l[k] = acc[k];
}

// out = a * b (mod p), 4-way.  out may alias a and/or b.  Scatter accumulation
// (d[i+j] += a_i*b_j): measured ~1.8x faster than a column-at-a-time register
// accumulator -- the many independent partial sums give the out-of-order engine
// the ILP to hide vpmuludq latency, which one serial per-column chain does not.
static inline void fe4_mul(fe4& out, const fe4& a, const fe4& b) {
    __m256i d[18];
    for (int k = 0; k < 18; k++) d[k] = _mm256_setzero_si256();
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++)
            d[i + j] = SF_ADD(d[i + j], SF_MUL(a.l[i], b.l[j]));
    sf_reduce(d, out);
}

// out = a^2 (mod p), 4-way.  Dedicated squaring: pre-double one operand so each
// off-diagonal product a_i*a_j (i<j) is counted twice, plus the diagonal a_i^2
// -- ~45 vpmuludq instead of 81, single pass.  (SIMD-faster than scalar even on
// the dev laptop.)
static inline void fe4_sqr(fe4& out, const fe4& a) {
    __m256i a2[9];
    for (int i = 0; i < 9; i++) a2[i] = SF_SHL(a.l[i], 1);   // 2*a_i (< 2^30)
    __m256i d[18];
    for (int k = 0; k < 18; k++) d[k] = _mm256_setzero_si256();
    for (int i = 0; i < 9; i++) {
        d[2 * i] = SF_ADD(d[2 * i], SF_MUL(a.l[i], a.l[i]));         // diagonal a_i^2
        for (int j = i + 1; j < 9; j++)
            d[i + j] = SF_ADD(d[i + j], SF_MUL(a2[i], a.l[j]));      // 2*a_i*a_j
    }
    sf_reduce(d, out);
}

// --- add / sub / neg / normalize (radix-2^29, magnitude-tracked) ------------
// In a redundant radix, add and sub are limb-wise (no carry) -- cheap -- and the
// result limbs just grow ("magnitude").  Negation/subtraction use a multiple of
// p, 2*(m+1)*P29, large enough that no limb underflows for an input of magnitude
// m (libsecp256k1's trick).  fe4_mul/fe4_sqr require magnitude-1 (normalized)
// inputs (so the column sums stay < 2^64); call fe4_normalize before feeding a
// grown value into a multiply.  P29 = the field prime's 9 limbs, set by sf_init.
static __m256i sf_P29[9];

static inline void sf_init(const uint64_t p_bits[4]) {
    for (int i = 0; i < 9; i++)
        sf_P29[i] = _mm256_set1_epi64x((long long)sf_get_limb(p_bits, i));
}

// r = a + b (limb-wise).  Magnitudes add: mag(r) = mag(a) + mag(b).
static inline void fe4_add(fe4& r, const fe4& a, const fe4& b) {
    for (int i = 0; i < 9; i++) r.l[i] = SF_ADD(a.l[i], b.l[i]);
}

// r = -a (mod p), for an input of magnitude m.  mag(r) ~ 2*(m+1).
static inline void fe4_neg(fe4& r, const fe4& a, int m) {
    __m256i K = _mm256_set1_epi64x(2 * (m + 1));
    for (int i = 0; i < 9; i++)
        r.l[i] = _mm256_sub_epi64(SF_MUL(K, sf_P29[i]), a.l[i]);
}

// r = a - b (mod p); b has magnitude mb.  mag(r) ~ mag(a) + 2*(mb+1).
static inline void fe4_sub(fe4& r, const fe4& a, const fe4& b, int mb) {
    __m256i K = _mm256_set1_epi64x(2 * (mb + 1));
    for (int i = 0; i < 9; i++)
        r.l[i] = SF_ADD(a.l[i], _mm256_sub_epi64(SF_MUL(K, sf_P29[i]), b.l[i]));
}

// Reduce any low-magnitude value (limbs < ~2^34) to < 2^256, magnitude 1.
static inline void fe4_normalize(fe4& r) {
    const __m256i m29 = _mm256_set1_epi64x(SF_M29);
    const __m256i c31264 = _mm256_set1_epi64x(31264);
    const __m256i m24 = _mm256_set1_epi64x(SF_M24);
    const __m256i c977 = _mm256_set1_epi64x(977);
    __m256i cc = sf_carry(r.l, 9, m29);               // -> bits >= 261 = 32R
    r.l[0] = SF_ADD(r.l[0], SF_MUL(cc, c31264));
    r.l[1] = SF_ADD(r.l[1], SF_SHL(cc, 8));
    __m256i hi = SF_SHR(r.l[8], 24);                  // limb-8 bits >=256 = R
    r.l[8] = SF_AND(r.l[8], m24);
    r.l[0] = SF_ADD(r.l[0], SF_MUL(hi, c977));
    r.l[1] = SF_ADD(r.l[1], SF_SHL(hi, 3));
    sf_carry(r.l, 9, m29);
}

#endif // SIMD_FIELD_H
