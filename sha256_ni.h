#ifndef SHA256_NI_H
#define SHA256_NI_H
#include <cstdint>

// SHA-256 via the x86 SHA extensions (SHA-NI). Same interfaces as the AVX2 path
// (sha256_avx2.h) so the two can be A/B'd. Inputs use the identical 64-byte
// block layout: 33-byte compressed pubkey, 0x80 at [33], zeros, and the 64-bit
// bit-length (264) at [56..63] -> bytes [62]=0x01,[63]=0x08. (The AVX2 path
// hardcodes the length as a schedule constant and ignores [36..63]; SHA-NI
// reads the whole block, so those bytes must be set.)

void sha256ni_8B_33(const uint8_t in[8][64], uint8_t out[8][32]);

// Hardware SHA-256 then 8-way AVX2 RIPEMD-160 (HASH160 of a compressed pubkey).
void hash160_pubkey_8_ni(const uint8_t in[8][64], uint8_t out[8][20]);

#endif // SHA256_NI_H
