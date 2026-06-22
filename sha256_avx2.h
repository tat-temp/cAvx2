#ifndef SHA256_AVX2_H
#define SHA256_AVX2_H

#include <cstdint>

void sha256avx2_8B(
    const uint8_t* data0, const uint8_t* data1, const uint8_t* data2, const uint8_t* data3,
    const uint8_t* data4, const uint8_t* data5, const uint8_t* data6, const uint8_t* data7,
    unsigned char* hash0, unsigned char* hash1, unsigned char* hash2, unsigned char* hash3,
    unsigned char* hash4, unsigned char* hash5, unsigned char* hash6, unsigned char* hash7
);

// Specialized single-block SHA-256 for 8 compressed public keys (hot path).
// in[i]  : >= 36-byte buffer holding the 33-byte pubkey followed by the SHA
//          padding start {0x80,0x00,0x00} at offsets [33..35].
// out[i] : 32-byte big-endian digest.
void sha256avx2_8B_33(const uint8_t in[8][64], uint8_t out[8][32]);

#endif // SHA256_AVX2_H

