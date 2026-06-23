// SHA-NI gate: bit-exactness + microbench for the hardware SHA-256 / HASH160
// path (sha256_ni.cpp) vs the trusted 8-way AVX2 path.
//   ./shanitest [iters]
#include "sha256_avx2.h"
#include "sha256_ni.h"
#include "ripemd160_avx2.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <chrono>

static void mkblock(uint8_t b[64], std::mt19937_64& rng) {
    memset(b, 0, 64);
    b[0] = 0x02 | (uint8_t)(rng() & 1);            // 02/03 prefix
    for (int i = 1; i < 33; i++) b[i] = (uint8_t)rng();  // x
    b[33] = 0x80;                                  // SHA padding
    b[62] = 0x01; b[63] = 0x08;                    // 64-bit length = 264 bits
}

static int correctness(uint64_t iters) {
    std::mt19937_64 rng(0xC0FFEEULL);
    uint64_t checks = 0, shaFail = 0, hFail = 0;
    for (uint64_t k = 0; k < iters; k += 8) {
        alignas(32) uint8_t in[8][64];
        for (int i = 0; i < 8; i++) mkblock(in[i], rng);

        alignas(32) uint8_t sa[8][32], sn[8][32];
        sha256avx2_8B_33(in, sa);
        sha256ni_8B_33(in, sn);
        alignas(32) uint8_t ha[8][20], hn[8][20];
        hash160_pubkey_8(in, ha);
        hash160_pubkey_8_ni(in, hn);

        for (int i = 0; i < 8; i++) {
            checks++;
            if (memcmp(sa[i], sn[i], 32) != 0) {
                if (shaFail < 4) {
                    fprintf(stderr, "SHA mismatch lane %d\n  avx2:", i);
                    for (int j = 0; j < 32; j++) fprintf(stderr, "%02x", sa[i][j]);
                    fprintf(stderr, "\n  ni  :");
                    for (int j = 0; j < 32; j++) fprintf(stderr, "%02x", sn[i][j]);
                    fprintf(stderr, "\n");
                }
                shaFail++;
            }
            if (memcmp(ha[i], hn[i], 20) != 0) hFail++;
        }
    }
    printf("sha256: %llu checks, %llu fail; hash160: %llu fail\n",
           (unsigned long long)checks, (unsigned long long)shaFail,
           (unsigned long long)hFail);
    printf((shaFail == 0 && hFail == 0) ? "SHANI TEST PASS\n" : "SHANI TEST FAIL\n");
    return (shaFail == 0 && hFail == 0) ? 0 : 1;
}

static uint64_t g_sink = 0;
static double median(double* v, int n) {
    for (int i = 1; i < n; i++) { double x = v[i]; int j = i - 1;
        while (j >= 0 && v[j] > x) { v[j+1] = v[j]; j--; } v[j+1] = x; }
    return (n & 1) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

// throttle-robust: alternate AVX2/NI in short bursts over many rounds, median.
static void microbench() {
    using clk = std::chrono::steady_clock;
    std::mt19937_64 rng(1234567ULL);
    const int ROUNDS = 15;
    const uint64_t BURST = 200000;      // 8 * BURST hashes per measurement
    double av[ROUNDS], ni[ROUNDS];

    alignas(32) uint8_t in[8][64];
    for (int i = 0; i < 8; i++) mkblock(in[i], rng);
    alignas(32) uint8_t out[8][20];
    auto secs = [](clk::time_point a, clk::time_point b){
        return std::chrono::duration<double>(b - a).count(); };

    for (int r = 0; r < ROUNDS; r++) {
        auto a0 = clk::now();
        for (uint64_t it = 0; it < BURST; it++) {
            hash160_pubkey_8(in, out);
            in[0][1] ^= out[0][0];          // dependency so it isn't hoisted
        }
        auto a1 = clk::now();
        g_sink ^= out[0][0];
        for (uint64_t it = 0; it < BURST; it++) {
            hash160_pubkey_8_ni(in, out);
            in[0][1] ^= out[0][0];
        }
        auto a2 = clk::now();
        g_sink ^= out[0][0];
        av[r] = secs(a0, a1); ni[r] = secs(a1, a2);
    }
    double a = median(av, ROUNDS), n = median(ni, ROUNDS);
    double hashes = (double)BURST * 8.0;
    printf("\nhash160 microbench (median of %d rounds, alternated; %.1fM hashes/measure)\n",
           ROUNDS, hashes / 1e6);
    printf("  avx2 : %7.1f Mhash/s\n", hashes / a / 1e6);
    printf("  sha-ni: %7.1f Mhash/s   speedup %.3fx  %s\n",
           hashes / n / 1e6, a / n, (a / n > 1.0) ? "[NI faster]" : "[NI slower]");
    printf("  (sink %llx)\n", (unsigned long long)g_sink);
}

int main(int argc, char** argv) {
    uint64_t iters = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 800000;
    int rc = correctness(iters);
    microbench();
    return rc;
}
