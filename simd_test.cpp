// ---------------------------------------------------------------------------
// Tier 3 standalone gate: bit-exactness + microbenchmark for the 4-way AVX2
// field multiply (simd_field.h) against the scalar secp256k1 field (ModMulK1).
//
//   ./simdtest            # correctness (must pass) + microbench
//   ./simdtest 2000000    # correctness iters
//
// The microbench is the GO/NO-GO: it times 16 scalar ModMulK1 vs 4 fe4_mul
// (= 16 lane-multiplies) per iteration, both with 16-wide ILP, and prints the
// speedup.  If 4-way SIMD does not beat 4x scalar here, Tier 3 stops before any
// hot-loop surgery.
// ---------------------------------------------------------------------------
#include "Int.h"
#include "SECP256K1.h"
#include "simd_field.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <chrono>

static Int g_P;   // field characteristic, for canonicalization

// Reduce x into [0, P) for congruence comparison (both sides are < 2^256).
static void canon(Int& x) { while (x.IsGreaterOrEqual(&g_P)) x.Sub(&g_P); }

static void setIntFrom4(Int& x, const uint64_t w[4]) {
    x.bits64[0] = w[0]; x.bits64[1] = w[1];
    x.bits64[2] = w[2]; x.bits64[3] = w[3];
    x.bits64[4] = 0;
}

// ---- correctness: four random products vs one fe4_mul, lane by lane --------
static int correctness(uint64_t iters) {
    std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
    Int* P = Int::GetFieldCharacteristic();
    g_P.Set(P);

    auto randField = [&](Int& x) {
        x.bits64[0] = rng(); x.bits64[1] = rng();
        x.bits64[2] = rng(); x.bits64[3] = rng();
        x.bits64[4] = 0; x.Mod(P);
    };

    // Edge elements: 0, 1, 2, P-1, P-2.
    Int sp[5]; sp[0].SetInt32(0); sp[1].SetInt32(1); sp[2].SetInt32(2);
    sp[3].Set(P); sp[3].SubOne(); sp[4].Set(P); sp[4].Sub(&sp[2]);

    uint64_t checks = 0, fails = 0;
    Int a[4], b[4], rs[4];
    for (uint64_t k = 0; k < iters; k += 4) {
        for (int l = 0; l < 4; l++) {
            uint64_t kk = k + l;
            if (kk < 25) { a[l].Set(&sp[(kk / 5) % 5]); b[l].Set(&sp[kk % 5]); }
            else if ((kk & 7) == 0) { a[l].Set(&sp[kk % 5]); randField(b[l]); }
            else if ((kk & 7) == 1) { randField(a[l]); b[l].Set(&sp[kk % 5]); }
            else { randField(a[l]); randField(b[l]); }
            rs[l].ModMulK1(&a[l], &b[l]); canon(rs[l]);     // scalar oracle
        }
        fe4 fa = fe4_pack(a[0].bits64, a[1].bits64, a[2].bits64, a[3].bits64);
        fe4 fb = fe4_pack(b[0].bits64, b[1].bits64, b[2].bits64, b[3].bits64);
        fe4 fr; fe4_mul(fr, fa, fb);

        for (int l = 0; l < 4; l++) {
            uint64_t w[4]; fe4_unpack_lane(fr, l, w);
            Int got; setIntFrom4(got, w); canon(got);
            checks++;
            if (!got.IsEqual(&rs[l])) {
                if (fails < 6)
                    fprintf(stderr,
                        "MISMATCH lane %d\n  a  =%s\n  b  =%s\n  simd=%s\n  ref =%s\n",
                        l, a[l].GetBase16().c_str(), b[l].GetBase16().c_str(),
                        got.GetBase16().c_str(), rs[l].GetBase16().c_str());
                fails++;
            }
        }

        // also exercise squaring
        fe4 fsq; fe4_sqr(fsq, fa);
        for (int l = 0; l < 4; l++) {
            uint64_t w[4]; fe4_unpack_lane(fsq, l, w);
            Int got; setIntFrom4(got, w); canon(got);
            Int ref; ref.ModSquareK1(&a[l]); canon(ref);
            checks++;
            if (!got.IsEqual(&ref)) {
                if (fails < 6)
                    fprintf(stderr, "SQR MISMATCH lane %d\n  a  =%s\n  simd=%s\n  ref =%s\n",
                        l, a[l].GetBase16().c_str(), got.GetBase16().c_str(),
                        ref.GetBase16().c_str());
                fails++;
            }
        }
    }
    printf("simd field: %llu checks, %llu fail\n",
           (unsigned long long)checks, (unsigned long long)fails);
    printf(fails == 0 ? "SIMDTEST PASS\n" : "SIMDTEST FAIL\n");
    return fails == 0 ? 0 : 1;
}

// ---- microbench: 16 scalar muls vs 4 fe4 muls per iter, 16-wide ILP --------
static void microbench() {
    using clk = std::chrono::steady_clock;
    std::mt19937_64 rng(12345);
    Int* P = Int::GetFieldCharacteristic();
    auto rf = [&](Int& x){ x.bits64[0]=rng();x.bits64[1]=rng();x.bits64[2]=rng();
                           x.bits64[3]=rng();x.bits64[4]=0;x.Mod(P); };

    const int K = 16;                 // 16-wide ILP, matches --ilp 16
    const uint64_t ITERS = 4000000;   // 16 * ITERS = 64M field multiplies

    Int sa[K], sb[K];
    for (int i = 0; i < K; i++) { rf(sa[i]); rf(sb[i]); }
    fe4 fa[4], fb[4];
    for (int g = 0; g < 4; g++) {
        fa[g] = fe4_pack(sa[4*g+0].bits64, sa[4*g+1].bits64, sa[4*g+2].bits64, sa[4*g+3].bits64);
        fb[g] = fe4_pack(sb[4*g+0].bits64, sb[4*g+1].bits64, sb[4*g+2].bits64, sb[4*g+3].bits64);
    }

    // scalar: sa[i] *= sb[i]
    auto t0 = clk::now();
    for (uint64_t it = 0; it < ITERS; it++)
        for (int i = 0; i < K; i++) sa[i].ModMulK1(&sb[i]);
    auto t1 = clk::now();
    uint64_t ssink = 0; for (int i = 0; i < K; i++) ssink ^= sa[i].bits64[0];

    // simd: fa[g] = fa[g] * fb[g]
    auto t2 = clk::now();
    for (uint64_t it = 0; it < ITERS; it++)
        for (int g = 0; g < 4; g++) fe4_mul(fa[g], fa[g], fb[g]);
    auto t3 = clk::now();
    uint64_t vsink = 0;
    for (int g = 0; g < 4; g++) { uint64_t w[4]; fe4_unpack_lane(fa[g], 0, w); vsink ^= w[0]; }

    double sdt = std::chrono::duration<double>(t1 - t0).count();
    double vdt = std::chrono::duration<double>(t3 - t2).count();
    double muls = (double)ITERS * 16.0;
    printf("\nmicrobench (%.0fM field-muls each, K=%d ILP)\n", muls / 1e6, K);
    printf("  scalar ModMulK1 : %7.3f s   %8.1f Mmul/s\n", sdt, muls / sdt / 1e6);
    printf("  simd   fe4_mul  : %7.3f s   %8.1f Mmul/s\n", vdt, muls / vdt / 1e6);
    printf("  speedup (scalar_t / simd_t) : %.3fx   %s\n",
           sdt / vdt, (sdt / vdt > 1.0) ? "[SIMD faster]" : "[SIMD slower]");
    printf("  (sinks %llx %llx)\n", (unsigned long long)ssink, (unsigned long long)vsink);
}

int main(int argc, char** argv) {
    Secp256K1 secp; secp.Init();                  // sets up the field (Int::P)
    uint64_t iters = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 2000000;
    int rc = correctness(iters);
    microbench();
    return rc;
}
