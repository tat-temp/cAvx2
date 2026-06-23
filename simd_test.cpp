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

// ---- microbench: throttle-robust. Alternates scalar/simd in short bursts over
// many rounds and reports the MEDIAN ratio, so a thermal ramp hits both equally
// (the absolute Mop/s still drift on a throttling laptop; the ratio is stable).
static uint64_t g_sink = 0;
static double median(double* v, int n) {
    for (int i = 1; i < n; i++) { double x = v[i]; int j = i - 1;
        while (j >= 0 && v[j] > x) { v[j+1] = v[j]; j--; } v[j+1] = x; }
    return (n & 1) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}
static void microbench() {
    using clk = std::chrono::steady_clock;
    std::mt19937_64 rng(12345);
    Int* P = Int::GetFieldCharacteristic();
    auto rf = [&](Int& x){ x.bits64[0]=rng();x.bits64[1]=rng();x.bits64[2]=rng();
                           x.bits64[3]=rng();x.bits64[4]=0;x.Mod(P); };

    const int K = 16;                 // 16-wide ILP, matches --ilp 16
    const int ROUNDS = 15;
    const uint64_t BURST = 250000;    // 16*BURST = 4M ops per measurement
    double smul[ROUNDS], vmul[ROUNDS], ssq[ROUNDS], vsq[ROUNDS];

    Int sa[K], sb[K]; fe4 fa[4], fb[4];
    auto reinit = [&]() {             // fresh, equal starting state each measurement
        for (int i = 0; i < K; i++) { rf(sa[i]); rf(sb[i]); }
        for (int g = 0; g < 4; g++) {
            fa[g] = fe4_pack(sa[4*g].bits64, sa[4*g+1].bits64, sa[4*g+2].bits64, sa[4*g+3].bits64);
            fb[g] = fe4_pack(sb[4*g].bits64, sb[4*g+1].bits64, sb[4*g+2].bits64, sb[4*g+3].bits64);
        }
    };
    auto secs = [](clk::time_point a, clk::time_point b){
        return std::chrono::duration<double>(b - a).count(); };

    for (int r = 0; r < ROUNDS; r++) {
        reinit();
        auto a0 = clk::now();
        for (uint64_t it = 0; it < BURST; it++) for (int i = 0; i < K; i++) sa[i].ModMulK1(&sb[i]);
        auto a1 = clk::now();
        for (int i = 0; i < K; i++) g_sink ^= sa[i].bits64[0];
        for (uint64_t it = 0; it < BURST; it++) for (int g = 0; g < 4; g++) fe4_mul(fa[g], fa[g], fb[g]);
        auto a2 = clk::now();
        for (int g = 0; g < 4; g++) { uint64_t w[4]; fe4_unpack_lane(fa[g],0,w); g_sink ^= w[0]; }
        smul[r] = secs(a0, a1); vmul[r] = secs(a1, a2);

        reinit();
        auto b0 = clk::now();
        for (uint64_t it = 0; it < BURST; it++) for (int i = 0; i < K; i++) sa[i].ModSquareK1(&sa[i]);
        auto b1 = clk::now();
        for (int i = 0; i < K; i++) g_sink ^= sa[i].bits64[0];
        for (uint64_t it = 0; it < BURST; it++) for (int g = 0; g < 4; g++) fe4_sqr(fa[g], fa[g]);
        auto b2 = clk::now();
        for (int g = 0; g < 4; g++) { uint64_t w[4]; fe4_unpack_lane(fa[g],0,w); g_sink ^= w[0]; }
        ssq[r] = secs(b0, b1); vsq[r] = secs(b1, b2);
    }

    double sm = median(smul,ROUNDS), vm = median(vmul,ROUNDS);
    double sq = median(ssq,ROUNDS),  vq = median(vsq,ROUNDS);
    double ops = (double)BURST * 16.0;
    printf("\nmicrobench (median of %d rounds, alternated; %.0fM ops/measure, K=%d ILP)\n",
           ROUNDS, ops/1e6, K);
    printf("  mul : scalar %7.1f Mop/s | simd %7.1f Mop/s | speedup %.3fx  %s\n",
           ops/sm/1e6, ops/vm/1e6, sm/vm, (sm/vm > 1.0) ? "[SIMD faster]" : "[SIMD slower]");
    printf("  sqr : scalar %7.1f Mop/s | simd %7.1f Mop/s | speedup %.3fx  %s\n",
           ops/sq/1e6, ops/vq/1e6, sq/vq, (sq/vq > 1.0) ? "[SIMD faster]" : "[SIMD slower]");
    printf("  2mul+1sqr (point-loop mix)  : speedup %.3fx  %s\n",
           (2*sm+sq)/(2*vm+vq), ((2*sm+sq)/(2*vm+vq) > 1.0) ? "[SIMD faster]" : "[SIMD slower]");
    printf("  (sink %llx)\n", (unsigned long long)g_sink);
}

int main(int argc, char** argv) {
    Secp256K1 secp; secp.Init();                  // sets up the field (Int::P)
    uint64_t iters = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 2000000;
    int rc = correctness(iters);
    microbench();
    return rc;
}
