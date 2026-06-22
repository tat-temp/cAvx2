#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <chrono>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <omp.h>
#include <array>
#include <utility>
#include <cstdint>
#include <climits>
#include <thread>

#include "p2pkh_decoder.h"
#include "sha256_avx2.h"
#include "ripemd160_avx2.h"
#include "SECP256K1.h"
#include "Point.h"
#include "Int.h"
#include "IntGroup.h"

#ifdef _MSC_VER
#define ALIGN32 __declspec(align(32))
#else
#define ALIGN32 __attribute__((aligned(32)))
#endif

static constexpr int    CPU_GROUP_SIZE = 4096;   // default group size (override with -g)
static constexpr int    HASH_BATCH_SIZE = 8;     // AVX2 lane count, fixed
static constexpr double STATUS_INTERVAL_SEC = 5.0;

// Sink for --skip-hash (EC-only) mode: keeps the generated points "live" so the
// optimizer cannot delete the point math we are trying to benchmark.
static volatile uint64_t g_sink = 0;

static inline std::string bytesToHex(const uint8_t* data, size_t len)
{
    static constexpr char lut[] = "0123456789abcdef";
    std::string out; out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        uint8_t b = data[i];
        out.push_back(lut[b >> 4]);
        out.push_back(lut[b & 0x0F]);
    }
    return out;
}

static void writeFoundKey(const std::string& privHex,
    const std::string& pubHex,
    const std::string& wif,
    const std::string& address)
{
    std::ofstream ofs("found_keys.txt", std::ios::app);
    if (!ofs) {
        std::cerr << "Cannot open found_keys.txt for writing\n";
        return;
    }
    ofs << privHex << ' ' << pubHex << ' ' << wif << ' ' << address << '\n';
}

std::vector<uint64_t> hexToBigNum(const std::string& hex)
{
    std::vector<uint64_t> bigNum;
    const size_t len = hex.size();
    bigNum.reserve((len + 15) / 16);
    for (size_t i = 0; i < len; i += 16) {
        size_t start = (len >= 16 + i) ? len - 16 - i : 0;
        size_t partLen = (len >= 16 + i) ? 16 : (len - i);
        uint64_t value = std::stoull(hex.substr(start, partLen), nullptr, 16);
        bigNum.push_back(value);
    }
    return bigNum;
}

std::string bigNumToHex(const std::vector<uint64_t>& num)
{
    std::ostringstream oss;
    oss << std::hex;
    for (auto it = num.rbegin(); it != num.rend(); ++it) {
        if (it != num.rbegin()) oss << std::setw(16) << std::setfill('0');
        oss << *it;
    }
    return oss.str();
}

std::vector<uint64_t> singleElementVector(uint64_t v) { return { v }; }

std::vector<uint64_t> bigNumAdd(const std::vector<uint64_t>& a,
    const std::vector<uint64_t>& b)
{
    std::vector<uint64_t> s;
    s.reserve(std::max(a.size(), b.size()) + 1);
    uint64_t carry = 0;
    for (size_t i = 0, sz = std::max(a.size(), b.size()); i < sz; ++i) {
        uint64_t x = (i < a.size()) ? a[i] : 0ULL;
        uint64_t y = (i < b.size()) ? b[i] : 0ULL;
        __uint128_t t = (__uint128_t)x + y + carry;
        carry = uint64_t(t >> 64);
        s.push_back(uint64_t(t));
    }
    if (carry) s.push_back(carry);
    return s;
}

std::vector<uint64_t> bigNumSubtract(const std::vector<uint64_t>& a,
    const std::vector<uint64_t>& b)
{
    std::vector<uint64_t> d = a;
    uint64_t borrow = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        uint64_t sub = b[i];
        if (d[i] < sub + borrow) {
            d[i] = d[i] + (~0ULL) - sub - borrow + 1ULL;
            borrow = 1ULL;
        }
        else {
            d[i] -= sub + borrow;
            borrow = 0ULL;
        }
    }
    for (size_t i = b.size(); borrow && i < d.size(); ++i) {
        if (d[i] == 0ULL) d[i] = ~0ULL;
        else { d[i] -= 1ULL; borrow = 0ULL; }
    }
    while (!d.empty() && d.back() == 0ULL) d.pop_back();
    return d;
}

std::pair<std::vector<uint64_t>, uint64_t> bigNumDivide(
    const std::vector<uint64_t>& a, uint64_t divisor)
{
    std::vector<uint64_t> q(a.size(), 0ULL);
    uint64_t r = 0ULL;
    for (int i = int(a.size()) - 1; i >= 0; --i) {
        __uint128_t t = ((__uint128_t)r << 64) | a[i];
        q[i] = uint64_t(t / divisor);
        r = uint64_t(t % divisor);
    }
    while (!q.empty() && q.back() == 0ULL) q.pop_back();
    return { q, r };
}

long double hexStrToLongDouble(const std::string& h)
{
    long double res = 0.0L;
    for (char c : h) {
        res *= 16.0L;
        if (c >= '0' && c <= '9') res += (c - '0');
        else if (c >= 'a' && c <= 'f') res += (c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') res += (c - 'A' + 10);
    }
    return res;
}

static inline std::string padHexTo64(const std::string& h)
{
    return (h.size() >= 64) ? h : std::string(64 - h.size(), '0') + h;
}

static inline Int hexToInt(const std::string& h)
{
    Int n; char buf[65] = { 0 };
    std::strncpy(buf, h.c_str(), 64);
    n.SetBase16(buf);
    return n;
}

static inline std::string intToHex(const Int& v)
{
    Int t; t.Set((Int*)&v); return t.GetBase16();
}

static inline bool isEven(const Int& n) { return n.IsEven(); }

static inline std::string intXToHex64(const Int& x)
{
    Int t; t.Set((Int*)&x);
    std::string h = t.GetBase16();
    if (h.size() < 64) h.insert(0, 64 - h.size(), '0');
    return h;
}

static inline std::string pointToCompressedHex(const Point& p)
{
    return (isEven(p.y) ? "02" : "03") + intXToHex64(p.x);
}

static void computeHash160BatchBinSingle3(
    Point* p,
    uint8_t outHash[][20])
{
    // Persistent per-thread input buffer. The SHA padding bytes [33..35] are
    // constant across calls, so write them once; each call refreshes only the
    // 33 pubkey bytes [0..32]. The fused hasher supplies all other padding and
    // keeps the SHA->RIPEMD digest in registers (no intermediate buffer).
    alignas(32) static thread_local uint8_t shaIn[HASH_BATCH_SIZE][64];
    static thread_local bool init = false;
    if (!init) {
        for (int i = 0; i < HASH_BATCH_SIZE; i++) {
            shaIn[i][33] = 0x80;
            shaIn[i][34] = 0x00;
            shaIn[i][35] = 0x00;
        }
        init = true;
    }

    for (int i = 0; i < HASH_BATCH_SIZE; i++)
    {
        // Branchless 02/03 prefix from y's low bit; Get32Bytes writes the
        // big-endian x directly (4x bswap64) instead of a 32-byte reverse_copy.
        shaIn[i][0] = 0x02 | (uint8_t)(p[i].y.bits64[0] & 1ULL);
        p[i].x.Get32Bytes(&shaIn[i][1]);
    }

    hash160_pubkey_8(shaIn, outHash);
}

static void printUsage(const char* prog)
{
    std::cerr
        << "Usage: " << prog << " -a <Base58_P2PKH> -r <START:END> [options]\n"
        << "  -a <addr>       target P2PKH address (optional in benchmark mode)\n"
        << "  -r <START:END>  hex private-key range (optional in benchmark mode)\n"
        << "  -t <threads>    worker threads (default: all logical CPUs)\n"
        << "  -g <size>       CPU group size, multiple of " << HASH_BATCH_SIZE
        << " (default " << CPU_GROUP_SIZE << "); cache/throughput tuning\n"
        << "  -b <seconds>    benchmark: run for <seconds> then report Mkeys/s\n"
        << "                  (no real target needed) and exit\n"
        << "  --skip-hash     skip SHA-256/RIPEMD-160 (EC point-gen only) to\n"
        << "                  profile the EC vs hashing split\n"
        << "  --ilp <0-8>     EC point-loop interleave width (0 = original loop,\n"
        << "                  N>0 = interleave 2xN independent point chains)\n";
}

static std::string formatElapsedTime(double sec)
{
    int h = int(sec) / 3600, m = (int(sec) % 3600) / 60, s = int(sec) % 60;
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << h << ":"
        << std::setw(2) << m << ":"
        << std::setw(2) << s;
    return oss.str();
}

static void printStats(int nCPU,
    const std::string& addr,
    const std::string& hashHex,
    const std::string& range,
    double mks,
    unsigned long long checked,
    double elapsed,
    long double prog)
{
    const int lines = 9;
    static bool first = true;
    if (!first) std::cout << "\033[" << lines << "A";
    else       first = false;

    std::cout << "================= WORK IN PROGRESS =================\n"
        << "Target Address: " << addr << "\n"
        << "Hash160       : " << hashHex << "\n"
        << "CPU Threads   : " << nCPU << "\n"
        << "Mkeys/s       : " << std::fixed << std::setprecision(2) << mks << "\n"
        << "Total Checked : " << checked << "\n"
        << "Elapsed Time  : " << formatElapsedTime(elapsed) << "\n"
        << "Range         : " << range << "\n"
        << "Progress      : " << std::fixed << std::setprecision(4) << prog << " %\n";
    std::cout.flush();
}

struct ThreadRange { std::string startHex, endHex; };
static std::vector<ThreadRange> g_threadRanges;

int main(int argc, char* argv[])
{
    bool aOK = false, rOK = false, tOK = false;

    int    userThreads  = 0;
    int    groupSize    = CPU_GROUP_SIZE;
    bool   benchmark    = false;
    double benchSeconds = 10.0;
    bool   skipHash     = false;
    int    ilpWidth     = 0;   // 0 = original point loop; >0 = interleave width

    std::string targetAddress, rangeStr;
    std::vector<uint8_t> targetHash160;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-a") && i + 1 < argc) {
            targetAddress = argv[++i]; aOK = true;
            targetHash160 = P2PKHDecoder::getHash160(targetAddress);
        }
        else if (!std::strcmp(argv[i], "-r") && i + 1 < argc) {
            rangeStr = argv[++i]; rOK = true;
        }
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) {
            userThreads = std::stoi(argv[++i]); tOK = true;
            if (userThreads < 1) {
                std::cerr << "-t must be >0\n"; return 1;
            }
        }
        else if (!std::strcmp(argv[i], "-g") && i + 1 < argc) {
            groupSize = std::stoi(argv[++i]);
            if (groupSize < HASH_BATCH_SIZE || groupSize % HASH_BATCH_SIZE != 0) {
                std::cerr << "-g must be a positive multiple of "
                          << HASH_BATCH_SIZE << "\n";
                return 1;
            }
        }
        else if ((!std::strcmp(argv[i], "-b") || !std::strcmp(argv[i], "--benchmark"))
                 && i + 1 < argc) {
            benchSeconds = std::stod(argv[++i]); benchmark = true;
            if (benchSeconds <= 0.0) { std::cerr << "-b must be >0\n"; return 1; }
        }
        else if (!std::strcmp(argv[i], "--skip-hash")) {
            skipHash = true;
        }
        else if (!std::strcmp(argv[i], "--ilp") && i + 1 < argc) {
            ilpWidth = std::stoi(argv[++i]);
            if (ilpWidth < 0 || ilpWidth > 8) {
                std::cerr << "--ilp must be 0..8 (0 = original loop)\n"; return 1;
            }
        }
        else {
            printUsage(argv[0]); return 1;
        }
    }

    // Benchmark mode: a real target/range are optional. Default to a wide range
    // and an all-zero (never-match) target so we time the pipeline, not a find.
    if (benchmark) {
        if (!rOK) { rangeStr = "8000000000000000:FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"; rOK = true; }
        if (!aOK) { targetHash160.assign(20, 0x00); targetAddress = "(benchmark)"; aOK = true; }
    }
    if (!aOK || !rOK) { printUsage(argv[0]); return 1; }

    int hwThreads = omp_get_num_procs();
    int numCPUs = tOK ? std::min(userThreads, hwThreads) : hwThreads;

    std::string targetHashHex = bytesToHex(targetHash160.data(),
        targetHash160.size());

    size_t colon = rangeStr.find(':');
    if (colon == std::string::npos) { std::cerr << "Bad range\n"; return 1; }
    std::string startHex = rangeStr.substr(0, colon);
    std::string endHex = rangeStr.substr(colon + 1);

    auto startBN = hexToBigNum(startHex), endBN = hexToBigNum(endHex);

    bool okRange = false;
    if (startBN.size() < endBN.size()) okRange = true;
    else if (startBN.size() == endBN.size()) {
        okRange = true;
        for (int i = int(startBN.size()) - 1; i >= 0; --i) {
            if (startBN[i] < endBN[i]) break;
            if (startBN[i] > endBN[i]) { okRange = false; break; }
        }
    }
    if (!okRange) { std::cerr << "Range start > end\n"; return 1; }

    auto rangeSize = bigNumAdd(bigNumSubtract(endBN, startBN),
        singleElementVector(1ULL));
    long double totalRangeLD = hexStrToLongDouble(bigNumToHex(rangeSize));

    auto [chunk, remainder] = bigNumDivide(rangeSize, (uint64_t)numCPUs);
    g_threadRanges.resize(numCPUs);
    std::vector<uint64_t> cur = startBN;
    for (int t = 0; t < numCPUs; ++t) {
        auto e = bigNumAdd(cur, chunk);
        if (t < remainder) e = bigNumAdd(e, singleElementVector(1ULL));
        e = bigNumSubtract(e, singleElementVector(1ULL));
        g_threadRanges[t].startHex = bigNumToHex(cur);
        g_threadRanges[t].endHex = bigNumToHex(e);
        cur = bigNumAdd(e, singleElementVector(1ULL));
    }
    std::string displayRange = g_threadRanges.front().startHex + ":" +
        g_threadRanges.back().endHex;

    unsigned long long globalChecked = 0ULL;
    double             globalElapsed = 0.0, mkeys = 0.0;
    auto tStart = std::chrono::high_resolution_clock::now();
    auto lastStat = tStart;
    const int hLength = (groupSize / 2 - 1);
    bool          matchFound = false;
    volatile bool benchStop  = false;
    std::string foundPriv, foundPub, foundWIF;

    Secp256K1 secp;
    secp.Init();

#pragma omp parallel num_threads(numCPUs) \
    shared(globalChecked,globalElapsed,mkeys,matchFound,benchStop, \
           foundPriv,foundPub,foundWIF, \
           tStart,lastStat)
    {
        int tid = omp_get_thread_num();

        Int priv = hexToInt(g_threadRanges[tid].startHex);
        Int privEnd = hexToInt(g_threadRanges[tid].endHex);       

        Int halfGroupSize; halfGroupSize.SetInt32(groupSize / 2);
        Int privStartMiddleGroup = priv; privStartMiddleGroup.Add(&halfGroupSize);

        Point startP = secp.ComputePublicKey(&privStartMiddleGroup);

        std::vector<Int> dx(groupSize / 2 + 1);
        IntGroup grp(groupSize / 2 + 1);
        std::vector<Point> pts(groupSize);

        grp.Set(dx.data());

        Int dy;
        Int dyn;
        Int _s;
        Int _p;
        Point pp;
        Point pn;

        // Interleave scratch for the --ilp point loop: MAXW independent (+G) and
        // (-G) chains processed in lockstep to hide field-multiply latency.
        static const int MAXW = 8;
        Int dyf[MAXW], sf[MAXW], pf[MAXW];
        Int dyb[MAXW], sb[MAXW], pb[MAXW];
        Point ppA[MAXW], pnA[MAXW];

        std::vector<Point> Gn(groupSize / 2);
        Point _2Gn;

        // Compute Generator table G[n] = (n+1)*G
        Point g = secp.G;
        Gn[0] = g;
        g = secp.DoubleDirect(g);
        Gn[1] = g;
        for (int i = 2; i < groupSize / 2; i++) {
            g = secp.AddDirect(g, secp.G);
            Gn[i] = g;
        }
        // _2Gn = groupSize*G
        _2Gn = secp.DoubleDirect(Gn[groupSize / 2 - 1]);

        ALIGN32 uint8_t hashRes[HASH_BATCH_SIZE][20];

        __m128i target16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(targetHash160.data()));

        while (!matchFound && !benchStop) {
            /*{
                std::cout << intToHex(priv) << ":" << intToHex(privEnd) << "\n";
            }*/
            if (priv.IsGreater((Int*)&privEnd)) break;

            // Clamp the final block so we never test (or count) keys past privEnd.
            uint64_t validCount = (uint64_t)groupSize;
            {
                Int blockLast = priv;
                blockLast.Add((uint64_t)(groupSize - 1));
                if (blockLast.IsGreater((Int*)&privEnd)) {
                    Int rem = privEnd;
                    rem.Sub(&priv);                       // 0 <= rem < groupSize
                    validCount = (uint64_t)rem.GetInt32() + 1ULL;
                }
            }

            int j;

            for (j = 0; j < hLength; j++) {
                dx[j].ModSub(&Gn[j].x, &startP.x);
            }
            dx[j].ModSub(&Gn[j].x, &startP.x);  // For the first point
            dx[j + 1].ModSub(&_2Gn.x, &startP.x); // For the next center point

            grp.ModInv();

            pts[groupSize / 2] = startP;

          if (ilpWidth <= 0) {
            for (j = 0; j < hLength; j++) {
                pp = startP;
                pn = startP;

                // P = startP + i*G
                dy.ModSub(&Gn[j].y, &pp.y);

                _s.ModMulK1(&dy, &dx[j]);       // s = (p2.y-p1.y)*inverse(p2.x-p1.x);
                _p.ModSquareK1(&_s);            // _p = pow2(s)

                pp.x.ModNeg();
                pp.x.ModAdd(&_p);
                pp.x.ModSub(&Gn[j].x);           // rx = pow2(s) - p1.x - p2.x;

                pp.y.ModSub(&Gn[j].x, &pp.x);
                pp.y.ModMulK1(&_s);
                pp.y.ModSub(&Gn[j].y);           // ry = - p2.y - s*(ret.x-p2.x);

                // P = startP - i*G  , if (x,y) = i*G then (x,-y) = -i*G
                dyn.Set(&Gn[j].y);
                dyn.ModNeg();
                dyn.ModSub(&pn.y);

                _s.ModMulK1(&dyn, &dx[j]);      // s = (p2.y-p1.y)*inverse(p2.x-p1.x);
                _p.ModSquareK1(&_s);            // _p = pow2(s)

                pn.x.ModNeg();
                pn.x.ModAdd(&_p);
                pn.x.ModSub(&Gn[j].x);          // rx = pow2(s) - p1.x - p2.x;

                pn.y.ModSub(&Gn[j].x, &pn.x);
                pn.y.ModMulK1(&_s);
                pn.y.ModAdd(&Gn[j].y);          // ry = - p2.y - s*(ret.x-p2.x);

                pts[groupSize / 2 + (j + 1)] = pp;
                pts[groupSize / 2 - (j + 1)] = pn;
            }
          } else {
            // --- ILP: process 2*W independent chains in lockstep so the
            //     dependent field-multiply latencies overlap. Each step is the
            //     same op for all W forward (+G) and W backward (-G) points.
            int W = ilpWidth; if (W > MAXW) W = MAXW;
            for (int jb = 0; jb < hLength; jb += W) {
                int w = (hLength - jb < W) ? (hLength - jb) : W;

                for (int k = 0; k < w; k++) { int jx = jb + k;
                    dyf[k].ModSub(&Gn[jx].y, &startP.y);                       // +G: dy
                    dyb[k].Set(&Gn[jx].y); dyb[k].ModNeg(); dyb[k].ModSub(&startP.y); // -G: dy
                }
                for (int k = 0; k < w; k++) { int jx = jb + k;
                    sf[k].ModMulK1(&dyf[k], &dx[jx]);                          // s = dy * dxinv
                    sb[k].ModMulK1(&dyb[k], &dx[jx]);
                }
                for (int k = 0; k < w; k++) {
                    pf[k].ModSquareK1(&sf[k]);                                 // s^2
                    pb[k].ModSquareK1(&sb[k]);
                }
                for (int k = 0; k < w; k++) { int jx = jb + k;
                    ppA[k].x.Set(&startP.x); ppA[k].x.ModNeg();
                    ppA[k].x.ModAdd(&pf[k]); ppA[k].x.ModSub(&Gn[jx].x);       // rx = s^2 - x1 - x2
                    pnA[k].x.Set(&startP.x); pnA[k].x.ModNeg();
                    pnA[k].x.ModAdd(&pb[k]); pnA[k].x.ModSub(&Gn[jx].x);
                }
                for (int k = 0; k < w; k++) { int jx = jb + k;
                    ppA[k].y.ModSub(&Gn[jx].x, &ppA[k].x);                     // x2 - rx
                    pnA[k].y.ModSub(&Gn[jx].x, &pnA[k].x);
                }
                for (int k = 0; k < w; k++) {
                    ppA[k].y.ModMulK1(&sf[k]);                                 // s*(x2 - rx)
                    pnA[k].y.ModMulK1(&sb[k]);
                }
                for (int k = 0; k < w; k++) { int jx = jb + k;
                    ppA[k].y.ModSub(&Gn[jx].y);                                // ry = s*(x2-rx) - y2
                    pnA[k].y.ModAdd(&Gn[jx].y);                                // ry = s*(x2-rx) + y2
                    pts[groupSize / 2 + (jx + 1)] = ppA[k];
                    pts[groupSize / 2 - (jx + 1)] = pnA[k];
                }
            }
            j = hLength;
          }

            // First point (startP - (GRP_SZIE/2)*G)
            pn = startP;
            dyn.Set(&Gn[j].y);
            dyn.ModNeg();
            dyn.ModSub(&pn.y);

            _s.ModMulK1(&dyn, &dx[j]);
            _p.ModSquareK1(&_s);

            pn.x.ModNeg();
            pn.x.ModAdd(&_p);
            pn.x.ModSub(&Gn[j].x);

            pn.y.ModSub(&Gn[j].x, &pn.x);
            pn.y.ModMulK1(&_s);
            pn.y.ModAdd(&Gn[j].y);

            pts[0] = pn;

            // Next start point (startP + GRP_SIZE*G)
            pp = startP;
            dy.ModSub(&_2Gn.y, &pp.y);

            _s.ModMulK1(&dy, &dx[j + 1]);
            _p.ModSquareK1(&_s);

            pp.x.ModNeg();
            pp.x.ModAdd(&_p);
            pp.x.ModSub(&_2Gn.x);

            pp.y.ModSub(&_2Gn.x, &pp.x);
            pp.y.ModMulK1(&_s);
            pp.y.ModSub(&_2Gn.y);
            startP = pp;

            //{
            //    std::cout << "Boom!" << "\n";

            //    for (int i = 0; i < CPU_GROUP_SIZE; ++i) {
            //        Int mPriv = priv;
            //        //int idx = i;
            //        //if (idx < 256) { Int off; off.SetInt32(idx); mPriv.Add(&off); }
            //        //else { Int off; off.SetInt32(idx - 256); mPriv.Sub(&off); }

            //        Int off; off.SetInt32(i); mPriv.Add(&off);

            //        foundPriv = padHexTo64(intToHex(mPriv));
            //        foundPub = pointToCompressedHex(pts[i]);

            //        //std::cout << padHexTo64(intToHex(priv)) << " " << foundPriv << " " << foundPub << "\n";
            //        std::cout << foundPriv << " " << foundPub << "\n";
            //    }
            //}

          if (!skipHash) {
            for (int i = 0; i < groupSize; i += HASH_BATCH_SIZE) {
                computeHash160BatchBinSingle3(
                    pts.data() + i,
                    hashRes);

                // Results check
                for (int j = 0; j < HASH_BATCH_SIZE; j++) {
                    if ((uint64_t)(i + j) >= validCount) break;
                    /*{
                        Int i;
                        i.Set32Bytes(hashRes[j]);
                        std::cout << "0 " << padHexTo64(intToHex(i)) << "\n";
                    }*/

                    __m128i cand16 = _mm_load_si128(reinterpret_cast<const __m128i*>(hashRes[j]));
                    __m128i cmp = _mm_cmpeq_epi8(cand16, target16);
                    if (_mm_movemask_epi8(cmp) == 0xFFFF) {
                        // Checking last 4 bytes (20 - 16)
                        if (std::memcmp(hashRes[j], targetHash160.data(), 20) == 0) {
#pragma omp critical(full_match)
                            {
                                if (!matchFound) {
                                    matchFound = true;
                                    Int mPriv = priv;
                                    int idx = i + j;

                                    Int off; off.SetInt32(idx);
                                    mPriv.Add(&off);

                                    foundPriv = padHexTo64(intToHex(mPriv));
                                    foundPub = pointToCompressedHex(pts[idx]);
                                    foundWIF = P2PKHDecoder::compute_wif(foundPriv, true);
                                }
                            }
#pragma omp cancel parallel
                        }
                    }
                }
            }
          } else {
            // EC-only profiling: touch every point so the optimizer keeps the
            // point math we are benchmarking.
            uint64_t acc = 0;
            for (int i = 0; i < groupSize; i++) acc ^= pts[i].x.bits64[0];
            g_sink ^= acc;
          }

            {
                priv.Add((uint64_t)groupSize);
            }

#pragma omp atomic
            globalChecked += validCount;

            if (tid == 0)
            {
                auto now = std::chrono::high_resolution_clock::now();

                if (benchmark &&
                    std::chrono::duration<double>(now - tStart).count() >= benchSeconds)
                    benchStop = true;

                if (std::chrono::duration<double>(now - lastStat).count()
                    >= STATUS_INTERVAL_SEC)
                {
//#pragma omp critical
                    {
                        auto now = std::chrono::high_resolution_clock::now();

                        globalElapsed =
                            std::chrono::duration<double>(now - tStart).count();

                        mkeys = globalChecked / globalElapsed / 1e6;
                        long double prog = totalRangeLD > 0.0L
                            ? (globalChecked / totalRangeLD * 100.0L)
                            : 0.0L;

                        printStats(numCPUs, targetAddress, targetHashHex, displayRange,
                            mkeys, globalChecked, globalElapsed, prog);

                        lastStat = now;
                    }
                }
            }
        }
    }

    if (benchmark) {
        double el = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - tStart).count();
        double rate = el > 0.0 ? globalChecked / el / 1e6 : 0.0;
        std::cout << "\n========== BENCHMARK RESULT ==========\n"
            << "Threads     : " << numCPUs << "\n"
            << "Group size  : " << groupSize << "\n"
            << "Hashing     : " << (skipHash ? "OFF (EC point-gen only)" : "ON") << "\n"
            << "ILP width   : " << ilpWidth << (ilpWidth ? "" : " (original loop)") << "\n"
            << "Keys done   : " << globalChecked << "\n"
            << "Elapsed     : " << std::fixed << std::setprecision(2) << el << " s\n"
            << "Throughput  : " << std::fixed << std::setprecision(2) << rate << " Mkeys/s\n";
        return 0;
    }

    if (!matchFound) {
        std::cout << "\nNo match found.\n";
        return 0;
    }
    writeFoundKey(foundPriv, foundPub, foundWIF, targetAddress);

    std::cout << "================== FOUND MATCH! ==================\n"
        << "Private Key   : " << foundPriv << "\n"
        << "Public Key    : " << foundPub << "\n"
        << "WIF           : " << foundWIF << "\n"
        << "P2PKH Address : " << targetAddress << "\n";
    return 0;
}