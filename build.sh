#!/usr/bin/env bash
# Cyclone build script.  (LF-only; enforced by .gitattributes.)
#
#   ./build.sh [baseline|native] [output-name]
#
# Locks -O3: -Ofast miscompiles the find path (access violation on a match)
# for zero throughput gain, since the hot path is integer + SIMD. The ONLY
# difference between the two targets is the arch flag set, so the Phase 1
# `-march=native` A/B is a single-word change:
#
#   ./build.sh baseline   # -mavx2 -mbmi2 -madx           -> ./Cyclone
#   ./build.sh native     # -march=native (no AVX-512)    -> ./Cyclone_native
#
set -euo pipefail

ARCH="${1:-baseline}"
OUT="${2:-}"

case "$ARCH" in
  baseline)
    ARCH_FLAGS="-mavx2 -mbmi2 -madx"
    DEF_OUT="Cyclone" ;;
  native)
    # -march=native implies -mavx2/-mbmi2/-madx when present and tunes for the
    # build host. Explicitly disable AVX-512 to honor the no-AVX-512 constraint
    # on CPUs that have it (no effect on CPUs that don't).
    ARCH_FLAGS="-march=native -mno-avx512f -mno-avx512vl -mno-avx512dq -mno-avx512bw -mno-avx512cd"
    DEF_OUT="Cyclone_native" ;;
  adx)
    # Baseline ISA + the hand-written dual-carry-chain (mulx/adcx/adox) field
    # multiply (Phase 2). A/B this against `baseline`.
    ARCH_FLAGS="-mavx2 -mbmi2 -madx -DFIELD_ADX"
    DEF_OUT="Cyclone_adx" ;;
  fused)
    # Baseline ISA + fused hash-in-point-loop (Tier 2.3): generate points in
    # FUSE_BLOCK-sized sub-blocks and hash each block while it is still L1-hot,
    # instead of buffering the whole group and reading the ~480 KB pts[] back.
    # Sweep the block size with e.g. `FUSE_BLOCK=512 ./build.sh fused Cyclone_f512`.
    # A/B this against `baseline`.
    FB="${FUSE_BLOCK:-256}"
    ARCH_FLAGS="-mavx2 -mbmi2 -madx -DFUSED_HASH -DFUSE_BLOCK=${FB}"
    DEF_OUT="Cyclone_fused" ;;
  simdtest)
    # Tier 3 gate: standalone bit-exactness + microbench for the 4-way AVX2
    # field multiply (simd_field.h) vs the scalar ModMulK1. Not the searcher --
    # builds ./simdtest. Run it (`./simdtest`) and read the speedup line: it is
    # the GO/NO-GO for vectorizing the point loop.
    ARCH_FLAGS="-mavx2 -mbmi2 -madx"
    DEF_OUT="simdtest"
    SRC_OVERRIDE="simd_test.cpp Int.cpp IntMod.cpp Point.cpp SECP256K1.cpp \
Random.cpp IntGroup.cpp" ;;
  simdfield)
    # Tier 3 (Milestone C): 4-way AVX2 SIMD field arithmetic (simd_field.h)
    # wired into the --ilp point-reconstruction loop. A/B against `baseline`.
    ARCH_FLAGS="-mavx2 -mbmi2 -madx -DSIMD_FIELD"
    DEF_OUT="Cyclone_simd" ;;
  shani)
    # Hardware SHA-256 (SHA extensions) for the HASH160 step instead of 8-way
    # AVX2 SHA -- runs on the dedicated SHA unit, off the AVX2 vector ports the
    # EC + RIPEMD contend for. A/B against `baseline`.
    #
    # sha256_ni.cpp is compiled as a SEPARATE, NON-LTO object (NOLTO_SRC): the
    # SHA-extension instructions are legacy-SSE-only, and with -flto the linker
    # re-inlines the SHA code into the AVX2 EC loop, triggering an AVX<->SSE
    # transition on ~every instruction (~18x slower!). Isolating the TU keeps the
    # SHA as one legacy block with a single zeroupper-guarded boundary.
    ARCH_FLAGS="-mavx2 -mbmi2 -madx -msha -DSHA_NI"
    DEF_OUT="Cyclone_shani"
    NOLTO_SRC="sha256_ni.cpp" ;;
  *)
    echo "usage: $0 [baseline|native|adx|fused|simdtest|simdfield|shani] [output-name]" >&2; exit 2 ;;
esac
OUT="${OUT:-$DEF_OUT}"

COMMON="-std=c++17 -O3 -funroll-loops -ftree-vectorize -fstrict-aliasing \
-fno-semantic-interposition -fvect-cost-model=unlimited -fno-trapping-math \
-fipa-ra -fipa-modref -flto -fassociative-math -fopenmp"

SRC="Cyclone.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp Point.cpp \
ripemd160_avx2.cpp p2pkh_decoder.cpp sha256_avx2.cpp ${EXTRA_SRC:-}"
SRC="${SRC_OVERRIDE:-$SRC}"

set -x
# NON-LTO objects (see the `shani` note): compiled in isolation so -flto can't
# inline them into a mismatched-ISA hot loop.
NOLTO_OBJS=""
for f in ${NOLTO_SRC:-}; do
  obj="${f%.cpp}.nolto.o"
  # shellcheck disable=SC2086
  g++ -std=c++17 -O3 $ARCH_FLAGS -c "$f" -o "$obj"
  NOLTO_OBJS="$NOLTO_OBJS $obj"
done
# shellcheck disable=SC2086
g++ $COMMON $ARCH_FLAGS -o "$OUT" $SRC $NOLTO_OBJS
