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
  *)
    echo "usage: $0 [baseline|native|adx] [output-name]" >&2; exit 2 ;;
esac
OUT="${OUT:-$DEF_OUT}"

COMMON="-std=c++17 -O3 -funroll-loops -ftree-vectorize -fstrict-aliasing \
-fno-semantic-interposition -fvect-cost-model=unlimited -fno-trapping-math \
-fipa-ra -fipa-modref -flto -fassociative-math -fopenmp"

SRC="Cyclone.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp Point.cpp \
ripemd160_avx2.cpp p2pkh_decoder.cpp sha256_avx2.cpp"

set -x
# shellcheck disable=SC2086
g++ $COMMON $ARCH_FLAGS -o "$OUT" $SRC
