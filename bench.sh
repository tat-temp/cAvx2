#!/usr/bin/env bash
# Matched A/B benchmark harness. Alternates the variants round-by-round (so
# thermal throttling hits each equally), then reports per-variant medians and
# the delta of each vs the first. Throughput is scraped from the binary's own
# "Throughput : N Mkeys/s" benchmark line.
#
# Two modes:
#   binaries : compare different binaries (default flags)
#     ./bench.sh -e Cyclone,Cyclone_f256,Cyclone_f512 -t 16 -s 30 -r 7
#   sweep    : compare flag-sets on ONE binary (pipe-separated; ""=default, first=ref)
#     ./bench.sh -e Cyclone --sweep "|-g 1024|-g 2048|-g 8192|-g 16384"
#     ./bench.sh -e Cyclone --sweep "|--ilp 2|--ilp 4|--ilp 6|--ilp 8"
#
#   -s seconds (default 20)   -r rounds (default 5)   --skip-hash  EC-only isolation
set -uo pipefail

EXES="Cyclone Cyclone_fused"
THREADS="16"
SECS=20
ROUNDS=5
SKIP=""
SWEEP=""

while [ $# -gt 0 ]; do
  case "$1" in
    -e|--exes)    EXES="$(echo "$2" | tr ',' ' ')"; shift 2;;
    -t|--threads) THREADS="$(echo "$2" | tr ',' ' ')"; shift 2;;
    -s|--seconds) SECS="$2"; shift 2;;
    -r|--rounds)  ROUNDS="$2"; shift 2;;
    --sweep)      SWEEP="$2"; shift 2;;
    --skip-hash)  SKIP="--skip-hash"; shift;;
    -h|--help)    sed -n '2,17p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

FIRST="$(set -- $EXES; echo "$1")"

# Build the variant list: parallel arrays of label / binary / extra-args.
VLABEL=(); VEXE=(); VARGS=()
if [ -n "$SWEEP" ]; then
  IFS='|' read -ra _sets <<< "$SWEEP"
  for s in "${_sets[@]}"; do
    lbl="$s"; [ -z "$lbl" ] && lbl="(default)"
    VLABEL+=("$lbl"); VEXE+=("$FIRST"); VARGS+=("$s")
  done
else
  for E in $EXES; do VLABEL+=("$E"); VEXE+=("$E"); VARGS+=(""); done
fi
NV=${#VLABEL[@]}

run_rate() { # $1=exe  $2=threads  $3=extra args -> final throughput
  ./"$1" -b "$SECS" -t "$2" $3 $SKIP 2>/dev/null | awk '/Throughput/{print $3}' | tail -1
}
median() { # whitespace-separated numbers on stdin -> median
  tr ' ' '\n' | grep -v '^$' | sort -n | \
    awk '{a[NR]=$1} END{ if(NR==0){print "nan"} \
          else if(NR%2){print a[(NR+1)/2]} else {printf "%.4f\n",(a[NR/2]+a[NR/2+1])/2} }'
}

mode="full pipeline"; [ -n "$SKIP" ] && mode="EC-only (--skip-hash)"
echo "bench: $FIRST | ${SECS}s x ${ROUNDS} rounds | $mode | ${NV} variants"

for T in $THREADS; do
  echo ""
  echo "--- -t$T ---"
  RATES=(); for ((i=0;i<NV;i++)); do RATES[$i]=""; done
  for ((r=1;r<=ROUNDS;r++)); do
    for ((i=0;i<NV;i++)); do
      RATES[$i]="${RATES[$i]} $(run_rate "${VEXE[$i]}" "$T" "${VARGS[$i]}")"
    done
  done
  base="$(echo "${RATES[0]}" | median)"
  for ((i=0;i<NV;i++)); do
    med="$(echo "${RATES[$i]}" | median)"
    raw="$(echo "${RATES[$i]}" | sed 's/^ *//;s/  */,/g')"
    if [ $i -eq 0 ]; then delta="(ref)"; else delta="$(awk -v m="$med" -v b="$base" 'BEGIN{printf "%+.1f%%",(m-b)/b*100}')"; fi
    printf '%-26s median=%8.2f  %-8s  raw: %s\n' "${VLABEL[$i]}" "$med" "$delta" "$raw"
  done
done
