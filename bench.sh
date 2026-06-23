#!/usr/bin/env bash
# Matched A/B benchmark harness. Alternates the given binaries round-by-round
# (so thermal throttling hits each equally), then reports per-binary medians and
# the delta of each vs the first binary. Throughput is scraped from the binary's
# own "Throughput : N Mkeys/s" benchmark line.
#
# Examples:
#   ./bench.sh                                       # Cyclone vs Cyclone_fused, -t16
#   ./bench.sh -t 8,16 -r 7 -s 30
#   ./bench.sh -e Cyclone,Cyclone_f256,Cyclone_f512  # FUSE_BLOCK sweep
#   ./bench.sh --skip-hash                           # EC-only (point-gen) isolation
set -uo pipefail

EXES="Cyclone Cyclone_fused"
THREADS="16"
SECS=20
ROUNDS=5
SKIP=""

while [ $# -gt 0 ]; do
  case "$1" in
    -e|--exes)    EXES="$(echo "$2" | tr ',' ' ')"; shift 2;;
    -t|--threads) THREADS="$(echo "$2" | tr ',' ' ')"; shift 2;;
    -s|--seconds) SECS="$2"; shift 2;;
    -r|--rounds)  ROUNDS="$2"; shift 2;;
    --skip-hash)  SKIP="--skip-hash"; shift;;
    -h|--help)    sed -n '2,11p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

FIRST="$(set -- $EXES; echo "$1")"

run_rate() { # $1=exe  $2=threads -> prints final throughput
  ./"$1" -b "$SECS" -t "$2" $SKIP 2>/dev/null | awk '/Throughput/{print $3}' | tail -1
}
median() { # numbers (whitespace-separated) on stdin -> median
  tr ' ' '\n' | grep -v '^$' | sort -n | \
    awk '{a[NR]=$1} END{ if(NR==0){print "nan"} \
          else if(NR%2){print a[(NR+1)/2]} else {printf "%.4f\n",(a[NR/2]+a[NR/2+1])/2} }'
}

mode="full pipeline"; [ -n "$SKIP" ] && mode="EC-only (--skip-hash)"
echo "bench: $EXES | ${SECS}s x ${ROUNDS} rounds | $mode"

for T in $THREADS; do
  echo ""
  echo "--- -t$T ---"
  unset RAW; declare -A RAW
  for E in $EXES; do RAW["$E"]=""; done
  for ((r=1; r<=ROUNDS; r++)); do
    for E in $EXES; do
      RAW["$E"]="${RAW["$E"]} $(run_rate "$E" "$T")"
    done
  done
  base="$(echo "${RAW[$FIRST]}" | median)"
  for E in $EXES; do
    med="$(echo "${RAW[$E]}" | median)"
    raw="$(echo "${RAW[$E]}" | sed 's/^ *//;s/  */,/g')"
    if [ "$E" = "$FIRST" ]; then
      delta="(ref)"
    else
      delta="$(awk -v m="$med" -v b="$base" 'BEGIN{printf "%+.1f%%", (m-b)/b*100}')"
    fi
    printf '%-22s median=%8.2f  %-8s  raw: %s\n' "$E" "$med" "$delta" "$raw"
  done
done
