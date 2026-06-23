# Matched A/B benchmark harness. Alternates the given binaries round-by-round
# (so thermal throttling hits both equally), then reports per-binary medians and
# the delta of each vs the first binary. Throughput is scraped from the binary's
# own "Throughput : N Mkeys/s" benchmark line.
#
# Examples:
#   ./bench.ps1                                   # base vs fused, -t16, full pipeline
#   ./bench.ps1 -Threads 8,16 -Rounds 7 -Seconds 30
#   ./bench.ps1 -Exes Cyclone.exe,Cyclone_f256.exe,Cyclone_f512.exe   # FUSE_BLOCK sweep
#   ./bench.ps1 -SkipHash                          # EC-only (point-gen) isolation
param(
    [string[]]$Exes    = @("Cyclone.exe", "Cyclone_fused.exe"),
    [int[]]   $Threads = @(16),
    [int]     $Seconds = 20,
    [int]     $Rounds  = 5,
    [switch]  $SkipHash
)
$ErrorActionPreference = "Stop"
$paths = $Exes | ForEach-Object { (Resolve-Path $_).Path }

function Get-Median([double[]]$xs) {
    $s = $xs | Sort-Object
    $n = $s.Count
    if ($n -eq 0) { return [double]::NaN }
    if ($n % 2) { return $s[[int](($n - 1) / 2)] }
    return ($s[$n / 2 - 1] + $s[$n / 2]) / 2.0
}
function Measure-Rate([string]$path, [int]$t) {
    $a = @("-b", "$Seconds", "-t", "$t")
    if ($SkipHash) { $a += "--skip-hash" }
    $o = & $path @a 2>&1 | Out-String
    $m = [regex]::Match($o, 'Throughput\s*:\s*([0-9.]+)')
    if ($m.Success) { return [double]$m.Groups[1].Value } else { return -1.0 }
}

$mode = if ($SkipHash) { "EC-only (--skip-hash)" } else { "full pipeline" }
Write-Host ("bench: {0}  | {1}s x {2} rounds | {3}" -f ($Exes -join " vs "), $Seconds, $Rounds, $mode)

foreach ($t in $Threads) {
    $acc = @{}; foreach ($e in $Exes) { $acc[$e] = @() }
    for ($r = 1; $r -le $Rounds; $r++) {
        for ($i = 0; $i -lt $Exes.Count; $i++) {
            $acc[$Exes[$i]] += (Measure-Rate $paths[$i] $t)
        }
    }
    Write-Host ("`n--- -t{0} ---" -f $t)
    $baseMed = Get-Median $acc[$Exes[0]]
    foreach ($e in $Exes) {
        $med = Get-Median $acc[$e]
        $delta = if ($e -eq $Exes[0]) { "  (ref)" } else { "{0,7:P1}" -f (($med - $baseMed) / $baseMed) }
        Write-Host ("{0,-22} median={1,7:N2}  {2}   raw: {3}" -f $e, $med, $delta, (($acc[$e] | ForEach-Object { "{0:N1}" -f $_ }) -join ","))
    }
}
