# Minimal end-to-end find gate. Derives addresses independently (Python ecdsa),
# runs the binary over a range containing each key, and checks the reported
# private key matches. Usage:  ./e2e.ps1 -Exe Cyclone.exe
param([string]$Exe = "Cyclone.exe", [string]$Range = "1:3FFF")
$ErrorActionPreference = "Stop"
$Exe = (Resolve-Path $Exe).Path   # cwd is not on PATH; resolve to a full path

# Boundaries + both parities + the keys verified in prior sessions (0x100/0x802/0x1771).
$keys = @(1, 2, 256, 2050, 6001, 16383)

$derive = @'
import sys, hashlib
from ecdsa import SECP256k1, SigningKey
from proof import base58_encode
for a in sys.argv[1:]:
    k=int(a)
    sk=SigningKey.from_string(k.to_bytes(32,'big'),curve=SECP256k1)
    xy=sk.get_verifying_key().to_string(); x=xy[:32]; y=xy[32:]
    pre=b'\x03' if (int.from_bytes(y,'big')&1) else b'\x02'
    h=hashlib.new('ripemd160'); h.update(hashlib.sha256(pre+x).digest())
    pay=b'\x00'+h.digest()
    chk=hashlib.sha256(hashlib.sha256(pay).digest()).digest()[:4]
    print(format(k,'x'), base58_encode(pay+chk))
'@

$pairs = & python -c $derive @($keys | ForEach-Object { "$_" })
$fail = 0
foreach ($line in $pairs) {
    $kHex, $addr = $line.Trim().Split(' ')
    $want = $kHex.TrimStart('0'); if ($want -eq '') { $want = '0' }
    $out = & $Exe -r $Range -a $addr -t 1 2>&1 | Out-String
    $m = [regex]::Match($out, 'Private Key\s*:\s*([0-9a-fA-F]+)')
    $got = if ($m.Success) { $m.Groups[1].Value.TrimStart('0').ToLower() } else { '<none>' }
    if ($got -eq '') { $got = '0' }
    if ($got -eq $want.ToLower()) {
        Write-Host ("PASS  key=0x{0,-5} addr={1}" -f $kHex, $addr)
    } else {
        Write-Host ("FAIL  key=0x{0,-5} want={1} got={2} addr={3}" -f $kHex, $want, $got, $addr)
        $fail++
    }
}
if ($fail -eq 0) { Write-Host "E2E PASS (all $($keys.Count))"; exit 0 }
else { Write-Host "E2E FAIL ($fail of $($keys.Count))"; exit 1 }
