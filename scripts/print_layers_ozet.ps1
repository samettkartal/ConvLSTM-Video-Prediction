# layers_ozet.html -> layers_ozet.pdf (mor tema, yazdırma renkleri açık)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$html = (Resolve-Path (Join-Path $root "layers_ozet.html")).Path
$pdf = Join-Path $root "layers_ozet.pdf"
$uri = "file:///" + ($html -replace "\\", "/")

$edgeCandidates = @(
    "${env:ProgramFiles}\Microsoft\Edge\Application\msedge.exe",
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
)
$edge = $edgeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $edge) {
    Write-Host "Microsoft Edge bulunamadi. layers_ozet.html dosyasini acip Ctrl+P ile PDF olarak kaydedin."
    exit 1
}

if (Test-Path $pdf) { Remove-Item $pdf -Force }
& $edge --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$pdf" $uri
if (-not (Test-Path $pdf)) {
    Write-Host "PDF olusturulamadi. layers_ozet.html -> Yazdir -> PDF olarak kaydet."
    exit 1
}
Write-Host "PDF hazir: $pdf"
