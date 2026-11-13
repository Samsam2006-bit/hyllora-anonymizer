# Builds the anonymizer's Windows installer: dist\HylloraAnonymizer-Setup-<version>.exe (ARCHITECTURE §13c).
# Run on a 64-bit Windows computer with the repo checked out, internet, and Inno Setup 6
# (winget install JRSoftware.InnoSetup):
#
#   powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1 -Version 0.1.0
#
# It lays out build\stage as it is installed: Python from python.org's NuGet package (a whole Python, not the
# embeddable one), torch for NVIDIA cards from PyTorch's CUDA index, the pinned packages, the program and the three
# models (packaging/stage.py), and WinSW for the service. Then Inno Setup packs it.
param(
  [Parameter(Mandatory = $true)][string]$Version,
  [string]$Python = "3.12.10",
  [string]$Cuda = "cu128",   # PyTorch's CUDA build: cu128 runs on RTX 20xx–50xx with a recent NVIDIA driver
  [string]$WinSW = "v2.12.0"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest is many times slower with the progress bar

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $Here "..\..")
$Build = Join-Path $Here "build"
$Stage = Join-Path $Build "stage"
$Py = Join-Path $Stage "python\python.exe"

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Run($exe) {
  & $exe @args
  if ($LASTEXITCODE -ne 0) { throw "$exe $args gav $LASTEXITCODE" }
}

New-Item -ItemType Directory -Force -Path $Build | Out-Null

if (-not (Test-Path $Py)) {
  Step "Python $Python"
  $nupkg = Join-Path $Build "python.$Python.zip"
  Invoke-WebRequest "https://www.nuget.org/api/v2/package/python/$Python" -OutFile $nupkg
  $unz = Join-Path $Build "python-nuget"
  Remove-Item -Recurse -Force $unz -ErrorAction SilentlyContinue
  Expand-Archive $nupkg $unz
  New-Item -ItemType Directory -Force -Path $Stage | Out-Null
  Move-Item (Join-Path $unz "tools") (Join-Path $Stage "python")
  Remove-Item -Recurse -Force $unz
}

Step "Paketen (torch $Cuda först, annars drar Ultralytics in versionen utan grafikkort)"
Run $Py -m pip install --upgrade --no-warn-script-location pip
Run $Py -m pip install --no-warn-script-location torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
Run $Py -m pip install --no-warn-script-location -r (Join-Path $Root "packaging\requirements.txt")
Run $Py -c "import torch; assert torch.version.cuda, 'torch utan CUDA'; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"
& $Py -m pip freeze | Out-File -Encoding utf8 (Join-Path $Build "packages.txt")

Step "Programmet och modellerna"
Run $Py (Join-Path $Root "packaging\stage.py") $Stage
Get-ChildItem -Path $Stage -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

Step "Kontroll: sidan och anonymiseraren går att läsa in med den inbyggda Pythonen"
Push-Location (Join-Path $Stage "app")
try {
  $env:HYLLORA_DATA = Join-Path $Build "check-data"
  $env:HYLLORA_MODELS = Join-Path $Stage "models"
  Run $Py -c "import anonymizer.server, anonymizer.main; from inferero import weights; [weights.path(m) for m in ('yolo26s-pose.pt', 'vit_pe_core_base_patch16_224.fb.safetensors')]; print('ok')"
} finally {
  Pop-Location
  Remove-Item Env:HYLLORA_DATA, Env:HYLLORA_MODELS
}

$Svc = Join-Path $Build "WinSW-x64.exe"
if (-not (Test-Path $Svc)) {
  Step "WinSW $WinSW"
  Invoke-WebRequest "https://github.com/winsw/winsw/releases/download/$WinSW/WinSW-x64.exe" -OutFile $Svc
}

Step "Installationsprogrammet"
$iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
          "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Hittar inte Inno Setup 6: winget install JRSoftware.InnoSetup" }
Push-Location $Here
try { Run $iscc "/DAppVersion=$Version" "/DStage=build\stage" "installer.iss" } finally { Pop-Location }
Write-Host "`nKlart: $(Join-Path $Root "dist\HylloraAnonymizer-Setup-$Version.exe")" -ForegroundColor Green
