# Hyllora anonymizer for testing on Windows (packaging/testrepo.py). start.bat runs this; it never changes itself.
#
# 1. A new version on GitHub? Fetch only the code (a few hundred KB) over this folder. .venv, data and models stay.
# 2. The packages: installed the first time (a few GB), and again only when packaging/requirements.txt changed.
# 3. The models (checked by checksum, fetched only if missing), then the page on http://127.0.0.1:8430.
#
# Plain ASCII on purpose: Windows PowerShell 5.1 reads a script without a BOM in the old code page.
param([switch]$NoUpdate)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is many times slower with the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Repo = "Samsam2006-bit/hyllora-anonymizer"
$Here = $PSScriptRoot
$Py = Join-Path $Here ".venv\Scripts\python.exe"
$Port = 8430
$Cuda = "cu128"
Set-Location $Here
$env:PYTHONUTF8 = "1"   # else Python reads and writes text in the old Windows code page
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"

function Fail($msg) {
  Write-Host ""
  Write-Host $msg -ForegroundColor Red
  Write-Host "Ta en skarmbild av fonstret och skicka den till Sameh."
  Read-Host "Tryck Enter for att stanga"
  exit 1
}

function Run {
  & $args[0] @($args | Select-Object -Skip 1)
  if ($LASTEXITCODE -ne 0) { Fail "Det gick inte: $($args -join ' ')" }
}

function PortInUse {
  $c = New-Object System.Net.Sockets.TcpClient
  try { $c.Connect("127.0.0.1", $Port); return $true } catch { return $false } finally { $c.Close() }
}

if (PortInUse) {
  Write-Host "Anonymiseraren kor redan. Oppnar sidan."
  Start-Process "http://127.0.0.1:$Port/"
  Start-Sleep 3
  exit 0
}

# 1. a new version
if (-not $NoUpdate) {
  try {
    $latest = Invoke-RestMethod "https://api.github.com/repos/$Repo/commits/main" -TimeoutSec 15 `
      -Headers @{ "User-Agent" = "hyllora-anonymizer" }
    $have = if (Test-Path ".version") { (Get-Content ".version" -Raw).Trim() } else { "" }
    if ($latest.sha -ne $have) {
      Write-Host "Hamtar ny version: $(($latest.commit.message -split "`n")[0])" -ForegroundColor Cyan
      $tmp = Join-Path $env:TEMP "hyllora-anonymizer-update"
      Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
      New-Item -ItemType Directory $tmp | Out-Null
      Invoke-WebRequest "https://github.com/$Repo/archive/$($latest.sha).zip" -OutFile "$tmp\code.zip" -TimeoutSec 120
      Expand-Archive "$tmp\code.zip" "$tmp\code"
      $src = (Get-ChildItem "$tmp\code" -Directory | Select-Object -First 1).FullName
      foreach ($d in "anonymizer", "inferero", "static", "packaging") {
        robocopy (Join-Path $src $d) (Join-Path $Here $d) /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy $d gav $LASTEXITCODE" }
      }
      # run.ps1, README ... but never start.bat: cmd reads it line by line while it waits for this script
      Get-ChildItem $src -File | Where-Object { $_.Name -ne "start.bat" } | Copy-Item -Destination $Here -Force
      Set-Content ".version" $latest.sha
      Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
      # the new run.ps1 from here on (this one was read whole before it was replaced)
      & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here "run.ps1") -NoUpdate
      exit $LASTEXITCODE
    }
  } catch {
    Write-Host "Kunde inte leta efter en ny version ($($_.Exception.Message)). Startar den som finns." -ForegroundColor Yellow
  }
}

# 2. the packages
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "Installerar uv, som hamtar Python ..." -ForegroundColor Cyan
  try { Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression } catch { Fail "Kunde inte installera uv: $($_.Exception.Message)" }
  if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Fail "uv installerades men hittas inte" }
}
$want = "$((Get-FileHash 'packaging\requirements.txt').Hash) $Cuda py3.12"
$ready = Join-Path $Here ".venv\ready"
$have = if (Test-Path $ready) { (Get-Content $ready -Raw).Trim() } else { "" }
if ($have -ne $want) {
  if (-not (Test-Path $Py)) {
    Write-Host "Forsta gangen: Python, torch for grafikkortet och modellerna. Det tar en stund." -ForegroundColor Cyan
    Remove-Item -Recurse -Force ".venv" -ErrorAction SilentlyContinue
    Run uv venv --python 3.12 .venv
  } else {
    Write-Host "Paketen har andrats: installerar om dem." -ForegroundColor Cyan
  }
  Run uv pip install --python $Py torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
  Run uv pip install --python $Py -r "packaging\requirements.txt"
  Set-Content $ready $want
}

# 3. the models, then the page
Run $Py get_models.py
& $Py -c "import torch; print('Grafikkort:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'INGET, kor pa processorn. Uppdatera NVIDIA-drivrutinen.')"
$version = if (Test-Path ".version") { (Get-Content ".version" -Raw).Trim().Substring(0, 7) } else { "?" }
Write-Host ""
Write-Host "Anonymiseraren kor pa http://127.0.0.1:$Port  (version $version)" -ForegroundColor Green
Write-Host "Stang det har fonstret for att stoppa den. Starta start.bat igen for att uppdatera."
Start-Process cmd -ArgumentList "/c timeout /t 6 >nul & start http://127.0.0.1:$Port/" -WindowStyle Hidden
& $Py -m uvicorn anonymizer.server:app --host 127.0.0.1 --port $Port
if ($LASTEXITCODE -ne 0) { Fail "Anonymiseraren stannade (kod $LASTEXITCODE)." }
