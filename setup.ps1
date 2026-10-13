# KeepSake - Phase 0 setup for Windows
#
# Run this once from PowerShell:
#     powershell -ExecutionPolicy Bypass -File C:\KeepSake\setup.ps1
#
# It is safe to run again: anything already done is skipped.

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"   # makes downloads far faster in Windows PowerShell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:PYTHONUTF8 = "1"
$Drive = $Root.Substring(0, 1)

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Ok($text)   { Write-Host "    OK    $text" -ForegroundColor Green }
function Warn($text) { Write-Host "    WARN  $text" -ForegroundColor Yellow }
function Fail($text) { Write-Host "    FAIL  $text" -ForegroundColor Red }
function Has($cmd)   { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Winget-Install($id, $name) {
    if (-not (Has "winget")) {
        Fail "winget is missing. Update 'App Installer' from the Microsoft Store, then run setup again."
        exit 1
    }
    Write-Host "    Installing $name (this can take a few minutes)..."
    winget install -e --id $id --silent --accept-package-agreements --accept-source-agreements | Out-Host
    Refresh-Path
}

Write-Host ""
Write-Host "KeepSake - Phase 0 setup" -ForegroundColor White
Write-Host "Folder: $Root"

# ---------------------------------------------------------------------------
Step "1/7  Checking this laptop"
New-Item -ItemType Directory -Force -Path "$Root\docs" | Out-Null
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$cs = Get-CimInstance Win32_ComputerSystem
$os = Get-CimInstance Win32_OperatingSystem
$ramGB = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
$freeGB = [math]::Round((Get-PSDrive -Name $Drive).Free / 1GB, 0)
$diskType = "unknown"
try {
    $part = Get-Partition -DriveLetter $Drive -ErrorAction Stop
    $disk = Get-PhysicalDisk -ErrorAction Stop | Where-Object { "$($_.DeviceId)" -eq "$($part.DiskNumber)" } | Select-Object -First 1
    if ($disk) { $diskType = "$($disk.MediaType), $($disk.BusType)" }
} catch { }
$report = @(
    "KeepSake laptop report - $(Get-Date -Format 'yyyy-MM-dd HH:mm')",
    "CPU:      $($cpu.Name.Trim())",
    "Cores:    $($cpu.NumberOfCores) physical, $($cpu.NumberOfLogicalProcessors) logical",
    "RAM:      $ramGB GB",
    "Windows:  $($os.Caption) ($($os.Version))",
    "Drive ${Drive}:  $diskType, $freeGB GB free"
)
$report | Set-Content -Encoding UTF8 "$Root\docs\phase0_laptop.txt"
$report | Select-Object -Skip 1 | ForEach-Object { Write-Host "    $_" }
if ($ramGB -lt 8)  { Warn "Under 8 GB RAM: keep test sets to about 500 photos." }
if ($freeGB -lt 150) { Warn "Under 150 GB free: put full weddings on an external SSD." }
if ($diskType -like "HDD*") { Warn "This drive is a hard disk; an SSD will be much faster for photos." }

# ---------------------------------------------------------------------------
Step "2/7  Python 3.11"
function Find-Python311 {
    if (Has "py") {
        $p = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $p) { return "$p".Trim() }
    }
    foreach ($c in @("$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
                     "C:\Program Files\Python311\python.exe")) {
        if (Test-Path $c) { return $c }
    }
    return $null
}
$Py311 = Find-Python311
if (-not $Py311) {
    Winget-Install "Python.Python.3.11" "Python 3.11"
    $Py311 = Find-Python311
}
if (-not $Py311) {
    Fail "Python 3.11 not found after install. Close PowerShell, open a new one, and run setup again."
    exit 1
}
Ok "Python 3.11 at $Py311"

# ---------------------------------------------------------------------------
Step "3/7  Node.js, Git and VS Code"
$needNode = $true
if (Has "node") {
    $nodeVer = "$(& node --version)".Trim()
    if ([int]($nodeVer.TrimStart("v").Split(".")[0]) -ge 20) { Ok "Node.js $nodeVer"; $needNode = $false }
    else { Warn "Node.js $nodeVer is older than 20; installing the LTS version" }
}
if ($needNode) {
    Winget-Install "OpenJS.NodeJS.LTS" "Node.js LTS"
    if (Has "node") { Ok "Node.js $(& node --version)" } else { Warn "Node.js installed; it works in a new PowerShell window" }
}
if (Has "git") { Ok "$(& git --version)" }
else {
    Winget-Install "Git.Git" "Git"
    if (Has "git") { Ok "$(& git --version)" } else { Warn "Git installed; it works in a new PowerShell window" }
}
if (Has "code") { Ok "VS Code" } else { Winget-Install "Microsoft.VisualStudioCode" "VS Code"; Ok "VS Code installed" }

# ---------------------------------------------------------------------------
Step "4/7  ExifTool"
$ExifDir = "$Root\tools\exiftool"
$ExifExe = "$ExifDir\exiftool.exe"
if (Test-Path $ExifExe) {
    Ok "ExifTool $(& $ExifExe -ver) in tools\exiftool"
} else {
    try {
        $ver = $null
        try { $ver = "$(Invoke-RestMethod -UseBasicParsing 'https://exiftool.org/ver.txt')".Trim() } catch { }
        if (-not ($ver -match '^\d+\.\d+$')) {
            $page = Invoke-WebRequest -UseBasicParsing "https://exiftool.org/"
            if ($page.Content -match 'exiftool-(\d+\.\d+)_64\.zip') { $ver = $Matches[1] } else { $ver = $null }
        }
        if (-not $ver) { throw "could not find the latest ExifTool version on exiftool.org" }
        $zip = Join-Path $env:TEMP "exiftool-${ver}_64.zip"
        $tmp = Join-Path $env:TEMP "exiftool-${ver}-extract"
        Write-Host "    Downloading ExifTool $ver ..."
        try {
            Invoke-WebRequest -UseBasicParsing -OutFile $zip "https://exiftool.org/exiftool-${ver}_64.zip"
        } catch {
            Invoke-WebRequest -UseBasicParsing -UserAgent "Wget" -OutFile $zip "https://sourceforge.net/projects/exiftool/files/exiftool-${ver}_64.zip/download"
        }
        if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
        Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force
        $kexe = Get-ChildItem -LiteralPath $tmp -Recurse -File | Where-Object { $_.Name -eq "exiftool(-k).exe" } | Select-Object -First 1
        if (-not $kexe) { throw "exiftool(-k).exe was not in the download" }
        New-Item -ItemType Directory -Force -Path $ExifDir | Out-Null
        Get-ChildItem -LiteralPath $kexe.DirectoryName | Copy-Item -Destination $ExifDir -Recurse -Force
        Rename-Item -LiteralPath (Join-Path $ExifDir "exiftool(-k).exe") -NewName "exiftool.exe" -Force
        Ok "ExifTool $(& $ExifExe -ver) installed in tools\exiftool"
    } catch {
        Fail "ExifTool download failed: $($_.Exception.Message)"
        Write-Host "    Manual fix: download the Windows 64-bit zip from https://exiftool.org,"
        Write-Host "    copy everything inside its folder into $ExifDir,"
        Write-Host "    and rename exiftool(-k).exe to exiftool.exe. Then run setup again."
    }
}

# ---------------------------------------------------------------------------
Step "5/7  Python environment and libraries"
$VenvPy = "$Root\.venv\Scripts\python.exe"
if (Test-Path $VenvPy) {
    $venvVer = "$(& $VenvPy -c "import sys; print('%d.%d' % sys.version_info[:2])")".Trim()
    if ($venvVer -ne "3.11") {
        Warn "Existing .venv uses Python $venvVer; rebuilding it with 3.11"
        Remove-Item -Recurse -Force "$Root\.venv"
    }
}
if (-not (Test-Path $VenvPy)) {
    Write-Host "    Creating .venv ..."
    & $Py311 -m venv "$Root\.venv"
}
if (-not (Test-Path $VenvPy)) { Fail "Could not create the .venv folder."; exit 1 }
& $VenvPy -m pip install --upgrade pip --quiet --disable-pip-version-check
Write-Host "    Installing libraries (first time: 2-5 minutes) ..."
& $VenvPy -m pip install -r "$Root\requirements.txt" --disable-pip-version-check
if ($LASTEXITCODE -ne 0) {
    Fail "Library install failed. Copy the red error text above and send it to Claude."
    exit 1
}
Ok "Libraries installed"
& $VenvPy "$Root\scripts\make_edge_cases.py" | Out-Host

# ---------------------------------------------------------------------------
Step "6/7  Git repository"
if (Has "git") {
    if (-not "$(& git config --global user.name)") {
        $n = Read-Host "    Your name for Git (for example: Kavita)"
        & git config --global user.name "$n"
    }
    if (-not "$(& git config --global user.email)") {
        $e = Read-Host "    The email you use on GitHub"
        & git config --global user.email "$e"
    }
    if (-not (Test-Path "$Root\.git")) {
        & git -C $Root init -b main | Out-Null
        & git -C $Root add -A
        & git -C $Root commit -q -m "Phase 0: KeepSake project skeleton"
        if ($LASTEXITCODE -eq 0) { Ok "Git repository created with the first commit" }
        else { Warn "Repository created, but the first commit failed. Run: git commit -m ""Phase 0""" }
    } else {
        Ok "Git repository already exists"
    }
    $tracked = & git -C $Root ls-files
    $photoFiles = $tracked | Where-Object { $_ -match '\.(cr2|cr3|nef|arw|raf|rw2|orf|dng|jpe?g|heic|hif)$' }
    if ($photoFiles) { Fail "Photos are tracked by Git: $($photoFiles -join ', ')" } else { Ok "No photos tracked by Git" }
} else {
    Warn "Git works in a new PowerShell window. Open one and run setup again to create the repository."
}

# ---------------------------------------------------------------------------
Step "7/7  Speed setting and final check"
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
$excluded = $false
try { $excluded = (Get-MpPreference -ErrorAction Stop).ExclusionPath -contains $Root } catch { }
if ($excluded) {
    Ok "Windows Defender already skips $Root"
} elseif ($isAdmin) {
    $answer = Read-Host "    Let Windows Defender skip $Root? Photos read much faster. (y/n)"
    if ($answer -eq "y") {
        try { Add-MpPreference -ExclusionPath $Root -ErrorAction Stop; Ok "Defender will skip $Root" }
        catch { Warn "Could not change Defender: $($_.Exception.Message)" }
    } else { Ok "Defender setting left unchanged" }
} else {
    Warn "Optional speed-up: in an Administrator PowerShell, run:"
    Write-Host "          Add-MpPreference -ExclusionPath $Root"
}

Write-Host ""
& $VenvPy "$Root\scripts\check_setup.py"
$checkOk = ($LASTEXITCODE -eq 0)
& $VenvPy -m pytest -q --rootdir "$Root" "$Root\tests" | Out-Host

Write-Host ""
if ($checkOk) {
    Write-Host "Setup complete." -ForegroundColor Green
} else {
    Write-Host "Setup finished with problems: see the FAIL lines above." -ForegroundColor Yellow
}
Write-Host "Next: put test photos in $Root\data\test_photos, then run:  .\ks list   and   .\ks previews"
