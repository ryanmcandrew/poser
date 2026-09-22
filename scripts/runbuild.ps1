<#
.SYNOPSIS
    Build the FPV Pose Tracker addon zip into dist/.

.DESCRIPTION
    Wraps build.py so a release does not depend on remembering which
    interpreter to call or which directory to stand in. Finds a Python,
    runs the build from the repository root whatever the current location
    is, and then reads the produced zip back to confirm it is actually
    installable rather than merely present.

    The verification is the point of the wrapper as much as the build is:
    the zip's top-level folder becomes the addon's Python package name in
    Blender, so a build that lays its files out wrongly produces a zip
    that installs and then fails to import. That is worth catching here
    rather than in Blender.

.PARAMETER Python
    Interpreter to use. Defaults to the first of `py -3`, `python` or
    `python3` that runs.

.PARAMETER Clean
    Prune older zips from dist/, keeping the newest -Keep of them. Only
    ever touches files matching the addon's own <package>-<version>.zip
    pattern, lists what it will remove, and asks first unless -Force.

    dist/ is gitignored, so the zips in it are the only copies of past
    releases that exist -- git cannot restore them and an older version
    cannot be rebuilt from current source. Hence the confirmation and the
    default of keeping several.

.PARAMETER Keep
    How many of the newest zips -Clean leaves in place. Default 5; 0
    removes every build except the one this run produces.

.PARAMETER Force
    Let -Clean delete without asking.

.PARAMETER SkipVerify
    Build without inspecting the resulting zip.

.PARAMETER Reveal
    Open dist/ in Explorer when the build succeeds.

.EXAMPLE
    .\scripts\runbuild.ps1
    Build and verify with an auto-detected Python.

.EXAMPLE
    .\scripts\runbuild.ps1 -Clean -Keep 3
    Build and verify, then prune dist/ down to the three newest zips
    after confirming the list.

.EXAMPLE
    .\scripts\runbuild.ps1 -Python "C:\Program Files\Blender Foundation\Blender 4.2\4.2\python\bin\python.exe"
    Build with a specific interpreter. build.py only needs the standard
    library, so any Python 3 works -- including Blender's own.
#>

# SupportsShouldProcess gives -WhatIf over the prune. ConfirmImpact is
# deliberately left at its default: raising it to High would make
# ShouldProcess prompt on its own, on top of the script's own prompt, and
# that second prompt cannot be waved off with -Force -- it throws outright
# in a non-interactive shell, which is exactly where -Force gets used.
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string] $Python,
    [switch] $Clean,
    [ValidateRange(0, 1000)]
    [int] $Keep = 5,
    [switch] $Force,
    [switch] $SkipVerify,
    [switch] $Reveal
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$BuildScript = Join-Path $Root 'build.py'
$Dist = Join-Path $Root 'dist'

if (-not (Test-Path $BuildScript)) {
    throw "build.py not found at $BuildScript -- is this script still inside the repo?"
}

function Resolve-Python {
    param([string] $Preferred)

    # Each candidate is the executable plus the arguments that select a
    # Python 3 from it; the `py` launcher needs to be told.
    if ($Preferred) {
        $candidates = @(, @($Preferred))
    } else {
        $candidates = @(@('py', '-3'), @('python'), @('python3'))
    }

    foreach ($candidate in $candidates) {
        $exe = $candidate[0]
        if ($candidate.Count -gt 1) {
            $prefix = $candidate[1..($candidate.Count - 1)]
        } else {
            $prefix = @()
        }
        # A bare name has to be on PATH; a path has to exist. Windows also
        # ships `python.exe` stubs that only open the Store, so the version
        # probe below is what actually decides.
        if (-not (Test-Path $exe) -and -not (Get-Command $exe -ErrorAction SilentlyContinue)) {
            continue
        }
        try {
            $version = & $exe @prefix '--version' 2>&1
        } catch {
            continue
        }
        if ($LASTEXITCODE -eq 0 -and "$version" -match 'Python 3') {
            return [pscustomobject]@{
                Exe     = $exe
                Prefix  = $prefix
                Version = ("$version").Trim()
            }
        }
    }
    return $null
}

$py = Resolve-Python -Preferred $Python
if ($null -eq $py) {
    if ($Python) {
        throw "'$Python' is not a working Python 3 interpreter."
    }
    throw "No Python 3 found. Install one, or pass -Python <path to python.exe>."
}
Write-Host "Interpreter: $($py.Exe) ($($py.Version))" -ForegroundColor DarkGray

# The package name build.py writes into the zip, read from build.py so the
# two cannot drift apart.
$pkg = 'fpv_pose_tracker'
$buildText = Get-Content -Raw -Path $BuildScript
if ($buildText -match "(?m)^\s*PKG\s*=\s*['`"]([^'`"]+)['`"]") {
    $pkg = $Matches[1]
}

function Invoke-DistPrune {
    param(
        [string] $Directory,
        [string] $Package,
        [int]    $KeepCount,
        [switch] $NoPrompt
    )

    if (-not (Test-Path $Directory)) {
        Write-Host "No dist/ yet -- nothing to prune" -ForegroundColor DarkGray
        return
    }
    $all = @(Get-ChildItem -Path $Directory -Filter "$Package-*.zip" -File |
             Sort-Object LastWriteTime -Descending)
    if ($all.Count -le $KeepCount) {
        Write-Host "dist/ holds $($all.Count) zip(s); keeping $KeepCount -- nothing to prune" -ForegroundColor DarkGray
        return
    }
    $doomed = @($all | Select-Object -Skip $KeepCount)

    Write-Host ""
    Write-Host "Prune $($doomed.Count) of $($all.Count) zip(s) in dist/, keeping the newest $KeepCount" -ForegroundColor Yellow
    foreach ($file in $doomed) {
        Write-Host ("  {0}  ({1:yyyy-MM-dd})" -f $file.Name, $file.LastWriteTime) -ForegroundColor Yellow
    }
    # dist/ is gitignored: these files are not in version control and an
    # older version cannot be rebuilt from current source, so this is not
    # an undoable convenience.
    Write-Host "  These are not in git and cannot be recovered." -ForegroundColor Yellow

    if (-not $NoPrompt) {
        $answer = Read-Host "Delete them? [y/N]"
        if ($answer -notmatch '^(y|yes)$') {
            Write-Host "Kept everything." -ForegroundColor DarkGray
            return
        }
    }
    foreach ($file in $doomed) {
        if ($PSCmdlet.ShouldProcess($file.FullName, 'Remove build artefact')) {
            Remove-Item -LiteralPath $file.FullName -Force -Confirm:$false
            Write-Host "Removed $($file.Name)" -ForegroundColor DarkGray
        }
    }
}

Push-Location $Root
try {
    & $py.Exe @($py.Prefix) $BuildScript
    $buildExit = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($buildExit -ne 0) {
    throw "build.py failed with exit code $buildExit"
}

$zip = Get-ChildItem -Path $Dist -Filter "$pkg-*.zip" -File |
       Sort-Object LastWriteTime -Descending |
       Select-Object -First 1
if ($null -eq $zip) {
    throw "build.py reported success but no $pkg-*.zip is in $Dist"
}

if (-not $SkipVerify) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($zip.FullName)
    try {
        $names = @($archive.Entries | ForEach-Object { $_.FullName })
    } finally {
        $archive.Dispose()
    }

    $problems = @()
    if ($names.Count -eq 0) {
        $problems += 'the zip is empty'
    }
    # Blender imports the addon as a package named after the zip's top
    # folder, so every entry has to sit under it.
    $stray = @($names | Where-Object { -not $_.StartsWith("$pkg/") })
    if ($stray) {
        $problems += "entries outside $pkg/: $($stray -join ', ')"
    }
    if ($names -notcontains "$pkg/__init__.py") {
        $problems += "missing $pkg/__init__.py (Blender would not see an addon)"
    }
    $junk = @($names | Where-Object { $_ -like '*__pycache__*' -or $_ -like '*.pyc' })
    if ($junk) {
        $problems += "build artefacts included: $($junk -join ', ')"
    }

    if ($problems) {
        foreach ($problem in $problems) { Write-Host "  - $problem" -ForegroundColor Red }
        throw "$($zip.Name) is not a valid addon zip"
    }

    $modules = @($names | Where-Object { $_ -like "$pkg/*.py" }).Count
    Write-Host "Verified: $($names.Count) entries, $modules modules, package '$pkg'" -ForegroundColor DarkGray
}

# Pruned only after a build that produced and passed a zip, so the fresh
# one is always among the kept and a failed run never deletes anything.
if ($Clean) {
    Invoke-DistPrune -Directory $Dist -Package $pkg -KeepCount $Keep -NoPrompt:$Force
}

$sizeKb = [math]::Round($zip.Length / 1KB)
Write-Host ""
Write-Host "Built $($zip.Name) ($sizeKb KB)" -ForegroundColor Green
Write-Host "  $($zip.FullName)"
Write-Host "Install: Blender > Edit > Preferences > Add-ons > Install..., pick that zip"
Write-Host "  (Blender 4.2+ also accepts the zip dragged onto a 3D Viewport)"

if ($Reveal) {
    Start-Process explorer.exe -ArgumentList "/select,`"$($zip.FullName)`""
}
