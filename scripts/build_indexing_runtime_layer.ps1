param(
    [string]$OutputDirectory = "build/indexing-runtime-layer",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = [IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..")
)
$AllowedBuildRoot = [IO.Path]::GetFullPath(
    (Join-Path $RepositoryRoot "build")
)
$RequestedOutput = if ([IO.Path]::IsPathRooted($OutputDirectory)) {
    [IO.Path]::GetFullPath($OutputDirectory)
} else {
    [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $OutputDirectory))
}
$AllowedPrefix = $AllowedBuildRoot.TrimEnd(
    [IO.Path]::DirectorySeparatorChar
) + [IO.Path]::DirectorySeparatorChar
if (-not $RequestedOutput.StartsWith(
    $AllowedPrefix,
    [StringComparison]::OrdinalIgnoreCase
)) {
    throw "Layer output must be a child of the repository build directory"
}

$ArchivePath = "$RequestedOutput.zip"
if ($Clean) {
    if (Test-Path -LiteralPath $RequestedOutput) {
        Remove-Item -LiteralPath $RequestedOutput -Recurse -Force
    }
    if (Test-Path -LiteralPath $ArchivePath) {
        Remove-Item -LiteralPath $ArchivePath -Force
    }
    Write-Output "Removed local indexing layer output"
    exit 0
}

$PythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($PythonVersion.Trim() -ne "3.12") {
    throw "The indexing runtime layer must be built with Python 3.12"
}
if ((Test-Path -LiteralPath $RequestedOutput) -or (
    Test-Path -LiteralPath $ArchivePath
)) {
    throw "Layer output already exists; review it or run this script with -Clean"
}

$LayerPython = Join-Path $RequestedOutput "python"
$LayerRequirements = Join-Path $RepositoryRoot `
    "lambda/indexing_runtime_requirements.lock.txt"
New-Item -ItemType Directory -Force -Path $LayerPython | Out-Null
python -m pip install `
    --requirement $LayerRequirements `
    --target $LayerPython `
    --platform manylinux2014_x86_64 `
    --implementation cp `
    --python-version 3.12 `
    --abi cp312 `
    --only-binary=:all: `
    --no-deps `
    --disable-pip-version-check `
    --no-compile
if ($LASTEXITCODE -ne 0) {
    throw "Layer dependency installation failed"
}

Compress-Archive -Path (Join-Path $RequestedOutput "*") `
    -DestinationPath $ArchivePath
python (Join-Path $RepositoryRoot "scripts/inspect_indexing_runtime_layer.py") `
    --archive $ArchivePath `
    --requirements $LayerRequirements
if ($LASTEXITCODE -ne 0) {
    throw "Layer archive inspection failed"
}
