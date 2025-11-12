# run_all.ps1
# Automatically run all Python scripts inside the "Plotting&Tables" folder and its subfolders

Write-Host "=== Running all plotting scripts ==="

# Get the folder containing this script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# If this script is inside the same folder as "Plotting&Tables", use that path
if (Test-Path (Join-Path $ScriptDir "Plotting&Tables")) {
    $PlottingDir = Join-Path $ScriptDir "Plotting&Tables"
}
else {
    # Otherwise, assume the script is already inside Plotting&Tables
    $PlottingDir = $ScriptDir
}

# Detect Python executable from the active environment
try {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
} catch {
    Write-Error "Python not found in PATH. Please activate your environment first."
    exit 1
}

Write-Host "Using Python: $PythonExe"
Write-Host "Plotting folder: $PlottingDir"

# Find all .py files in AVQE, VQD, and VQE subfolders
$pyFiles = Get-ChildItem -Path $PlottingDir -Recurse -Filter *.py

if (-not $pyFiles) {
    Write-Host "No Python scripts found in $PlottingDir"
    exit 0
}

# Run each script sequentially
foreach ($file in $pyFiles) {
    Write-Host "`n--- Running $($file.FullName) ---"
    & $PythonExe $file.FullName
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Script $($file.Name) exited with code $LASTEXITCODE"
    }
}

Write-Host "`n=== All plotting scripts finished ==="
