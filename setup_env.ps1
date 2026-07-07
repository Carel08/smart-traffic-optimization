<#
Setup script for Smart Traffic Light Optimization on Windows PowerShell.

Usage:
    powershell -ExecutionPolicy Bypass -File setup_env.ps1
    powershell -ExecutionPolicy Bypass -File setup_env.ps1 trafficopt_test
#>

param(
    [string]$EnvName = "trafficopt"
)

$ErrorActionPreference = "Stop"

Write-Host "Smart Traffic Light Optimization - Windows setup" -ForegroundColor Cyan
Write-Host "Environment name: $EnvName" -ForegroundColor Cyan

# Check conda
try {
    conda --version | Out-Null
} catch {
    Write-Host "Conda was not found on PATH." -ForegroundColor Red
    Write-Host "Open Anaconda Prompt or PowerShell after running 'conda init powershell'." -ForegroundColor Yellow
    exit 1
}

# Create requirements.txt
$requirements = @"
numpy
pandas
scikit-learn
scipy
networkx
matplotlib
plotly
streamlit
pytest
"@

Set-Content -Path "requirements.txt" -Value $requirements -Encoding UTF8
Write-Host "requirements.txt created/updated." -ForegroundColor Green

# Check whether environment exists
$envList = conda env list | Out-String
if ($envList -match "^\s*$EnvName\s") {
    Write-Host "Conda environment '$EnvName' already exists. Reusing it." -ForegroundColor Yellow
} else {
    Write-Host "Creating conda environment '$EnvName' with Python 3.11..." -ForegroundColor Cyan
    conda create -n $EnvName python=3.11 -y
}

# Install requirements using conda run so activation is not required inside the script
Write-Host "Installing Python packages..." -ForegroundColor Cyan
conda run -n $EnvName python -m pip install --upgrade pip
conda run -n $EnvName python -m pip install -r requirements.txt

# Create output folders
New-Item -ItemType Directory -Force -Path "outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs/charts" | Out-Null
Write-Host "Output folders created." -ForegroundColor Green

# Validate imports
Write-Host "Validating imports..." -ForegroundColor Cyan
conda run -n $EnvName python -c "import numpy, pandas, sklearn, scipy, networkx, matplotlib, plotly, streamlit; print('All imports passed')"

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "Next commands:" -ForegroundColor Cyan
Write-Host "conda activate $EnvName"
Write-Host "pytest"
Write-Host "python main.py --num-intersections 4 --scenario normal --duration 30 --controller fixed_equal"
Write-Host "streamlit run streamlit_app.py"
