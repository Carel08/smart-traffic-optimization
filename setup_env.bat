@echo off
set ENVNAME=%1
if "%ENVNAME%"=="" set ENVNAME=trafficopt

echo Smart Traffic Light Optimization - Windows setup
echo Environment name: %ENVNAME%

where conda >nul 2>nul
if errorlevel 1 (
    echo Conda was not found on PATH.
    echo Open Anaconda Prompt and run this file again.
    exit /b 1
)

(
echo numpy
echo pandas
echo scikit-learn
echo scipy
echo networkx
echo matplotlib
echo plotly
echo streamlit
echo pytest
) > requirements.txt

conda env list | findstr /R /C:"^%ENVNAME% " >nul
if errorlevel 1 (
    echo Creating conda environment %ENVNAME%...
    conda create -n %ENVNAME% python=3.11 -y
) else (
    echo Conda environment %ENVNAME% already exists. Reusing it.
)

echo Installing requirements...
conda run -n %ENVNAME% python -m pip install --upgrade pip
conda run -n %ENVNAME% python -m pip install -r requirements.txt

if not exist outputs mkdir outputs
if not exist outputs\charts mkdir outputs\charts

echo Validating imports...
conda run -n %ENVNAME% python -c "import numpy, pandas, sklearn, scipy, networkx, matplotlib, plotly, streamlit; print('All imports passed')"

echo.
echo Setup complete.
echo.
echo Next commands:
echo conda activate %ENVNAME%
echo pytest
echo python main.py --num-intersections 4 --scenario normal --duration 30 --controller fixed_equal
echo streamlit run streamlit_app.py
