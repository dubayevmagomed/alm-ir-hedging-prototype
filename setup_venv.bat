@echo off

echo ===============================
echo Creating fresh virtual environment
echo ===============================

REM Remove old venv if exists
if exist .venv (
    echo Deleting old .venv...
    rmdir /s /q .venv
)

REM Create new venv using py launcher
py -m venv .venv

REM Activate venv
call .venv\Scripts\activate

echo ===============================
echo Upgrading pip
echo ===============================
py -m pip install --upgrade pip

echo ===============================
echo Installing dependencies
echo ===============================

py -m pip install -r requirements.txt

echo ===============================
echo Setup complete
echo ===============================

echo.
echo To activate environment later:
echo .venv\Scripts\activate

pause