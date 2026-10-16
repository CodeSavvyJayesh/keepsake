@echo off
rem KeepSake command runner. Uses the project's own Python, so there is no need to activate .venv.
rem Usage:  ks <command> [folder or file]
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "PYTHONUTF8=1"
set "CMD=%~1"

if "%CMD%"=="" goto help
if /i "%CMD%"=="help" goto help
if /i "%CMD%"=="setup" goto setup

if not exist "%PY%" (
  echo The Python environment is missing. Run first:  ks setup
  exit /b 1
)

if /i "%CMD%"=="check" goto check
if /i "%CMD%"=="list" goto list
if /i "%CMD%"=="previews" goto previews
if /i "%CMD%"=="edge" goto edge
if /i "%CMD%"=="match" goto match
if /i "%CMD%"=="ingest" goto ingest
if /i "%CMD%"=="quality" goto quality
if /i "%CMD%"=="faces" goto faces
if /i "%CMD%"=="eyecheck" goto eyecheck
if /i "%CMD%"=="models" goto models
if /i "%CMD%"=="test" goto test
echo Unknown command: %CMD%
goto help

:setup
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%setup.ps1"
exit /b %errorlevel%

:check
"%PY%" "%ROOT%scripts\check_setup.py"
exit /b %errorlevel%

:list
"%PY%" "%ROOT%scripts\list_photos.py" %2
exit /b %errorlevel%

:previews
"%PY%" "%ROOT%scripts\extract_previews.py" %2
exit /b %errorlevel%

:edge
"%PY%" "%ROOT%scripts\make_edge_cases.py"
exit /b %errorlevel%

:match
"%PY%" "%ROOT%scripts\match_selected.py" %2
exit /b %errorlevel%

:ingest
"%PY%" "%ROOT%scripts\ingest.py" %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%

:quality
"%PY%" "%ROOT%scripts\quality.py" %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%

:faces
"%PY%" "%ROOT%scripts\faces.py" %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%

:eyecheck
"%PY%" "%ROOT%scripts\eyecheck.py" %2
exit /b %errorlevel%

:models
"%PY%" "%ROOT%scripts\models.py" %2 %3
exit /b %errorlevel%

:test
"%PY%" -m pytest --rootdir "%ROOT%." "%ROOT%tests"
exit /b %errorlevel%

:help
echo.
echo KeepSake commands (run inside C:\KeepSake):
echo   ks setup              install or repair everything
echo   ks check              check tools and this laptop
echo   ks list [folder]      count photos by camera, type and folder
echo   ks previews [folder]  pull previews out of RAW files and time them
echo   ks edge               create the photo folders and edge-case files
echo   ks match "folder"     match a wedding's selected photos to the originals
echo   ks ingest "folder"    read a whole shoot once: catalog, previews, thumbnails
echo   ks ingest "folder" --bench   speed test: best number of workers for this drive
echo   ks quality "folder"   measure sharpness and exposure of an ingested shoot
echo   ks faces "folder"     find faces; read eyes, smiles and head pose
echo   ks eyecheck "folder"  make the 50-face eye check page (Phase 2 exit test)
echo   ks models             download the AI model files (faces, eyes)
echo   ks test               run the automatic tests
echo.
echo Without a folder, list and previews use data\test_photos.
echo Put folder names with spaces in quotes:  ks list "E:\Sharma Wedding"
exit /b 0
