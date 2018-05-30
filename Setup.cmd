@echo off
rem Setup script for Windows consoles.
rem Can be run over and over if you need to. If you see it change when
rem you do a git pull, you should run it again.

where pip3
if ERRORLEVEL 1 echo ERROR: pip3 not found. Did you install Python 3.6 or higher? && exit /b 1

echo.
echo ==========================================================================
echo Ensuring we have needed Python packages:
echo   python_speech_features - Mel-Frequency Cepstral Coefficients library.
echo     https://github.com/jameslyons/python_speech_features
echo   pylint - Linter for Python, useful in Visual Studio Code.
echo ==========================================================================
echo.
call pip3 install python_speech_features pylint
if ERRORLEVEL 1 echo ERROR: pip3 install failed with errorlevel %ERRORLEVEL% && exit /b 1

echo.
echo ==========================================================================
echo Complete!
echo ==========================================================================
echo.
