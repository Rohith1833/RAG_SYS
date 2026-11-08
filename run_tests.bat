@echo off
title ULTRON - Test Runner
echo ===================================================
echo             ULTRON MASTER TEST RUNNER             
echo ===================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe!
    pause
    exit /b 1
)

echo [1/5] Running Tokenizer and Model Tests...
.\.venv\Scripts\python.exe tests\test_tokenizer_and_models.py
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Tokenizer and model tests failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/5] Running RAG Pipeline and Components Tests...
.\.venv\Scripts\python.exe tests\test_rag_pipeline_and_components.py
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] RAG pipeline tests failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [3/5] Running Advanced Features Tests...
.\.venv\Scripts\python.exe tests\test_advanced_features.py
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Advanced features tests failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [4/5] Running Flask API Endpoints Tests...
.\.venv\Scripts\python.exe tests\test_flask_api.py
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Flask API tests failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [5/5] Running Master Scenarios A through N...
.\.venv\Scripts\python.exe tests\test_scenarios_a_to_n.py
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Scenarios A through N failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===================================================
echo       ALL 5 TEST SUITES PASSED SUCCESSFULLY!       
echo ===================================================
echo.
pause

