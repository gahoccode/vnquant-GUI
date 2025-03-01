@echo off
echo Starting VNQuant-GUI application...

:: Check if venv directory exists
if not exist "venv" (
    echo Virtual environment not found. Creating new virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    echo Installing required packages...
    pip install -e .
) else (
    echo Activating existing virtual environment...
    call venv\Scripts\activate
)

:: Run the Streamlit app
echo Starting Streamlit app...
streamlit run streamlit_app.py

:: Keep the window open if there's an error
if %ERRORLEVEL% NEQ 0 (
    echo An error occurred while running the application.
    pause
)
