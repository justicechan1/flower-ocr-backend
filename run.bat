@echo off
echo =======================================
echo   Flower OCR FastAPI 서버 실행 중...
echo =======================================
echo.

REM 가상환경 활성화
call venv\Scripts\activate

REM FastAPI 서버 실행
uvicorn main:app --reload --port 8000

REM uvicorn 종료 후 잠시 대기
pause
