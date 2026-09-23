@echo off
title Telco Churn Analytics Dashboard
color 0B
echo.
echo  ==========================================
echo   Telco Customer Churn Analytics Dashboard
echo  ==========================================
echo.
echo  Starting server... please wait ~15 seconds
echo.

cd /d "%~dp0"
"C:\Users\goura\AppData\Roaming\Python\Python314\Scripts\streamlit.exe" run telco_churn_dashboard.py --server.port 8501

pause
