@echo off
echo ============================================================
echo   TASK 6 - Installation des dependances + Execution
echo ============================================================
echo.

echo [1/2] Installation des dependances Python...
pip install matplotlib seaborn numpy pandas scikit-learn librosa joblib kagglehub tensorflow
echo.

if %ERRORLEVEL% NEQ 0 (
    echo ERREUR lors de l'installation !
    echo Essai avec python -m pip...
    python -m pip install matplotlib seaborn numpy pandas scikit-learn librosa joblib kagglehub tensorflow
)

echo.
echo [2/2] Lancement du script principal...
echo ============================================================
cd /d "%~dp0"
python main.py

echo.
echo ============================================================
echo  Termine ! Appuyez sur une touche pour fermer...
echo ============================================================
pause

