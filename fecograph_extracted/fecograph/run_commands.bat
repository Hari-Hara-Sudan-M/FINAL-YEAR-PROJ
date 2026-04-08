@echo off
REM ============================================================================
REM SCAFFOLD Zero-Shot Training and Visualization Commands
REM ============================================================================

echo.
echo ========================================================================
echo SCAFFOLD ZERO-SHOT NOVELTY DETECTION - QUICK COMMANDS
echo ========================================================================
echo.

:menu
echo Select an option:
echo.
echo [1] Run Training (SCAFFOLD Zero-Shot)
echo [2] Generate Visualizations (After Training)
echo [3] Run Both (Training + Visualizations)
echo [4] View Results Directory
echo [5] Exit
echo.
set /p choice="Enter choice (1-5): "

if "%choice%"=="1" goto train
if "%choice%"=="2" goto visualize
if "%choice%"=="3" goto both
if "%choice%"=="4" goto view_results
if "%choice%"=="5" goto end
echo Invalid choice. Please try again.
echo.
goto menu

:train
echo.
echo ========================================================================
echo RUNNING TRAINING...
echo ========================================================================
echo.
call "D:\Final yr project\FRCOGRAPH\feco_env\Scripts\activate.bat"
cd "D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph"
python run_scaffold_zeroshot.py
echo.
echo Training complete! Results saved to: results/scaffold_zeroshot/
echo.
pause
goto menu

:visualize
echo.
echo ========================================================================
echo GENERATING VISUALIZATIONS...
echo ========================================================================
echo.
call "D:\Final yr project\FRCOGRAPH\feco_env\Scripts\activate.bat"
cd "D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph\visualizations"
python plot_results.py
echo.
echo Visualizations saved to: visualizations/output/
echo.
echo Opening output folder...
start "" "D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph\visualizations\output"
echo.
pause
goto menu

:both
echo.
echo ========================================================================
echo RUNNING TRAINING + VISUALIZATIONS...
echo ========================================================================
echo.
call "D:\Final yr project\FRCOGRAPH\feco_env\Scripts\activate.bat"
cd "D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph"
echo [1/2] Running training...
python run_scaffold_zeroshot.py
echo.
echo [2/2] Generating visualizations...
cd visualizations
python plot_results.py
echo.
echo Complete! Opening results...
start "" "D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph\visualizations\output"
echo.
pause
goto menu

:view_results
echo.
echo Opening results directories...
start "" "D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph\results\scaffold_zeroshot"
start "" "D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph\visualizations\output"
echo.
pause
goto menu

:end
echo.
echo Exiting...
echo.
exit

REM ============================================================================
REM MANUAL COMMANDS (Copy-paste if needed)
REM ============================================================================
REM
REM 1. ACTIVATE VIRTUAL ENVIRONMENT:
REM    D:\Final yr project\FRCOGRAPH\feco_env\Scripts\activate
REM
REM 2. RUN TRAINING:
REM    cd "D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph"
REM    python run_scaffold_zeroshot.py
REM
REM 3. GENERATE VISUALIZATIONS:
REM    cd "D:\Final yr project\FRCOGRAPH\fecograph 1.0.2\fecograph\visualizations"
REM    python plot_results.py
REM
REM 4. VIEW RESULTS:
REM    Training results: results/scaffold_zeroshot/scaffold_zeroshot_results.json
REM    t-SNE plot: results/scaffold_zeroshot/scaffold_zeroshot_tsne.png
REM    Visualizations: visualizations/output/*.png
REM
REM ============================================================================
