@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
title local-document-rag-pipeline (CLI)
cd /d "%~dp0"

:menu
cls
echo ================================
echo   LOCAL-DOCUMENT-RAG-PIPELINE
echo ================================
echo.
echo  Note: Provider claims Zero Data Retention (ZDR) -- not independently verified.
echo  Safe for: public and educational content only.
echo  Do NOT ingest: company docs, personal notes, project files, or any private data.
echo  Your data leaves this machine on every query. Proceed with awareness.
echo.
echo --------------------------------
echo.
echo  [1] Query (interactive)
echo  [2] Ingest docs folder
echo  [3] Re-ingest docs folder
echo  [4] View DB stats
echo  [5] Delete ingested file
echo  [6] Delete from delete_files.txt
echo  [7] Exit
echo.
set /p CHOICE="Select option: "

if "!CHOICE!"=="1" goto query
if "!CHOICE!"=="2" goto ingest
if "!CHOICE!"=="3" goto reingest
if "!CHOICE!"=="4" goto stats
if "!CHOICE!"=="5" goto delete
if "!CHOICE!"=="6" goto deletefromfile
if "!CHOICE!"=="7" exit /b
goto menu

:query
cls
echo.
echo  Loading environment and models, please wait...
echo.
.venv\Scripts\python main.py --query
pause
goto menu

:ingest
cls
set /p FOLDER="Enter folder path to ingest: "
if "!FOLDER!"=="" (
    echo.
    echo  No path entered. Press any key to return to menu.
    pause >nul
    goto menu
)
echo.
echo  Loading environment and models, please wait...
echo  (This may take a few seconds on first run)
echo.
.venv\Scripts\python main.py --ingest "!FOLDER!"
pause
goto menu

:reingest
cls
set /p FOLDER="Enter folder path to re-ingest: "
if "!FOLDER!"=="" (
    echo.
    echo  No path entered. Press any key to return to menu.
    pause >nul
    goto menu
)
echo.
echo  Loading environment and models, please wait...
echo  (This may take a few seconds on first run)
echo.
.venv\Scripts\python main.py --reingest "!FOLDER!"
pause
goto menu

:delete
cls
echo.
echo  Loading ingested files, please wait...
echo.
.venv\Scripts\python main.py --delete
pause
goto menu

:deletefromfile
cls
echo.
echo  Reading delete_files.txt, please wait...
echo.
.venv\Scripts\python main.py --delete-from delete_files.txt
pause
goto menu

:stats
cls
echo.
echo  Loading stats, please wait...
echo.
.venv\Scripts\python main.py --stats
pause
goto menu