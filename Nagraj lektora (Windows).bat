@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo ==========================================
echo   Postaw na milion - nagrywanie lektora
echo ==========================================
echo.

where node >nul 2>nul
if errorlevel 1 (
  echo Node.js ist nicht installiert.
  echo.
  echo Bitte einmalig hier herunterladen und installieren:
  echo   https://nodejs.org   ^(die linke, gruene Schaltflaeche^)
  echo.
  echo Danach diese Datei erneut doppelklicken.
  echo.
  pause
  exit /b 1
)

if not exist ".env" (
  echo Bitte den ElevenLabs-Schluessel einfuegen.
  echo ^(Rechtsklick ins Fenster fuegt die Zwischenablage ein.^)
  echo.
  set /p KEY=Schluessel:
  if "!KEY!"=="" (
    echo.
    echo Kein Schluessel eingegeben - Abbruch.
    pause
    exit /b 1
  )
  > .env echo ELEVENLABS_API_KEY=!KEY!
  echo.
  echo Gespeichert. Beim naechsten Mal wird nicht mehr gefragt.
  echo.
)

echo Zuerst eine Hoerprobe - fuenf Saetze, kostet fast nichts.
echo.
node --env-file=.env tools/voice-build.mjs --limit 5
if errorlevel 1 goto fehler

echo.
echo ==========================================
echo Die Aufnahmen liegen in .voice-cache\
echo Hoer sie dir an. Passt die Stimme?
echo.
set /p WEITER=Alle 617 Saetze aufnehmen? (j/n):
if /i not "!WEITER!"=="j" (
  echo.
  echo Abgebrochen. Nichts verloren - die fuenf Saetze bleiben gespeichert.
  pause
  exit /b 0
)

echo.
echo Das dauert 10 bis 20 Minuten. Abbrechen ist jederzeit erlaubt,
echo beim naechsten Start geht es weiter, wo es aufgehoert hat.
echo.
node --env-file=.env tools/voice-build.mjs
if errorlevel 1 goto fehler

echo.
echo ==========================================
echo Fertig. Die Datei liegt unter:
echo   assets\voice\pack.js
echo.
echo Diese Datei bitte an Claude schicken.
echo ==========================================
pause
exit /b 0

:fehler
echo.
echo Da ist etwas schiefgegangen - die Meldung steht oben.
echo Haeufigste Ursache: falscher oder abgelaufener Schluessel.
echo Dann die Datei .env loeschen und neu starten.
pause
exit /b 1
