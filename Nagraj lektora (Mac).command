#!/bin/bash
cd "$(dirname "$0")" || exit 1

echo
echo "=========================================="
echo "  Postaw na milion – Sprecher aufnehmen"
echo "=========================================="
echo

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js ist nicht installiert."
  echo
  echo "Bitte einmalig hier herunterladen und installieren:"
  echo "  https://nodejs.org   (die linke, grüne Schaltfläche)"
  echo
  echo "Danach diese Datei erneut doppelklicken."
  echo
  read -r -p "Mit Enter schließen…" _
  exit 1
fi

if [ ! -f .env ]; then
  echo "Bitte den ElevenLabs-Schlüssel einfügen (Cmd+V) und Enter drücken."
  echo
  read -r -p "Schlüssel: " KEY
  if [ -z "$KEY" ]; then
    echo
    echo "Kein Schlüssel eingegeben – Abbruch."
    read -r -p "Mit Enter schließen…" _
    exit 1
  fi
  printf 'ELEVENLABS_API_KEY=%s\n' "$KEY" > .env
  chmod 600 .env
  echo
  echo "Gespeichert. Beim nächsten Mal wird nicht mehr gefragt."
  echo
fi

echo "Zuerst eine Hörprobe – fünf Sätze, kostet fast nichts."
echo
if ! node --env-file=.env tools/voice-build.mjs --limit 5; then
  echo
  echo "Da ist etwas schiefgegangen – die Meldung steht oben."
  echo "Häufigste Ursache: falscher oder abgelaufener Schlüssel."
  echo "Dann die Datei .env löschen und neu starten."
  read -r -p "Mit Enter schließen…" _
  exit 1
fi

echo
echo "=========================================="
echo "Die Aufnahmen liegen in .voice-cache/"
echo "Hör sie dir an. Passt die Stimme?"
echo
read -r -p "Alle übrigen Sätze aufnehmen? (j/n): " WEITER
if [ "$WEITER" != "j" ] && [ "$WEITER" != "J" ]; then
  echo
  echo "Abgebrochen. Nichts verloren – die fünf Sätze bleiben gespeichert."
  read -r -p "Mit Enter schließen…" _
  exit 0
fi

echo
echo "Das dauert 10 bis 20 Minuten. Abbrechen ist jederzeit erlaubt,"
echo "beim nächsten Start geht es weiter, wo es aufgehört hat."
echo
if ! node --env-file=.env tools/voice-build.mjs; then
  echo
  echo "Abgebrochen oder Fehler – der Fortschritt ist gespeichert."
  read -r -p "Mit Enter schließen…" _
  exit 1
fi

echo
echo "=========================================="
echo "Fertig. Die Datei liegt unter:"
echo "  assets/voice/pack.js"
echo
echo "Diese Datei bitte an Claude schicken."
echo "=========================================="
read -r -p "Mit Enter schließen…" _
