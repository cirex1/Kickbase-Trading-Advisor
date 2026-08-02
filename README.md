# Postaw na milion

Der polnische TV-Klassiker als Browser-Spiel: **1 000 000 zł in Jetons, 10 Fragen, keine zweite Chance.**
Die Spieloberfläche und alle Fragen sind auf Polnisch — der Code und diese Doku auf Deutsch.

Kein Build-Schritt, keine Abhängigkeiten, kein Framework. Reines HTML, CSS und ES-Module.

---

## Spielprinzip

1. Du startest mit der vollen Summe in Jetons (Nominale von 1 000 bis 500 000 zł).
2. Zu jeder Frage verteilst du **die gesamte Summe** auf die Antwortfelder — alles auf ein Feld
   oder abgesichert auf mehrere.
3. Bestätigt wird mit **Zatwierdzam**. Danach bleibt nur, was auf den *richtigen* Feldern liegt.
   Der Rest ist weg.
4. Manche Fragen haben **mehrere richtige Antworten** — das Spiel warnt vor der Runde.
5. Läuft die Zeit ab, wird die Runde automatisch abgerechnet; nicht verteiltes Geld verfällt.
6. Bei 0 zł ist Schluss. Wer alle 10 Runden mit voller Summe übersteht, holt die Million.

Zeitlimit pro Runde: 90 s zu Beginn, 45 s in der letzten Runde.

## Starten

Das Spiel nutzt ES-Module, deshalb braucht es einen HTTP-Server — ein Doppelklick auf
`index.html` reicht nicht (Browser blockieren Module über `file://`).

```bash
# irgendein statischer Server, z. B.
python3 -m http.server 8080
# oder
npx http-server -p 8080
```

Dann `http://localhost:8080` öffnen.

## Tests

Die Spiellogik in `src/engine.js` ist frei von DOM-Zugriffen und wird direkt in Node getestet —
ohne Test-Framework, nur mit dem eingebauten Runner:

```bash
npm test
```

## Projektstruktur

```
index.html              Grundgerüst, alle drei Screens (Start / Spiel / Ende)
assets/css/style.css    komplettes Styling, Dark-Theme, responsiv
src/engine.js           Spiellogik: Zustand, Einsätze, Abrechnung (pure functions)
src/questions.js        Fragenkatalog
src/app.js              Verbindung Logik ↔ DOM, Timer, Tastatursteuerung
src/audio.js            Sounds, komplett per Web Audio API erzeugt (keine Dateien)
src/confetti.js         Konfetti auf dem Endscreen
tests/engine.test.js    Tests der Spiellogik
```

## Eigene Fragen ergänzen

Ein Eintrag in `src/questions.js` genügt — der Rest passt sich automatisch an:

```js
{
  id: 'geo-wisla',              // eindeutig
  category: 'Geografia',
  difficulty: 3,                // 1 = leicht … 5 = schwer
  text: 'Która rzeka jest najdłuższa w Polsce?',
  answers: [
    { text: 'Wisła', correct: true },
    { text: 'Odra' },
    { text: 'Warta' },
    { text: 'Bug' },
  ],
  note: 'Wisła ma 1047 km długości.',   // wird nach der Auflösung eingeblendet
}
```

* Erlaubt sind 3 bis 6 Antworten, davon **mindestens eine** richtige und mindestens eine falsche.
* Mehrere `correct: true` ergeben eine Mehrfachantwort-Frage.
* Die Reihenfolge der Antworten wird im Spiel gemischt.
* Pro Schwierigkeitsgrad werden zwei Fragen pro Partie gezogen — es sollten also mindestens
  zwei Fragen je Stufe vorhanden sein. `npm test` prüft das.

## Spielcode (Seed)

Jede Partie hat einen Code, der Fragenauswahl und -reihenfolge festlegt. Er steht auf dem
Endscreen und lässt sich auf dem Startbildschirm oder per URL wieder eingeben:

```
index.html?kod=DEMO
```

Gleicher Code → gleiche Partie. Praktisch, um sich mit jemandem zu messen.

## Tastatur

| Taste | Wirkung |
| --- | --- |
| `1`–`6` | Jeton-Nominal wählen |
| `A`–`E` | gewählten Jeton auf ein Antwortfeld legen |
| `Backspace` | letzten Zug zurücknehmen |
| `Enter` | Runde bestätigen bzw. weiter |

## Deployment

Das Repository ist ohne Build-Schritt direkt hostbar. Der Workflow
`.github/workflows/pages.yml` veröffentlicht es auf GitHub Pages, sobald Pages in den
Repository-Einstellungen auf *GitHub Actions* gestellt ist.

> Hinweis: Für **private** Repositories setzt GitHub Pages einen kostenpflichtigen Plan voraus.
> In einem privaten Repo läuft das Spiel lokal — oder das Repo auf öffentlich stellen.

## Lizenz

MIT, siehe [LICENSE](LICENSE). Das Spielformat ist an den Fernseh-Teleturniej angelehnt;
Fragen und Code sind eigenständig und ohne Bezug zur Produktion entstanden.
