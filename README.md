# Postaw na milion

Der polnische TV-Klassiker als Browser-Spiel — mit dem, was das Format ausmacht:
**Falltüren.** Das Geld liegt als Bündelstapel auf den Klappen unter den Antworten. Nach dem
Bestätigen öffnen sie sich **eine nach der anderen**: erst die leeren Felder, dann die mit
wachsendem Einsatz, die richtige zuletzt.

Spieloberfläche und Fragen sind polnisch, Code und Doku deutsch.
Kein Build-Schritt, keine Abhängigkeiten, kein Framework — nur HTML, CSS und ES-Module.

---

## Spielprinzip

1. Du bekommst **40 Bündel zu je 25 000 zł** — zusammen 1 000 000 zł.
2. Vor jeder Frage wählst du eine von **zwei Kategorien**. Die Frage fällt erst danach.
3. Unter jeder Antwort liegt eine Falltür. Du verteilst **alle** Bündel darauf: alles auf eine
   Klappe oder abgesichert auf mehrere.
4. **Eine Falltür muss leer bleiben** — auf alle gleichzeitig zu setzen ist nicht erlaubt. Das
   Spiel blockiert das letzte freie Feld automatisch.
5. Bestätigt wird mit **Zatwierdzam**. Kurze Stille, dann geht Klappe für Klappe auf — im
   Scheinwerfer, mit dem laufenden Betrag, der bei jedem Sturz mitzählt. Die Animation lässt
   sich mit *Pokaż wynik* oder `Enter` überspringen.
6. Läuft die Zeit ab, wird automatisch abgerechnet — Bündel, die noch in der Hand liegen, fallen
   mit.
7. Acht Runden. Im **Finale** bleiben nur zwei Falltüren, also alles auf eine Antwort.
   Bei 0 zł ist Schluss.

Zeitlimit pro Runde: 90 s zu Beginn, 50 s im Finale.

## Spielen

**Am schnellsten:** `dist/postaw-na-milion.html` herunterladen und doppelklicken. Die Datei
enthält alles — Markup, Styles, Logik, Fragen, Sounds — und läuft ohne Server und ohne
Internet.

**Aus den Quellen:** `index.html` nutzt ES-Module, die Browser über `file://` blockieren.
Hier braucht es also einen HTTP-Server:

```bash
python3 -m http.server 8080     # oder: npx http-server -p 8080
```

Dann `http://localhost:8080` öffnen.

**Einzeldatei neu bauen** — nach jeder Änderung an `src/` oder am CSS:

```bash
npm run build
```

Das Skript `tools/build-single.mjs` fügt Stylesheet und Module in `index.html` ein und
schreibt `dist/postaw-na-milion.html`. Es ist Bequemlichkeit, keine Voraussetzung: Bearbeitet
werden weiterhin die Dateien in `src/`.

## Vorlesestimme

Das Spiel liest Hasła, Fragen und Auflösungen vor. Zwei Stufen, beide offline:

**Ohne alles** nutzt es die Stimmen des Betriebssystems (`src/speech.js`). Kostet nichts,
braucht keine Dateien — aber die Qualität hängt am Gerät, und auf einem System ohne
polnisches Sprachpaket gibt es gar keine Stimme. Dann bleibt der Lektor-Knopf ausgeblendet.

**Mit Sprachpaket** klingt es nach Sendung. `tools/voice-build.mjs` nimmt alle Zeilen einmal
auf und legt sie als `assets/voice/pack.js` ab; danach ist nichts mehr nachzuladen.

### Den API-Schlüssel hinterlegen

Der Schlüssel gehört **nie ins Repository**. Zwei Wege:

```bash
# einmalig für einen Lauf
ELEVENLABS_API_KEY=sk_... node tools/voice-build.mjs

# oder dauerhaft in einer ignorierten .env (Vorlage: .env.example)
cp .env.example .env        # Schlüssel eintragen
node --env-file=.env tools/voice-build.mjs
```

`.env` steht in `.gitignore`. Unter Windows PowerShell entspricht der erste Weg
`$env:ELEVENLABS_API_KEY="sk_..."` vor dem Aufruf.

**Erst probehören, dann alles aufnehmen:**

```bash
node --env-file=.env tools/voice-build.mjs --limit 5
```

Das nimmt fünf Zeilen auf und kostet fast nichts. Passt die Stimme, lass den Lauf ohne
`--limit` durch — fertige Zeilen liegen in `.voice-cache/` und werden nicht doppelt bezahlt.

### Die drei Wege

| Weg | Kosten | Befehl |
| --- | --- | --- |
| ElevenLabs | einmalig ~12 500 Zeichen | `node --env-file=.env tools/voice-build.mjs` |
| Piper, lokal | gratis | `node tools/voice-build.mjs --engine piper --piper-model pl_PL-gosia-medium.onnx` |
| woanders aufgenommen | — | `--manifest kwestie.csv`, dann `--from ./nagrania` |

Alle drei enden bei derselben Datei. Credits fallen **nur beim Aufnehmen** an, nie beim Spielen.

## Tests

Die Spiellogik in `src/engine.js` ist frei von DOM-Zugriffen und wird direkt in Node getestet —
ohne Test-Framework, nur mit dem eingebauten Runner:

```bash
npm test
```

Abgedeckt sind unter anderem die Ein-Feld-muss-leer-bleiben-Regel, die Abrechnung der Runde,
der Zwei-Türen-Finalmodus und die Reproduzierbarkeit über den Spielcode.

## Projektstruktur

```
index.html                  Grundgerüst, alle drei Screens (Start / Spiel / Ende)
assets/css/style.css        komplettes Styling inkl. der 3D-Bühne mit den Falltüren
src/engine.js               Spiellogik: Bündel, Einsätze, Abrechnung (pure functions)
src/questions.js            Fragenkatalog
src/app.js                  Verbindung Logik ↔ DOM, Timer, Türöffnungs-Choreografie, Tastatur
src/audio.js                Sounds, komplett per Web Audio API erzeugt (keine Dateien)
src/confetti.js             Konfetti auf dem Endscreen
src/storage.js              gekapselter localStorage-Zugriff (Privatmodus wirft sonst)
tools/build-single.mjs      baut die Einzeldatei
tests/engine.test.js        Tests der Spiellogik
dist/postaw-na-milion.html  gebaute Einzeldatei (generiert)
```

### Wie die Bühne aufgebaut ist

Zwei Ebenen liegen auf demselben Spaltenraster übereinander:

* `.floor` — das Podest, per `rotateX(66deg)` in die Perspektive gekippt. Jede Falltür besteht
  aus zwei Flügeln, die an den Außenkanten scharniert sind. Innerhalb der gekippten Ebene zeigt
  die lokale Y-Achse in die Tiefe, deshalb klappt `rotateY(±104deg)` die Flügel nach unten weg.
* `.lanes` — Antwortpanels und Geldstapel, in normaler Bildschirmebene. Dadurch ist der Sturz
  eine simple `translateY`-Animation, ohne Kampf mit dem 3D-Koordinatensystem.

Darunter liegt die Grube (`--pit-h`), in der die Bündel verschwinden.

## Eigene Fragen ergänzen

Ein Eintrag in `src/questions.js` genügt:

```js
{
  id: 'geo-wisla',              // eindeutig
  category: 'Geografia',
  difficulty: 3,                // 1 = leicht … 5 = schwer
  text: 'Która rzeka jest najdłuższa w Polsce?',
  answers: [                    // genau vier — so viele Falltüren gibt es
    { text: 'Wisła', correct: true },   // genau eine richtige
    { text: 'Odra' },
    { text: 'Warta' },
    { text: 'Bug' },
  ],
  note: 'Wisła ma 1047 km długości.',   // wird nach dem Öffnen eingeblendet
}
```

* Genau **vier** Antworten, davon **genau eine** richtige.
* Die Reihenfolge wird im Spiel gemischt.
* Weil vor jeder Frage zwei Kategorien zur Wahl stehen, braucht jede Schwierigkeitsstufe
  Fragen aus **mehreren Kategorien** — sonst hätte die Auswahl nichts anzubieten. `npm test`
  rechnet nach, ob genug übrig bleibt, wenn frühere Runden schon Fragen verbraucht haben.
* Gute Fragen prüfen nicht das Gedächtnis, sondern den Reflex: der Tag auf der Venus ist länger
  als ihr Jahr, der Februar 2100 hat 28 Tage, der nächste Verwandte des Flusspferds ist der Wal.

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
| `1`, `2` | Kategorie wählen (auf dem Auswahlbildschirm) |
| `1`–`4` | wie viele Bündel auf einmal (25 000 / 50 000 / 100 000 / 250 000 zł) |
| `A`–`D` | Bündel auf diese Falltür legen |
| `Shift` + `A`–`D` | den ganzen Rest auf eine Falltür |
| `Backspace` | letzten Zug zurücknehmen |
| `Enter` | Runde bestätigen, Türöffnung überspringen, weiter |

## Deployment

Das Repository ist ohne Build-Schritt direkt hostbar. Der Workflow
`.github/workflows/pages.yml` veröffentlicht es auf GitHub Pages, sobald Pages in den
Repository-Einstellungen auf *GitHub Actions* gestellt ist.

> Hinweis: Für **private** Repositories setzt GitHub Pages einen kostenpflichtigen Plan voraus.
> In einem privaten Repo läuft das Spiel lokal — oder das Repo auf öffentlich stellen.

## Bildmarke

Der Schriftzug über dem Geldkoffer ist vollständig in CSS gebaut (`.mark` in `style.css`) — eine
eigene Nachempfindung im Stil einer TV-Titelkarte, kein Asset der Produktion. Ein einziger
Komponentenbaum bedient Kopfzeile und Startbildschirm; skaliert wird über `--mark`:

```css
.mark--small { --mark: 0.62; }   /* Kopfzeile */
.mark--hero  { --mark: 1.55; }   /* Startbildschirm */
```

## Lizenz

MIT, siehe [LICENSE](LICENSE). Fanprojekt: Das Spielformat ist an den Fernseh-Teleturniej
angelehnt, Fragen, Grafik und Code sind eigenständig und ohne Bezug zur Produktion entstanden.
