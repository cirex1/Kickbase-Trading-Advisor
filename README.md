# Postaw na milion

Der polnische TV-Klassiker als Browser-Spiel — mit dem, was das Format ausmacht:
**Falltüren.** Das Geld liegt als Bündelstapel auf den Klappen unter den Antworten. Nach dem
Bestätigen öffnen sich die Türen unter den falschen Antworten und die Bündel stürzen in die Tiefe.

Spieloberfläche und Fragen sind polnisch, Code und Doku deutsch.
Kein Build-Schritt, keine Abhängigkeiten, kein Framework — nur HTML, CSS und ES-Module.

---

## Spielprinzip

1. Du bekommst **40 Bündel zu je 25 000 zł** — zusammen 1 000 000 zł.
2. Unter jeder Antwort liegt eine Falltür. Du verteilst **alle** Bündel darauf: alles auf eine
   Klappe oder abgesichert auf mehrere.
3. **Eine Falltür muss leer bleiben** — auf alle gleichzeitig zu setzen ist nicht erlaubt. Das
   Spiel blockiert das letzte freie Feld automatisch.
4. Bestätigt wird mit **Zatwierdzam**. Kurze Stille, dann klappen die Türen unter den falschen
   Antworten auf und das Geld fällt.
5. Läuft die Zeit ab, wird automatisch abgerechnet — Bündel, die noch in der Hand liegen, fallen
   mit.
6. Acht Runden. Im **Finale** bleiben nur zwei Falltüren, also alles auf eine Antwort.
   Bei 0 zł ist Schluss.

Zeitlimit pro Runde: 90 s zu Beginn, 50 s im Finale.

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

Abgedeckt sind unter anderem die Ein-Feld-muss-leer-bleiben-Regel, die Abrechnung der Runde,
der Zwei-Türen-Finalmodus und die Reproduzierbarkeit über den Spielcode.

## Projektstruktur

```
index.html              Grundgerüst, alle drei Screens (Start / Spiel / Ende)
assets/css/style.css    komplettes Styling inkl. der 3D-Bühne mit den Falltüren
src/engine.js           Spiellogik: Bündel, Einsätze, Abrechnung (pure functions)
src/questions.js        Fragenkatalog
src/app.js              Verbindung Logik ↔ DOM, Timer, Falltür-Choreografie, Tastatur
src/audio.js            Sounds, komplett per Web Audio API erzeugt (keine Dateien)
src/confetti.js         Konfetti auf dem Endscreen
tests/engine.test.js    Tests der Spiellogik
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
* Pro Schwierigkeitsgrad zieht eine Partie bis zu zwei Fragen — es sollten also mindestens zwei
  je Stufe vorhanden sein. `npm test` prüft all das.

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
| `1`–`4` | wie viele Bündel auf einmal (25 000 / 50 000 / 100 000 / 250 000 zł) |
| `A`–`D` | Bündel auf diese Falltür legen |
| `Shift` + `A`–`D` | den ganzen Rest auf eine Falltür |
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
