/**
 * Składa nagrywarkę: jeden plik HTML, w którym siedzi cała lista kwestii.
 *
 *   node tools/build-studio.mjs      → dist/nagrywarka.html
 *
 * Po co osobne narzędzie, skoro `voice-build.mjs` już umie zbudować pakiet?
 * Bo tamto wymaga Node.js i wiersza poleceń, a to jest strona do dwukliku.
 * Kto chce nagrać lektora własnym głosem, nie powinien niczego instalować.
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { GROUPS, MOOD, collectLines } from './lines.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

const lines = await collectLines();
const payload = Object.fromEntries(
  [...lines].map(([key, line]) => [key, { text: line.text, mood: line.mood, group: line.group }]),
);
const hints = Object.fromEntries(Object.entries(MOOD).map(([k, v]) => [k, v.hint ?? '']));

const script = await readFile(join(ROOT, 'tools/studio.js'), 'utf8');

const perGroup = GROUPS.map((g) => {
  const n = Object.values(payload).filter((l) => l.group === g.id).length;
  return `${g.name} ${n}`;
}).join(' · ');
console.log(`kwestii: ${Object.keys(payload).length} — ${perGroup}`);

const html = `<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nagrywarka — Postaw na milion</title>
<style>
:root {
  --bg: #05040f;
  --panel: rgba(20, 14, 52, 0.72);
  --line: rgba(255, 255, 255, 0.12);
  --ink: #f4f1ff;
  --dim: #a89fd0;
  --gold: #ffc247;
  --good: #34e2a0;
  --bad: #ff5f7a;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  padding: clamp(0.8rem, 3vw, 2rem);
  background:
    radial-gradient(120% 70% at 50% -5%, #3a1b6b 0%, transparent 58%),
    linear-gradient(180deg, #05040f, #0f0926 60%, #05040f);
  color: var(--ink);
  font: 16px/1.55 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.wrap { max-width: 880px; margin: 0 auto; display: grid; gap: 1rem; }
h1 { margin: 0; font-size: clamp(1.3rem, 1rem + 1.6vw, 2rem); }
.lead { margin: 0; color: var(--dim); max-width: 62ch; }
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: clamp(0.9rem, 2.5vw, 1.4rem);
  backdrop-filter: blur(14px);
}
.tabs { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.tab {
  flex: 1 1 9rem;
  display: grid;
  gap: 0.1rem;
  padding: 0.5rem 0.8rem;
  font: inherit;
  color: var(--ink);
  text-align: left;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--line);
  border-radius: 12px;
  cursor: pointer;
}
.tab.is-active { border-color: var(--gold); background: rgba(255, 194, 71, 0.14); }
.tab span { font-size: 0.78rem; color: var(--dim); font-variant-numeric: tabular-nums; }
.head { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }
.head b { font-variant-numeric: tabular-nums; }
.head span { color: var(--dim); font-size: 0.85rem; }
.track { height: 6px; border-radius: 3px; background: rgba(255, 255, 255, 0.1); overflow: hidden; margin: 0.6rem 0 1rem; }
.track i { display: block; height: 100%; background: linear-gradient(90deg, #ffdf8c, var(--gold)); transition: width 0.25s ease; }
.slot { text-align: center; padding: 1.4rem 0.6rem; border: 1px dashed var(--line); border-radius: 14px; }
.slot.is-done { border-color: rgba(52, 226, 160, 0.5); background: rgba(52, 226, 160, 0.06); }
.slot__mood { color: var(--gold); font-size: 0.7rem; letter-spacing: 0.22em; text-transform: uppercase; }
.slot__text { margin: 0.5rem auto; max-width: 32ch; font-size: clamp(1.2rem, 1rem + 1.4vw, 1.9rem); font-weight: 700; line-height: 1.25; text-wrap: balance; }
.slot__key { color: rgba(168, 159, 208, 0.5); font-size: 0.7rem; font-family: ui-monospace, monospace; }
.slot__state { color: var(--dim); font-size: 0.8rem; }
.level { height: 4px; border-radius: 2px; background: rgba(255, 255, 255, 0.08); overflow: hidden; margin-top: 1rem; }
.level i { display: block; height: 100%; width: 0; background: var(--good); transition: width 0.08s linear; }
.row { display: flex; gap: 0.5rem; flex-wrap: wrap; justify-content: center; margin-top: 1rem; }
button, .file {
  font: inherit;
  font-weight: 600;
  color: var(--ink);
  background: rgba(255, 255, 255, 0.07);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.55rem 1.1rem;
  cursor: pointer;
}
button:hover:not(:disabled), .file:hover { background: rgba(255, 255, 255, 0.14); }
button:disabled { opacity: 0.35; cursor: not-allowed; }
.btn--rec { background: linear-gradient(180deg, #ffdf8c, var(--gold) 45%, #ff9d1e); color: #2a1c00; border-color: rgba(255, 255, 255, 0.35); min-width: 11rem; }
.btn--rec.is-live { background: linear-gradient(180deg, #ff9db0, var(--bad)); color: #2a0008; animation: blink 1s ease-in-out infinite alternate; }
@keyframes blink { to { box-shadow: 0 0 26px rgba(255, 95, 122, 0.7); } }
.status { min-height: 1.4rem; margin: 0.9rem 0 0; text-align: center; font-size: 0.9rem; color: var(--dim); }
.status.is-good { color: var(--good); }
.status.is-bad { color: var(--bad); }
.status.is-live { color: var(--gold); }
.file { display: inline-block; }
.file input { display: none; }
.foot { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; justify-content: space-between; }
.keys { margin: 0; padding: 0; list-style: none; display: flex; gap: 0.9rem; flex-wrap: wrap; color: var(--dim); font-size: 0.78rem; }
kbd { border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 5px; padding: 0 0.35rem; background: rgba(255, 255, 255, 0.08); }
details { color: var(--dim); font-size: 0.88rem; }
summary { cursor: pointer; color: var(--ink); font-weight: 600; }
details p, details ol { max-width: 66ch; }
</style>
</head>
<body>
<div class="wrap">
  <div>
    <h1>Sprecher aufnehmen</h1>
    <p class="lead">
      Jeder Satz wird einmal gesprochen. Was aufgenommen ist, hört man später im Spiel;
      was fehlt, liest weiterhin die Computerstimme vor — du kannst also jederzeit
      aufhören und später weitermachen. Nichts verlässt diesen Rechner.
    </p>
  </div>

  <div class="panel">
    <div class="tabs" id="tabs"></div>
  </div>

  <div class="panel">
    <div class="head">
      <b id="counter">—</b>
      <span id="done"></span>
    </div>
    <div class="track"><i id="bar"></i></div>

    <div class="slot" id="slot">
      <div class="slot__mood" id="mood"></div>
      <p class="slot__text" id="text"></p>
      <div class="slot__key" id="key"></div>
      <div class="slot__state" id="state"></div>
    </div>

    <div class="level"><i id="level"></i></div>

    <div class="row">
      <button class="btn--rec" id="rec" type="button">Aufnehmen</button>
      <button id="play" type="button">Anhören</button>
      <button id="again" type="button">Nochmal</button>
    </div>
    <div class="row">
      <button id="prev" type="button">← zurück</button>
      <button id="next" type="button">weiter →</button>
      <label class="file"><input type="checkbox" id="auto" checked style="display:inline"> automatisch weiter</label>
    </div>

    <p class="status" id="status"></p>

    <ul class="keys">
      <li><kbd>Leertaste</kbd> aufnehmen / stoppen</li>
      <li><kbd>P</kbd> anhören</li>
      <li><kbd>R</kbd> nochmal</li>
      <li><kbd>Enter</kbd> weiter</li>
    </ul>
  </div>

  <div class="panel foot">
    <span id="total"></span>
    <div class="row" style="margin:0">
      <button id="save" type="button">pack.js speichern</button>
      <label class="file">pack.js laden<input type="file" id="load-pack" accept=".js"></label>
      <label class="file">Dateien einlesen<input type="file" id="load-files" accept="audio/*" multiple></label>
      <button id="wipe" type="button">alles löschen</button>
    </div>
  </div>

  <div class="panel">
    <details>
      <summary>Wie das ins Spiel kommt</summary>
      <ol>
        <li>Aufnehmen, so viel du magst — der Fortschritt bleibt gespeichert.</li>
        <li>„pack.js speichern" drücken. Die Datei landet im Download-Ordner.</li>
        <li>Sie nach <code>dist/assets/voice/pack.js</code> legen, direkt neben das Spiel.</li>
        <li>Spiel öffnen → „Zasady". Dort steht dann, wie viele Sätze aus Aufnahmen kommen.</li>
      </ol>
      <p>
        Kein Mikrofon, oder der Browser lässt nicht zu? Nimm mit einem beliebigen Programm auf
        (Sprachmemo, Audacity, Windows-Sprachrekorder), benenne jede Datei nach dem grauen
        Schlüssel unter dem Satz — Doppelpunkte werden zu <code>__</code>, also
        <code>geo-wawel__tresc.mp3</code> — und lies sie über „Dateien einlesen" ein.
      </p>
    </details>
  </div>
</div>

<script>
window.LINES = ${JSON.stringify(payload)};
window.GROUPS = ${JSON.stringify(GROUPS)};
window.MOOD_HINTS = ${JSON.stringify(hints)};
</script>
<script>
${script}
</script>
</body>
</html>
`;

await mkdir(join(ROOT, 'dist'), { recursive: true });
await writeFile(join(ROOT, 'dist/nagrywarka.html'), html);
console.log(`dist/nagrywarka.html — ${(html.length / 1024).toFixed(0)} kB`);
