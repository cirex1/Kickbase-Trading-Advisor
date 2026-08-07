/**
 * Nagrywarka — logika strony do nagrywania lektora własnym głosem.
 *
 * Plik jest wstrzykiwany do `dist/nagrywarka.html` przez `build-studio.mjs`
 * razem ze spisem kwestii (`window.LINES`). Trzymamy go osobno, bo edytuje
 * się go jak zwykły JavaScript, a nie jak łańcuch znaków w generatorze.
 *
 * Trzy rzeczy, na których stoi ta strona:
 *
 * 1. Nagrania idą do IndexedDB, nie do localStorage. Blob w localStorage
 *    trzeba by zamieniać na tekst, a limit pięciu megabajtów skończyłby się
 *    po kilkudziesięciu kwestiach.
 * 2. Nic nie wychodzi na zewnątrz. Strona nie ma ani jednego zapytania do
 *    sieci — działa z dwukliku, także w samolocie.
 * 3. Mikrofon bywa niedostępny (Safari przez file://, odmowa uprawnień,
 *    brak sprzętu). Dlatego druga droga — wczytanie gotowych plików
 *    z dysku — jest równorzędna, a nie awaryjna.
 */

const LINES = window.LINES;
const GROUPS = window.GROUPS;
const MOOD_HINTS = window.MOOD_HINTS;

const $ = (id) => document.getElementById(id);
const clips = new Map(); // klucz → { blob, type, hash }

/**
 * Odcisk treści zdania (FNV-1a).
 *
 * Pytania w grze będą się jeszcze zmieniać, a klucz nagrania zostaje ten sam.
 * Bez odcisku nagranie starej wersji zdania siedziałoby w pakiecie w nieskończoność
 * i lektor czytałby coś innego, niż stoi na ekranie. Z odciskiem widać, które
 * kwestie trzeba powtórzyć — i to jedyny sposób, żeby to w ogóle zauważyć.
 */
function fingerprint(text) {
  let h = 2166136261;
  const s = String(text);
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(36);
}

/** Nagranie jest, ale zdanie zmieniło treść od czasu nagrania. */
function isStale(key) {
  const clip = clips.get(key);
  return Boolean(clip && clip.hash && clip.hash !== fingerprint(LINES[key]?.text ?? ''));
}

const staleKeys = () => Object.keys(LINES).filter(isStale);

let order = [];
let at = 0;
let group = GROUPS[0].id;
let stream = null;
let recorder = null;
let recording = false;
let db = null;
let meter = null;

/* ------------------------------------------------------------------ *
 * Trwałość — IndexedDB
 * ------------------------------------------------------------------ */

function openDb() {
  return new Promise((resolve) => {
    const request = indexedDB.open('pnm-nagrywarka', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('clips');
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(null);
  });
}

function put(key, value) {
  if (!db) return;
  const tx = db.transaction('clips', 'readwrite');
  tx.objectStore('clips').put(value, key);
}

function drop(key) {
  if (!db) return;
  const tx = db.transaction('clips', 'readwrite');
  tx.objectStore('clips').delete(key);
}

function loadAll() {
  return new Promise((resolve) => {
    if (!db) return resolve();
    const store = db.transaction('clips', 'readonly').objectStore('clips');
    const keys = store.getAllKeys();
    const values = store.getAll();
    values.onsuccess = () => {
      keys.result.forEach((key, i) => clips.set(key, values.result[i]));
      resolve();
    };
    values.onerror = () => resolve();
  });
}

/* ------------------------------------------------------------------ *
 * Lista kwestii
 * ------------------------------------------------------------------ */

function rebuild() {
  order = Object.keys(LINES).filter((key) => LINES[key].group === group);
  at = Math.min(at, Math.max(0, order.length - 1));
}

/** Pierwsza kwestia bez nagrania — po wczytaniu postępu zaczynamy właśnie tam. */
function jumpToGap() {
  const gap = order.findIndex((key) => !clips.has(key));
  at = gap >= 0 ? gap : 0;
}

const currentKey = () => order[at];

/* ------------------------------------------------------------------ *
 * Rysowanie
 * ------------------------------------------------------------------ */

function renderTabs() {
  $('tabs').innerHTML = '';
  for (const g of GROUPS) {
    const keys = Object.keys(LINES).filter((k) => LINES[k].group === g.id);
    const done = keys.filter((k) => clips.has(k)).length;
    const stale = keys.filter(isStale).length;
    const tab = document.createElement('button');
    tab.className = `tab${g.id === group ? ' is-active' : ''}`;
    tab.type = 'button';
    tab.innerHTML = `<b></b><span>${done} / ${keys.length}${stale ? ` · ${stale} veraltet` : ''}</span>`;
    tab.querySelector('b').textContent = g.name;
    tab.title = g.note;
    tab.onclick = () => {
      group = g.id;
      rebuild();
      jumpToGap();
      render();
    };
    $('tabs').append(tab);
  }
}

function render() {
  renderTabs();
  const key = currentKey();
  const line = LINES[key];
  const done = order.filter((k) => clips.has(k)).length;

  $('counter').textContent = order.length ? `${at + 1} / ${order.length}` : '—';
  $('done').textContent = `${done} aufgenommen`;
  $('bar').style.width = order.length ? `${(done / order.length) * 100}%` : '0';

  $('text').textContent = line ? line.text : 'Nichts zu tun.';
  $('mood').textContent = line ? MOOD_HINTS[line.mood] ?? '' : '';
  $('key').textContent = key ? fileFor(key) : '';
  $('slot').classList.toggle('is-done', Boolean(key && clips.has(key)) && !isStale(key));
  $('slot').classList.toggle('is-stale', Boolean(key && isStale(key)));
  $('state').textContent = !key || !clips.has(key)
    ? 'noch nichts'
    : isStale(key)
      ? 'aufgenommen — aber der Satz hat sich seitdem geändert'
      : 'aufgenommen';

  $('play').disabled = !key || !clips.has(key);
  $('again').disabled = !key || !clips.has(key);
  $('prev').disabled = at <= 0;
  $('next').disabled = at >= order.length - 1;

  const total = Object.keys(LINES).length;
  const stale = staleKeys().length;
  $('total').textContent =
    `${clips.size} von ${total} Sätzen im Kasten` + (stale ? ` · ${stale} veraltet` : '');
  $('save').disabled = clips.size === 0;
}

function say(message, kind = '') {
  $('status').textContent = message;
  $('status').className = `status ${kind}`;
}

/* ------------------------------------------------------------------ *
 * Mikrofon
 * ------------------------------------------------------------------ */

async function ensureMic() {
  if (stream) return true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (error) {
    // Najczęstsze powody: odmowa uprawnień, brak mikrofonu, Safari przez file://.
    say(
      `Mikrofon nicht verfügbar (${error.name}). ` +
        'Erlaube den Zugriff in der Adressleiste, oder nimm mit einem anderen ' +
        'Programm auf und lies die Dateien unten ein.',
      'is-bad',
    );
    return false;
  }
  startMeter();
  return true;
}

/** Wskaźnik poziomu — jedyny sposób, żeby od razu było widać, że mikrofon żyje. */
function startMeter() {
  try {
    const audio = new (window.AudioContext || window.webkitAudioContext)();
    const source = audio.createMediaStreamSource(stream);
    const analyser = audio.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    const buffer = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
      analyser.getByteTimeDomainData(buffer);
      let peak = 0;
      for (const v of buffer) peak = Math.max(peak, Math.abs(v - 128));
      $('level').style.width = `${Math.min(100, (peak / 90) * 100)}%`;
      meter = requestAnimationFrame(tick);
    };
    tick();
  } catch {
    /* wskaźnik to wygoda, nie warunek */
  }
}

function pickMime() {
  const types = [
    'audio/webm;codecs=opus',
    'audio/ogg;codecs=opus',
    'audio/mp4',
    'audio/webm',
    '',
  ];
  return types.find((t) => !t || MediaRecorder.isTypeSupported(t));
}

async function toggleRecord() {
  if (recording) return stopRecord();
  const key = currentKey();
  if (!key) return;
  if (!(await ensureMic())) return;

  const chunks = [];
  const mime = pickMime();
  recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  recorder.ondataavailable = (event) => event.data.size && chunks.push(event.data);
  recorder.onstop = async () => {
    recording = false;
    $('rec').classList.remove('is-live');
    $('rec').textContent = 'Aufnehmen';
    const type = recorder.mimeType || mime || 'audio/webm';
    const blob = new Blob(chunks, { type });
    if (blob.size < 400) {
      say('Zu kurz oder still — nichts gespeichert. Nochmal.', 'is-bad');
      render();
      return;
    }
    const clip = { blob, type, hash: fingerprint(LINES[key].text) };
    clips.set(key, clip);
    put(key, clip);
    say('Gespeichert.', 'is-good');
    if ($('auto').checked && at < order.length - 1) at++;
    render();
  };
  recorder.start();
  recording = true;
  $('rec').classList.add('is-live');
  $('rec').textContent = 'Stopp';
  say('Läuft — sprich jetzt.', 'is-live');
}

function stopRecord() {
  if (recorder && recording) recorder.stop();
}

function playCurrent() {
  const clip = clips.get(currentKey());
  if (!clip) return;
  const audio = new Audio(URL.createObjectURL(clip.blob));
  audio.play().catch(() => say('Abspielen ging nicht.', 'is-bad'));
}

function redo() {
  const key = currentKey();
  clips.delete(key);
  drop(key);
  render();
  say('Gelöscht — bereit für einen neuen Versuch.');
}

/* ------------------------------------------------------------------ *
 * Wejście i wyjście
 * ------------------------------------------------------------------ */

/** Nazwa pliku dla kwestii — dwukropek nie wszędzie jest legalny w nazwie. */
const fileFor = (key) => key.replace(/:/g, '__');

/**
 * Spis kwestii do pobrania.
 *
 * Kto woli nagrać gdzie indziej — u siebie w programie, w ElevenLabs, kimkolwiek —
 * dostaje tu tekst i nazwę pliku obok siebie. To wszystko, czego trzeba, żeby
 * wrócić z gotowymi nagraniami i wczytać je hurtem.
 */
function saveList() {
  const rows = [['plik', 'ton', 'tekst']];
  for (const key of order) {
    rows.push([`${fileFor(key)}.mp3`, MOOD_HINTS[LINES[key].mood] ?? '', LINES[key].text]);
  }
  const csv = rows
    .map((row) => row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(';'))
    .join('\r\n');
  // BOM — bez niego Excel zjada polskie znaki
  download(new Blob([`﻿${csv}`], { type: 'text/csv' }), `kwestie-${group}.csv`);
  say(`Liste mit ${order.length} Sätzen gespeichert.`, 'is-good');
}

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

const toBase64 = (blob) =>
  new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',')[1] ?? '');
    reader.onerror = () => resolve('');
    reader.readAsDataURL(blob);
  });

async function savePack() {
  say('Packe … das dauert einen Moment.');
  const lines = {};
  for (const [key, clip] of clips) {
    const data = await toBase64(clip.blob);
    if (data) lines[key] = { d: data, t: clip.type, h: clip.hash };
  }
  const file =
    '/** Pakiet lektora — nagrany własnym głosem w nagrywarce. */\n' +
    `window.PNM_VOICE = ${JSON.stringify({ format: 'audio/webm', lines })};\n`;
  download(new Blob([file], { type: 'text/javascript' }), 'pack.js');
  const stale = staleKeys().length;
  say(
    `pack.js gespeichert (${Object.keys(lines).length} Sätze). ` +
      'Die Datei nach dist/assets/voice/ legen.' +
      (stale ? ` Achtung: ${stale} davon sind veraltet.` : ''),
    stale ? 'is-bad' : 'is-good',
  );
}

/** Wczytanie istniejącego pakietu — żeby dało się dograć resztę innego dnia. */
async function importPack(file) {
  const text = await file.text();
  const match = text.match(/window\.PNM_VOICE\s*=\s*([\s\S]*);?\s*$/);
  if (!match) return say('Das sieht nicht nach einer pack.js aus.', 'is-bad');
  let parsed;
  try {
    parsed = JSON.parse(match[1].replace(/;\s*$/, ''));
  } catch {
    return say('Die Datei ließ sich nicht lesen.', 'is-bad');
  }
  let added = 0;
  for (const [key, value] of Object.entries(parsed.lines ?? {})) {
    const data = typeof value === 'string' ? value : value.d;
    const type = typeof value === 'string' ? parsed.format : value.t;
    const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
    const clip = { blob: new Blob([bytes], { type }), type, hash: value.h };
    clips.set(key, clip);
    put(key, clip);
    added++;
  }
  jumpToGap();
  render();
  say(`${added} Sätze übernommen.`, 'is-good');
}

/**
 * Wczytanie nagrań zrobionych gdzie indziej. Nazwa pliku jest kluczem —
 * dokładnie ta, którą wypisuje `--manifest`, więc dyktafon w telefonie
 * albo Audacity też są dobrą drogą.
 */
function importFiles(files) {
  let taken = 0;
  const unknown = [];
  for (const file of files) {
    const key = file.name
      .replace(/\.[^.]+$/, '') // rozszerzenie
      .replace(/\s*\(\d+\)$/, '') // „(1)” dopisane przez przeglądarkę przy powtórce
      .trim()
      .replace(/__/g, ':');
    if (!LINES[key]) {
      unknown.push(file.name);
      continue;
    }
    const clip = {
      blob: file,
      type: file.type || 'audio/mpeg',
      hash: fingerprint(LINES[key].text),
    };
    clips.set(key, clip);
    put(key, clip);
    taken++;
  }
  jumpToGap();
  render();
  say(
    `${taken} Dateien übernommen` +
      (unknown.length
        ? `. ${unknown.length} übersprungen — der Name passt zu keinem Satz: ${unknown
            .slice(0, 3)
            .join(', ')}${unknown.length > 3 ? ' …' : ''}`
        : '.'),
    taken ? 'is-good' : 'is-bad',
  );
}

function wipe() {
  if (!confirm('Wirklich alle Aufnahmen löschen?')) return;
  clips.clear();
  if (db) db.transaction('clips', 'readwrite').objectStore('clips').clear();
  at = 0;
  render();
  say('Alles gelöscht.');
}

/* ------------------------------------------------------------------ *
 * Sterowanie
 * ------------------------------------------------------------------ */

$('rec').onclick = toggleRecord;
$('play').onclick = playCurrent;
$('again').onclick = redo;
$('prev').onclick = () => {
  if (at > 0) at--;
  render();
};
$('next').onclick = () => {
  if (at < order.length - 1) at++;
  render();
};
$('save').onclick = savePack;
$('list').onclick = saveList;
$('wipe').onclick = wipe;
$('load-pack').onchange = (e) => e.target.files[0] && importPack(e.target.files[0]);
$('load-files').onchange = (e) => e.target.files.length && importFiles([...e.target.files]);

document.addEventListener('keydown', (event) => {
  if (event.target.matches('input, textarea')) return;
  const keys = {
    ' ': toggleRecord,
    Enter: () => $('next').click(),
    ArrowRight: () => $('next').click(),
    ArrowLeft: () => $('prev').click(),
    p: playCurrent,
    r: redo,
  };
  const action = keys[event.key.length === 1 ? event.key.toLowerCase() : event.key];
  if (action) {
    event.preventDefault();
    action();
  }
});

window.addEventListener('beforeunload', () => {
  if (meter) cancelAnimationFrame(meter);
});

/* ------------------------------------------------------------------ *
 * Start
 * ------------------------------------------------------------------ */

(async () => {
  db = await openDb();
  if (!db) say('Kein Speicher verfügbar — Aufnahmen gehen beim Schließen verloren.', 'is-bad');
  await loadAll();
  rebuild();
  jumpToGap();
  render();
  if (clips.size) say(`${clips.size} Aufnahmen von letztem Mal geladen.`, 'is-good');
})();

// podgląd w konsoli — przydaje się, gdy coś nie gra
window.__studio = { clips, get key() { return currentKey(); } };
