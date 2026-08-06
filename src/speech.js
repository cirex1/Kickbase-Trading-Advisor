/**
 * Lektor — czyta hasła, pytania i rozwiązania polskim głosem systemowym.
 *
 * Korzystamy z Web Speech API, bo jako jedyna droga spełnia wszystkie trzy
 * warunki tego projektu naraz: nie waży ani bajta, działa spod file:// i nie
 * dotyka jej Content-Security-Policy — synteza mowy nie jest z punktu widzenia
 * CSP żadnym pobraniem, więc nie trzeba otwierać `media-src`.
 *
 * Cena: głos daje system, nie my. Dlatego bierzemy wyłącznie głosy
 * z `localService === true`. To gwarantuje działanie bez sieci, a przy okazji
 * omija błąd Chromium, w którym mowa z głosów serwerowych urywa się po
 * kilkunastu sekundach.
 *
 * SSML odpada — przeglądarki go nie wspierają, a macOS potrafi przeczytać
 * znaczniki na głos. Pauzy i zmiany tempa robimy więc dzieląc tekst na
 * kawałki i czekając między nimi.
 */

import { readSetting, writeSetting } from './storage.js';

const ON_KEY = 'pnm.lektor';
const VOICE_KEY = 'pnm.glos';

/** Znane dobre głosy polskie, od najlepszego. Dopasowanie po fragmencie nazwy. */
const PREFERRED = [
  'google polski',
  'google polish',
  'zosia',
  'krzysztof',
  'ewa',
  'jacek',
  'agnieszka',
  'paulina',
  'adam',
];

/** Syntezatory ostatniej szansy — zrozumiałe, ale brzmią jak robot z lat 80. */
const ROBOTIC = /espeak|festival|mbrola|pico|flite|sam\b/;

/** Chrome tnie dłuższe wypowiedzi, więc dzielimy je na zdania. */
const MAX_CHUNK = 180;

const synth = typeof window !== 'undefined' ? window.speechSynthesis : null;

let available = [];
let voice = null;
let unlocked = false;
let enabled = readSetting(ON_KEY) === 'on';
let held = null; // twarda referencja: bez niej zbieracz śmieci potrafi uciszyć mowę
let generation = 0; // rośnie przy każdym stop() — przerywa trwającą sekwencję

const isPolish = (v) => /^pl($|[-_])/i.test(v.lang || '');

function score(v) {
  const name = `${v.name || ''} ${v.voiceURI || ''}`.toLowerCase();
  let points = 0;
  const rank = PREFERRED.findIndex((k) => name.includes(k));
  if (rank >= 0) points += (PREFERRED.length - rank) * 100;
  if (ROBOTIC.test(name)) points -= 1000;
  if (v.localService) points += 400; // bez sieci i bez błędu z urywaniem
  if (/pl[-_]pl/i.test(v.lang)) points += 20;
  return points;
}

/**
 * `getVoices()` bywa puste przy pierwszym wywołaniu. Zdarzenie `voiceschanged`
 * na części przeglądarek nie pada nigdy, a Safari wypełnia listę od razu —
 * dlatego nasłuchujemy, odpytujemy w pętli i tak czy siak kończymy po chwili.
 */
function loadVoices(timeout = 3000) {
  return new Promise((resolve) => {
    if (!synth) return resolve([]);
    let settled = false;
    let poll = 0;
    let bail = 0;

    const finish = () => {
      if (settled) return;
      settled = true;
      synth.removeEventListener('voiceschanged', check);
      clearInterval(poll);
      clearTimeout(bail);
      resolve(synth.getVoices() || []);
    };
    const check = () => {
      if ((synth.getVoices() || []).length) finish();
    };

    if ((synth.getVoices() || []).length) return finish();
    synth.addEventListener('voiceschanged', check);
    poll = setInterval(check, 100);
    bail = setTimeout(finish, timeout);
  });
}

export function isSupported() {
  return Boolean(synth);
}

export async function init() {
  if (!synth) return { voices: [], voice: null, offlineSafe: false };
  try {
    synth.cancel(); // sprzątamy kolejkę po poprzedniej wizycie na stronie
  } catch {
    /* nic nie szkodzi */
  }

  const all = await loadVoices();
  available = all.filter(isPolish).sort((a, b) => score(b) - score(a));

  const remembered = readSetting(VOICE_KEY);
  voice =
    available.find((v) => v.voiceURI === remembered || v.name === remembered) ??
    available.find((v) => v.localService) ??
    available[0] ??
    null;

  return { voices: available, voice, offlineSafe: Boolean(voice?.localService) };
}

export function voices() {
  return available;
}

export function currentVoice() {
  return voice;
}

export function setVoice(idOrName) {
  const found = available.find((v) => v.voiceURI === idOrName || v.name === idOrName);
  if (!found) return false;
  voice = found;
  writeSetting(VOICE_KEY, found.voiceURI || found.name);
  return true;
}

export function isOn() {
  return enabled && Boolean(voice);
}

export function toggle() {
  enabled = !enabled;
  writeSetting(ON_KEY, enabled ? 'on' : 'off');
  if (!enabled) stop();
  return enabled;
}

/**
 * Pierwsze `speak()` musi wyjść z gestu użytkownika, inaczej przeglądarka
 * odmawia. Wypuszczamy więc ciszę przy kliknięciu „Zaczynamy”.
 */
export function unlock() {
  if (unlocked || !synth) return;
  const silence = new SpeechSynthesisUtterance(' ');
  silence.volume = 0;
  try {
    synth.speak(silence);
    unlocked = true;
  } catch {
    /* trudno — powiemy przy następnym kliknięciu */
  }
}

/** Dzieli tekst na kawałki na granicach zdań. */
function chunk(text) {
  const clean = String(text).replace(/\s+/g, ' ').trim();
  if (clean.length <= MAX_CHUNK) return clean ? [clean] : [];

  const sentences = clean.match(/[^.!?…]+[.!?…]*/g) || [clean];
  const out = [];
  let buffer = '';
  const flush = () => {
    if (buffer.trim()) out.push(buffer.trim());
    buffer = '';
  };
  for (const sentence of sentences) {
    if ((buffer + ' ' + sentence).length > MAX_CHUNK) flush();
    if (sentence.length > MAX_CHUNK) {
      for (const word of sentence.split(' ')) {
        if ((buffer + ' ' + word).length > MAX_CHUNK) flush();
        buffer += (buffer ? ' ' : '') + word;
      }
      flush();
    } else {
      buffer += (buffer ? ' ' : '') + sentence;
    }
  }
  flush();
  return out;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function speakChunk(text, { rate = 0.95, pitch = 1 }) {
  return new Promise((resolve) => {
    const utterance = new SpeechSynthesisUtterance(text);
    held = utterance;
    if (voice) utterance.voice = voice;
    utterance.lang = voice?.lang || 'pl-PL';
    utterance.rate = rate;
    utterance.pitch = pitch;

    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      held = null;
      resolve();
    };
    utterance.onend = finish;
    utterance.onerror = finish;
    // koło ratunkowe: gdyby żadne zdarzenie nie padło, nie blokujemy sekwencji
    setTimeout(finish, Math.max(3500, text.length * 200));
    synth.speak(utterance);
  });
}

/**
 * Mówi po kolei. Segment to tekst albo `{ text, rate, pitch, pauseAfter }`.
 * Nie czekamy na to w interfejsie — mowa ma towarzyszyć grze, nie wstrzymywać jej.
 */
export async function say(segments, defaults = {}) {
  if (!isOn() || !unlocked) return false;
  const mine = ++generation;
  try {
    synth.cancel();
  } catch {
    return false;
  }

  const list = (Array.isArray(segments) ? segments : [segments])
    .filter(Boolean)
    .map((s) => (typeof s === 'string' ? { text: s } : s));

  for (const segment of list) {
    for (const part of chunk(segment.text)) {
      if (mine !== generation) return false; // ktoś w międzyczasie przerwał
      await speakChunk(part, { ...defaults, ...segment });
    }
    if (segment.pauseAfter) await sleep(segment.pauseAfter);
  }
  return mine === generation;
}

export function stop() {
  generation++;
  held = null;
  try {
    synth?.cancel();
  } catch {
    /* nic nie szkodzi */
  }
}

if (typeof window !== 'undefined') {
  window.addEventListener('pagehide', stop);
}
