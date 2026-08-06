/**
 * Dźwięki generowane w locie przez Web Audio API — zero plików, zero pobierania.
 * Kontekst audio tworzymy dopiero przy pierwszym kliknięciu, bo przeglądarki
 * blokują dźwięk bez interakcji użytkownika.
 */

import { readSetting, writeSetting } from './storage.js';

const STORAGE_KEY = 'pnm.sound';

let ctx = null;
let master = null;
let enabled = readSetting(STORAGE_KEY) !== 'off';

function ensureContext() {
  if (ctx) return ctx;
  const Ctor = window.AudioContext || window.webkitAudioContext;
  if (!Ctor) return null;
  ctx = new Ctor();
  master = ctx.createGain();
  master.gain.value = 0.22;
  master.connect(ctx.destination);
  return ctx;
}

/** Pojedynczy oscylator z obwiednią ADSR w wersji minimalnej. */
function tone({ freq, type = 'sine', start = 0, duration = 0.18, gain = 1, sweepTo = null }) {
  if (!enabled) return;
  const audio = ensureContext();
  if (!audio) return;
  if (audio.state === 'suspended') audio.resume();

  const t0 = audio.currentTime + start;
  const osc = audio.createOscillator();
  const env = audio.createGain();

  osc.type = type;
  osc.frequency.setValueAtTime(freq, t0);
  if (sweepTo) osc.frequency.exponentialRampToValueAtTime(sweepTo, t0 + duration);

  env.gain.setValueAtTime(0.0001, t0);
  env.gain.exponentialRampToValueAtTime(gain, t0 + 0.012);
  env.gain.exponentialRampToValueAtTime(0.0001, t0 + duration);

  osc.connect(env).connect(master);
  osc.start(t0);
  osc.stop(t0 + duration + 0.05);
}

/** Krótki szum — używany do „zmiatania” przegranych żetonów. */
function noise({ duration = 0.5, gain = 0.5, from = 2400, to = 200 } = {}) {
  if (!enabled) return;
  const audio = ensureContext();
  if (!audio) return;
  if (audio.state === 'suspended') audio.resume();

  const frames = Math.floor(audio.sampleRate * duration);
  const buffer = audio.createBuffer(1, frames, audio.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < frames; i++) {
    data[i] = (Math.random() * 2 - 1) * (1 - i / frames);
  }
  const src = audio.createBufferSource();
  src.buffer = buffer;

  const filter = audio.createBiquadFilter();
  filter.type = 'bandpass';
  filter.frequency.setValueAtTime(from, audio.currentTime);
  filter.frequency.exponentialRampToValueAtTime(to, audio.currentTime + duration);

  const env = audio.createGain();
  env.gain.value = gain;

  src.connect(filter).connect(env).connect(master);
  src.start();
}

export const sfx = {
  /** Chwycenie pliku paczek. */
  grab: () => tone({ freq: 880, type: 'triangle', duration: 0.09, gain: 0.5 }),
  /** Paczka ląduje na zapadni — krótkie, papierowe klapnięcie. */
  place: () => {
    noise({ duration: 0.09, gain: 0.22, from: 1800, to: 500 });
    tone({ freq: 210, type: 'sine', duration: 0.09, gain: 0.3 });
  },
  remove: () => tone({ freq: 320, type: 'triangle', duration: 0.1, gain: 0.35, sweepTo: 180 }),
  /** Próba położenia paczki na ostatnim wolnym polu. */
  blocked: () => tone({ freq: 150, type: 'square', duration: 0.14, gain: 0.3 }),
  tick: () => tone({ freq: 1200, type: 'square', duration: 0.04, gain: 0.25 }),
  hurry: () => tone({ freq: 1500, type: 'square', duration: 0.06, gain: 0.45 }),
  /** Zatwierdzenie — zamek się blokuje. */
  lock: () => {
    tone({ freq: 200, type: 'sawtooth', duration: 0.45, gain: 0.4, sweepTo: 90 });
    tone({ freq: 400, type: 'sine', start: 0.05, duration: 0.35, gain: 0.25 });
  },
  /** Napięcie tuż przed otwarciem zapadni. */
  tension: () => {
    tone({ freq: 55, type: 'sine', duration: 1.1, gain: 0.5 });
    [0, 0.45].forEach((s) => tone({ freq: 70, type: 'sine', start: s, duration: 0.22, gain: 0.6 }));
  },
  /** Mechanizm zapadni: zgrzyt zasuwy i uderzenie skrzydeł. */
  doors: () => {
    tone({ freq: 320, type: 'square', duration: 0.07, gain: 0.35 });
    noise({ duration: 0.35, gain: 0.4, from: 900, to: 120 });
    tone({ freq: 90, type: 'sawtooth', start: 0.12, duration: 0.4, gain: 0.55, sweepTo: 45 });
  },
  /** Paczki lecą w ciemność — szum opadający razem z nimi. */
  fall: () => {
    noise({ duration: 1.1, gain: 0.4, from: 2600, to: 90 });
    tone({ freq: 260, type: 'sine', duration: 1.0, gain: 0.3, sweepTo: 55 });
    tone({ freq: 60, type: 'sine', start: 0.85, duration: 0.5, gain: 0.5 });
  },
  /** Pieniądze zostały na zapadni. */
  safe: () => {
    [523.25, 659.25, 783.99, 1046.5].forEach((f, i) =>
      tone({ freq: f, type: 'triangle', start: i * 0.09, duration: 0.35, gain: 0.55 }),
    );
  },
  wrong: () => {
    tone({ freq: 180, type: 'sawtooth', duration: 0.55, gain: 0.5, sweepTo: 70 });
    noise({ duration: 0.6, gain: 0.35 });
  },
  win: () => {
    [523.25, 659.25, 783.99, 1046.5, 1318.5, 1567.98].forEach((f, i) =>
      tone({ freq: f, type: 'triangle', start: i * 0.11, duration: 0.6, gain: 0.6 }),
    );
    tone({ freq: 261.63, type: 'sine', start: 0.55, duration: 1.4, gain: 0.4 });
  },
  gameOver: () => {
    [392, 349.23, 293.66, 196].forEach((f, i) =>
      tone({ freq: f, type: 'sine', start: i * 0.18, duration: 0.6, gain: 0.45 }),
    );
  },
};

export function isSoundOn() {
  return enabled;
}

export function toggleSound() {
  enabled = !enabled;
  writeSetting(STORAGE_KEY, enabled ? 'on' : 'off');
  if (enabled) sfx.grab();
  return enabled;
}
