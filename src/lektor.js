/**
 * Lektor — jedno wejście dla całej gry, dwa źródła dźwięku.
 *
 * Każda kwestia ma identyfikator (`key`) i zapasowy tekst. Jeśli w pakiecie
 * nagrań leży plik o tym identyfikatorze, słychać prawdziwy głos. Jeśli nie —
 * czyta syntezator systemowy ten sam tekst. Reszta gry nie musi wiedzieć,
 * która droga zadziałała, i dlatego wszystkie wywołania wyglądają tak samo:
 *
 *   lektor.say([
 *     { key: 'poprawna', text: 'Poprawna odpowiedź to:', pauseAfter: 500 },
 *     { key: `${q.id}:odp2`, text: 'Canberra' },
 *   ]);
 *
 * Identyfikatory muszą się zgadzać z tymi, które nadaje `tools/voice-build.mjs` —
 * to jedyne miejsce, gdzie te dwa światy się spotykają.
 */

import * as pack from './voicepack.js';
import * as tts from './speech.js';

/** Rośnie przy każdym przerwaniu — kończy sekwencję, która jest w trakcie. */
let generation = 0;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const normalize = (segments) =>
  (Array.isArray(segments) ? segments : [segments])
    .filter(Boolean)
    .map((s) => (typeof s === 'string' ? { text: s } : s));

/* ------------------------------------------------------------------ *
 * Stan
 * ------------------------------------------------------------------ */

export const isSupported = () => tts.isSupported() || typeof Audio !== 'undefined';

/** Włączony i ma czym mówić: albo nagraniami, albo syntezatorem. */
export function isOn() {
  return tts.isEnabled() && (pack.isReady() || tts.hasVoice());
}

export function toggle() {
  const on = tts.toggle();
  if (!on) stop();
  return on;
}

export function hasPack() {
  return pack.isReady();
}

export function packSize() {
  return pack.count();
}

export const voices = tts.voices;
export const currentVoice = tts.currentVoice;
export const setVoice = tts.setVoice;

/**
 * Pakiet nagrań próbujemy wczytać raz, przy starcie. Gdy go nie ma, nic się
 * nie dzieje — `init` i tak zwróci to, co znalazł syntezator.
 */
export async function init() {
  const [state] = await Promise.all([tts.init(), pack.load()]);
  return { ...state, pack: pack.isReady(), packLines: pack.count() };
}

/** Pierwsze odtworzenie musi wyjść z gestu użytkownika — stąd cisza na start. */
export function unlock() {
  tts.unlock();
}

/* ------------------------------------------------------------------ *
 * Mówienie
 * ------------------------------------------------------------------ */

/**
 * Mówi po kolei. Nie czekamy na to w interfejsie — mowa ma towarzyszyć grze,
 * nie wstrzymywać jej.
 */
export async function say(segments, defaults = {}) {
  if (!isOn()) return false;

  const list = normalize(segments);
  const mine = ++generation;
  pack.stop();

  // Bez pakietu oddajemy całość syntezatorowi jednym wywołaniem: sam dzieli
  // tekst i sam pilnuje kolejki, więc nie ma po co się w to wtrącać.
  if (!pack.isReady()) return tts.say(list, defaults);

  for (const segment of list) {
    if (mine !== generation) return false;

    const played = segment.key ? await pack.play(segment.key) : false;
    if (!played) {
      if (mine !== generation) return false;
      // pojedynczy segment — syntezator dopowiada to, czego nie nagrano
      await tts.say(segment, defaults);
    }
    if (segment.pauseAfter) await sleep(segment.pauseAfter);
  }
  return mine === generation;
}

export function stop() {
  generation++;
  pack.stop();
  tts.stop();
}
