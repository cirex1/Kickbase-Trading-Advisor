/**
 * Pakiet nagrań lektora — prawdziwy głos zamiast syntezatora.
 *
 * Nagrania powstają raz, poza grą (`tools/voice-build.mjs`), i lądują
 * w jednym pliku `assets/voice/pack.js`. Plik jest zwykłym skryptem, nie
 * modułem — i to jest cała sztuczka: moduły ES przeglądarki blokują przy
 * otwarciu przez file://, a klasyczny <script src="…"> ładuje się także
 * z dwukliku. Dzięki temu gra mówi ludzkim głosem również offline, bez
 * serwera i bez konta.
 *
 * Pakiet dokładamy osobno, a nie do wersji jednoplikowej, z prozaicznego
 * powodu: waży kilka megabajtów. Wklejony w HTML musiałby się wczytać
 * w całości, zanim cokolwiek się pokaże.
 *
 * Czego tu nie ma: pobierania czegokolwiek z sieci w trakcie gry. Jeśli
 * pakietu nie ma albo się nie wczyta, moduł mówi „nie mam” i lektor wraca
 * do głosu systemowego. Brak nagrań nigdy nie jest błędem.
 */

/**
 * Gdzie szukać pakietu — pierwsza pasująca ścieżka wygrywa. Druga jest dla
 * wersji jednoplikowej uruchamianej z `dist/`, gdy pakiet został przy źródłach.
 *
 * Gdy pakietu nie ma, konsola przeglądarki zapisze „ERR_FILE_NOT_FOUND”. To nie
 * jest usterka, tylko jedyny sposób, w jaki da się sprawdzić obecność pliku
 * spod file:// — fetch jest tam zablokowany.
 */
const CANDIDATES = ['assets/voice/pack.js', '../assets/voice/pack.js'];

let pack = null;
let format = 'audio/mpeg';
let loading = null;
/** Adresy Blobów, po jednym na kwestię — tworzone dopiero przy pierwszym użyciu. */
const urls = new Map();
let current = null;

/** Czy pakiet jest już w pamięci. */
export function isReady() {
  return Boolean(pack);
}

export function count() {
  return pack ? Object.keys(pack).length : 0;
}

export function has(key) {
  return Boolean(pack && pack[key]);
}

function adopt(source) {
  if (!source || typeof source !== 'object') return false;
  const lines = source.lines ?? source.VOICE_PACK ?? source;
  if (!lines || typeof lines !== 'object' || !Object.keys(lines).length) return false;
  pack = lines;
  format = source.format ?? source.VOICE_FORMAT ?? format;
  return true;
}

function loadScript(src) {
  return new Promise((resolve) => {
    const tag = document.createElement('script');
    tag.src = src;
    tag.async = true;
    tag.onload = () => resolve(true);
    tag.onerror = () => {
      tag.remove();
      resolve(false);
    };
    document.head.append(tag);
  });
}

/**
 * Próbuje wczytać pakiet. Wywołanie jest bezpieczne wielokrotnie — kolejne
 * dostają tę samą obietnicę, a brak pliku kończy się spokojnym `false`.
 */
export function load() {
  if (pack) return Promise.resolve(true);
  if (loading) return loading;

  loading = (async () => {
    if (typeof window === 'undefined') return false;
    // ktoś mógł dołożyć pakiet własnym <script>-em w index.html
    if (adopt(window.PNM_VOICE)) return true;

    for (const src of CANDIDATES) {
      if (await loadScript(src)) {
        if (adopt(window.PNM_VOICE)) return true;
      }
    }
    return false;
  })();

  return loading;
}

/**
 * Kwestia bywa zapisana na dwa sposoby. Syntezator oddaje same dane, bo
 * wszystkie nagrania mają ten sam format. Nagrywarka dokłada do każdej własny
 * typ — bo przeglądarki nagrywają w czym innym, a do tego wolno dorzucić pliki
 * z dysku, więc w jednym pakiecie potrafi być i webm, i mp3.
 */
function urlFor(key) {
  if (!has(key)) return null;
  if (urls.has(key)) return urls.get(key);
  try {
    const entry = pack[key];
    const data = typeof entry === 'string' ? entry : entry.d;
    const type = (typeof entry === 'string' ? format : entry.t) || format;

    const binary = atob(data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const url = URL.createObjectURL(new Blob([bytes], { type }));
    urls.set(key, url);
    return url;
  } catch {
    return null;
  }
}

/**
 * Odtwarza jedną kwestię i czeka, aż wybrzmi. Zwraca `false`, gdy nagrania
 * nie ma albo przeglądarka odmówiła — wtedy woła się syntezator.
 */
export function play(key) {
  const url = urlFor(key);
  if (!url) return Promise.resolve(false);

  return new Promise((resolve) => {
    const audio = new Audio(url);
    current = audio;
    let settled = false;
    const finish = (ok) => {
      if (settled) return;
      settled = true;
      if (current === audio) current = null;
      resolve(ok);
    };
    audio.onended = () => finish(true);
    audio.onerror = () => finish(false);
    // koło ratunkowe: gdyby żadne zdarzenie nie padło, nie blokujemy sekwencji
    setTimeout(() => finish(true), 20_000);
    audio.play().catch(() => finish(false));
  });
}

export function stop() {
  if (!current) return;
  try {
    current.pause();
  } catch {
    /* nic nie szkodzi */
  }
  current = null;
}
