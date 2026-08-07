/**
 * Nagrywa lektora i pakuje go do jednego pliku.
 *
 * Skrypt uruchamia się RAZ, u siebie. Gotowy pakiet ląduje w
 * `assets/voice/pack.js` i od tej pory gra mówi bez sieci — nagrania siedzą
 * w pliku, więc nic się nie dogrywa i nic nie kosztuje przy graniu.
 * Bez pakietu też nic się nie psuje: `src/speech.js` czyta wtedy głosem
 * systemowym przeglądarki.
 *
 * Są dwie drogi i obie kończą się tak samo — plikiem, który działa offline:
 *
 *   # 1. ElevenLabs: lepszy głos, jednorazowo zużywa znaki z abonamentu
 *   ELEVENLABS_API_KEY=... node tools/voice-build.mjs
 *
 *   # 2. Piper: model open source, liczy na własnym procesorze, za darmo
 *   node tools/voice-build.mjs --engine piper --piper-model pl_PL-gosia-medium.onnx
 *
 * Przebieg można bezpiecznie przerwać i wznowić: gotowe kwestie leżą
 * w .voice-cache/ i przy powtórce nie są nagrywane ponownie.
 *
 * Przydatne przełączniki:
 *   --dry             tylko policz, ile pójdzie do syntezy (nic nie wysyła)
 *   --engine <nazwa>  elevenlabs (domyślnie) albo piper
 *   --voice <id>      głos ElevenLabs (domyślnie ten wybrany do gry)
 *   --model <id>      model ElevenLabs, domyślnie eleven_v3 (zna znaczniki nastroju)
 *   --piper-model <p> ścieżka do pliku .onnx z głosem Pipera
 *   --limit <n>       nagraj tylko n pierwszych kwestii (do posłuchania próbki)
 *   --out <ścieżka>   domyślnie assets/voice/pack.js
 *
 * Jest jeszcze droga trzecia, gdy nagrania powstają gdzie indziej — choćby
 * ręcznie w przeglądarce albo cudzym narzędziem:
 *
 *   # spisz, co trzeba nagrać, razem z proponowanymi nazwami plików
 *   node tools/voice-build.mjs --manifest kwestie.csv
 *
 *   # złóż pakiet z katalogu gotowych nagrań (pliki nazwane jak w spisie)
 *   node tools/voice-build.mjs --from ./nagrania
 *
 * Nagrania trafiają do pakietu jako base64. Odtwarzamy je przez Blob, a nie
 * przez `data:` w atrybucie src — Firefox miewa problemy z dużymi data-URL-ami
 * w elemencie audio.
 */

import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

const run = promisify(execFile);

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const API = 'https://api.elevenlabs.io/v1/text-to-speech';

/** Głos wybrany do tej gry. Można nadpisać przełącznikiem --voice. */
const DEFAULT_VOICE = 'vwCkUb31or9ua8PEADUR';

/**
 * Nagrania trafiają najpierw tutaj, po jednym pliku na kwestię. Dzięki temu
 * przerwany albo powtórzony przebieg nie zamawia drugi raz tego, co już jest —
 * a że każde zamówienie kosztuje znaki z abonamentu, to nie drobiazg.
 */
const CACHE = '.voice-cache';

/* ------------------------------------------------------------------ *
 * Argumenty
 * ------------------------------------------------------------------ */

function arg(name, fallback = null) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}
const has = (name) => process.argv.includes(`--${name}`);

const options = {
  dry: has('dry'),
  engine: arg('engine', 'elevenlabs'),
  voice: arg('voice', DEFAULT_VOICE),
  model: arg('model', 'eleven_v3'),
  piperModel: arg('piper-model'),
  limit: Number(arg('limit', '0')) || 0,
  out: arg('out', 'assets/voice/pack.js'),
  manifest: arg('manifest'),
  from: arg('from'),
};

/** Identyfikator kwestii bywa z dwukropkiem, a ten nie wszędzie jest legalny w nazwie pliku. */
const fileNameFor = (id) => `${id.replace(/:/g, '__')}.mp3`;

/* ------------------------------------------------------------------ *
 * Co właściwie trzeba nagrać
 * ------------------------------------------------------------------ */

/**
 * Nastroje prowadzącego.
 *
 * Ten sam głos, trzy różne temperatury. `tag` trafia wyłącznie do syntezy —
 * model eleven_v3 czyta go jako wskazówkę aktorską i nie wypowiada na głos.
 * Do pakietu zapisujemy nagranie, a nie znacznik, więc żaden „[excited]”
 * nie wypłynie w grze.
 */
const MOOD = {
  // rzeczowa zapowiedź: numer rundy, polecenia
  spokojnie: { stability: 0.6, style: 0.2 },
  // samo pytanie — czytane wyraźnie, ale bez emfazy
  pytanie: { stability: 0.5, style: 0.3 },
  // chwila przed otwarciem zapadni
  napiecie: { stability: 0.55, style: 0.3, tag: '[whispers]' },
  // pieniądze zostały
  triumf: { stability: 0.25, style: 0.7, tag: '[excited]' },
  // pieniądze poleciały w dół
  zawod: { stability: 0.45, style: 0.5, tag: '[disappointed]' },
};

/**
 * Stałe kwestie prowadzącego. Klucz jest identyfikatorem nagrania w pakiecie.
 * „Pieniądze wracają do was” to zawołanie z teleturnieju — pada wtedy, gdy po
 * otwarciu zapadni cokolwiek zostało na stole.
 */
const FIXED = {
  intro: ['Witamy w grze Postaw na milion. Przed wami milion złotych w czterdziestu paczkach.', 'spokojnie'],
  wybierz: ['Dwa hasła. Proszę wybrać jedno.', 'spokojnie'],
  // odpowiedzi idą pojedynczo, pytanie dopiero po nich
  odpowiedzi: ['Oto odpowiedzi.', 'napiecie'],
  ...Object.fromEntries(
    ['A', 'B', 'C', 'D', 'E', 'F'].map((litera) => [
      `litera-${litera.toLowerCase()}`,
      [`${litera}.`, 'spokojnie'],
    ]),
  ),
  rozkladaj: ['Proszę rozłożyć pieniądze. Jedna zapadnia musi zostać pusta.', 'spokojnie'],
  'czas-start': ['Czas start!', 'triumf'],
  'czas-minal': ['Czas minął.', 'spokojnie'],
  zatwierdzone: ['Zatwierdzone. Nie ma odwrotu.', 'spokojnie'],
  cisza: ['Cisza na sali. Otwieramy zapadnie.', 'napiecie'],
  'puste-pole': ['To pole było puste.', 'spokojnie'],
  'leci-w-dol': ['I te pieniądze lecą w dół.', 'zawod'],
  poprawna: ['Poprawna odpowiedź to:', 'napiecie'],
  wracaja: ['Pieniądze wracają do was!', 'triumf'],
  'zostaje-nic': ['Niestety. Na stole nie został ani grosz.', 'zawod'],
  final: ['Finał. Zostały dwie zapadnie. Cała kwota musi trafić na jedną odpowiedź.', 'napiecie'],
  gratulacje: ['Gratulacje! Ta kwota jedzie do domu.', 'triumf'],
  milion: ['Milion złotych! Lepiej się nie da!', 'triumf'],
  ...Object.fromEntries(
    ['pierwsze', 'drugie', 'trzecie', 'czwarte', 'piąte', 'szóste', 'siódme', 'ósme'].map(
      (slowo, i) => [`runda-${i + 1}`, [`Pytanie ${slowo}.`, 'spokojnie']],
    ),
  ),
};

/** Tekst, który ma usłyszeć gracz — `spoken` bije `text`, gdy jest podany. */
const speakable = (item) => (item.spoken ?? item.text ?? '').trim();

async function collectLines() {
  const { QUESTIONS } = await import(join(ROOT, 'src/questions.js'));
  const lines = new Map();

  for (const [key, [text, mood]] of Object.entries(FIXED)) {
    lines.set(key, { text, mood });
  }

  // „stan na rok …” prowadzący czyta razem z pytaniem o rekord
  for (const rok of [...new Set(QUESTIONS.map((q) => q.asOf).filter(Boolean))].sort()) {
    lines.set(`stan-${rok}`, { text: `Stan na rok ${rok}.`, mood: 'spokojnie' });
  }

  for (const question of QUESTIONS) {
    const label = question.spokenLabel ?? question.label;
    if (label) lines.set(`${question.id}:haslo`, { text: `${label}.`, mood: 'spokojnie' });
    lines.set(`${question.id}:tresc`, { text: speakable(question), mood: 'pytanie' });
    question.answers.forEach((answer, i) => {
      const text = speakable(answer);
      if (text) lines.set(`${question.id}:odp${i}`, { text, mood: 'spokojnie' });
    });
  }

  return lines;
}

/* ------------------------------------------------------------------ *
 * Synteza
 * ------------------------------------------------------------------ */

/** Klucz w pamięci podręcznej: głos, model i treść — zmiana czegokolwiek nagrywa od nowa. */
function cacheKey(text, mood) {
  return createHash('sha1')
    .update(`${options.voice}|${options.model}|${mood}|${text}`)
    .digest('hex');
}

async function fromCache(text, mood) {
  try {
    return await readFile(join(ROOT, CACHE, `${cacheKey(text, mood)}.mp3`));
  } catch {
    return null;
  }
}

async function synthesize(text, mood, key) {
  const cached = await fromCache(text, mood);
  if (cached) return { audio: cached, cached: true };

  if (options.engine === 'piper') {
    return { audio: await synthesizePiper(text, mood), cached: false };
  }

  const { tag, ...settings } = MOOD[mood] ?? MOOD.spokojnie;

  const response = await fetch(`${API}/${options.voice}?output_format=mp3_22050_32`, {
    method: 'POST',
    headers: {
      'xi-api-key': key,
      'Content-Type': 'application/json',
      Accept: 'audio/mpeg',
    },
    body: JSON.stringify({
      // znacznik nastroju idzie tylko do modelu — w nagraniu go nie słychać
      text: tag ? `${tag} ${text}` : text,
      model_id: options.model,
      voice_settings: { similarity_boost: 0.75, use_speaker_boost: true, ...settings },
    }),
  });

  if (response.status === 429) {
    // limit zapytań — czekamy i próbujemy raz jeszcze, zamiast tracić dorobek przebiegu
    await new Promise((r) => setTimeout(r, 20_000));
    return synthesize(text, mood, key);
  }
  if (!response.ok) {
    throw new Error(`ElevenLabs ${response.status}: ${(await response.text()).slice(0, 300)}`);
  }

  const audio = Buffer.from(await response.arrayBuffer());
  await mkdir(join(ROOT, CACHE), { recursive: true });
  await writeFile(join(ROOT, CACHE, `${cacheKey(text, mood)}.mp3`), audio);
  return { audio, cached: false };
}

/* ------------------------------------------------------------------ *
 * Piper — synteza na własnym procesorze, bez konta i bez limitów
 * ------------------------------------------------------------------ */

/**
 * Piper nie zna znaczników nastroju, więc `[excited]` i spółka odpadają.
 * Zostaje to, czym da się sterować: tempo. Wolniej brzmi poważniej,
 * szybciej — żywiej. To uboższe niż ElevenLabs, ale nic nie kosztuje.
 */
const PIPER_RATE = {
  spokojnie: 1.0,
  pytanie: 1.05,
  napiecie: 1.15, // length_scale > 1 = wolniej
  triumf: 0.88,
  zawod: 1.12,
};

async function haveCommand(cmd) {
  try {
    await run(cmd, ['--help']);
    return true;
  } catch (error) {
    return error.code !== 'ENOENT';
  }
}

async function synthesizePiper(text, mood) {
  const wav = join(ROOT, CACHE, `${cacheKey(text, mood)}.wav`);
  const mp3 = join(ROOT, CACHE, `${cacheKey(text, mood)}.mp3`);

  await mkdir(join(ROOT, CACHE), { recursive: true });
  await run('piper', [
    '--model', options.piperModel,
    '--length_scale', String(PIPER_RATE[mood] ?? 1),
    '--output_file', wav,
  ], { input: text });

  // WAV z Pipera jest zbyt ciężki na pakiet w jednym pliku — kompresujemy
  await run('ffmpeg', ['-y', '-loglevel', 'error', '-i', wav, '-b:a', '32k', '-ac', '1', mp3]);
  await rm(wav, { force: true });
  return readFile(mp3);
}

/* ------------------------------------------------------------------ *
 * Główny przebieg
 * ------------------------------------------------------------------ */

const lines = await collectLines();
const entries = [...lines.entries()].filter(([, line]) => line.text.length > 0);
const budget = entries.reduce((sum, [, line]) => sum + line.text.length, 0);

console.log(`kwestii do nagrania: ${entries.length}`);
console.log(`znaków do syntezy:   ${budget.toLocaleString('pl-PL')}`);
console.log(`najdłuższa kwestia:  ${Math.max(...entries.map(([, l]) => l.text.length))} znaków`);

/* --- spis kwestii do nagrania gdzie indziej --- */
if (options.manifest) {
  const rows = [['plik', 'nastroj', 'znacznik', 'tekst']];
  for (const [id, line] of entries) {
    const { tag } = MOOD[line.mood] ?? {};
    rows.push([fileNameFor(id), line.mood, tag ?? '', line.text]);
  }
  const csv = rows
    .map((row) => row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(','))
    .join('\n');
  await writeFile(resolve(ROOT, options.manifest), `\uFEFF${csv}`); // BOM — Excel inaczej zjada polskie znaki
  console.log(`\n${options.manifest} — ${entries.length} kwestii do nagrania.`);
  console.log('Nagraj każdą jako osobny plik o nazwie z pierwszej kolumny, wrzuć do jednego');
  console.log('katalogu i złóż pakiet: node tools/voice-build.mjs --from <katalog>');
  process.exit(0);
}

/* --- złożenie pakietu z gotowych nagrań --- */
if (options.from) {
  const pack = {};
  const missing = [];
  for (const [id] of entries) {
    try {
      pack[id] = (await readFile(resolve(ROOT, options.from, fileNameFor(id)))).toString('base64');
    } catch {
      missing.push(fileNameFor(id));
    }
  }
  if (!Object.keys(pack).length) {
    console.error(`\nW katalogu ${options.from} nie ma ani jednego pasującego pliku.`);
    process.exit(1);
  }
  await writePack(pack, `z katalogu ${options.from}`, Object.keys(pack).length);
  if (missing.length) {
    console.log(`\nBrakuje ${missing.length} nagrań — te kwestie przeczyta głos systemowy:`);
    for (const name of missing.slice(0, 10)) console.log(`  ${name}`);
    if (missing.length > 10) console.log(`  …i ${missing.length - 10} więcej`);
  }
  process.exit(0);
}

if (options.dry) {
  const perMood = {};
  for (const [, line] of entries) perMood[line.mood] = (perMood[line.mood] ?? 0) + 1;
  console.log('nastroje:           ', Object.entries(perMood).map(([m, n]) => `${m} ${n}`).join(' · '));
  console.log('\n--dry: nic nie wysłano. Przykładowe kwestie:');
  for (const [key, line] of entries.slice(0, 10)) {
    const { tag } = MOOD[line.mood] ?? {};
    console.log(`  ${key.padEnd(16)} ${(tag ?? '').padEnd(15)} ${line.text}`);
  }
  process.exit(0);
}

const key = process.env.ELEVENLABS_API_KEY;

if (options.engine === 'piper') {
  if (!options.piperModel) {
    console.error('Brakuje --piper-model <plik.onnx>. Głosy: rhasspy/piper, katalog pl_PL.');
    process.exit(1);
  }
  for (const cmd of ['piper', 'ffmpeg']) {
    if (!(await haveCommand(cmd))) {
      console.error(`Nie znalazłem polecenia „${cmd}”. Piper potrzebuje obu.`);
      process.exit(1);
    }
  }
} else if (!key) {
  console.error('Brakuje ELEVENLABS_API_KEY. Uruchom z kluczem, dodaj --dry albo --engine piper.');
  process.exit(1);
}
const todo = options.limit ? entries.slice(0, options.limit) : entries;
const pack = {};
let done = 0;
let reused = 0;

console.log(
  options.engine === 'piper'
    ? `\nsilnik: piper · model: ${options.piperModel}`
    : `\nsilnik: elevenlabs · głos: ${options.voice} · model: ${options.model}`,
);
for (const [id, line] of todo) {
  const { audio, cached } = await synthesize(line.text, line.mood, key);
  pack[id] = audio.toString('base64');
  done++;
  if (cached) reused++;
  process.stdout.write(`\rnagrano ${done}/${todo.length} (z pamięci: ${reused})`);
}
process.stdout.write('\n');

await writePack(pack, stampFor(), done);
if (reused) console.log(`${reused} kwestii wzięto z ${CACHE}/ — nie kosztowały nic.`);

function stampFor() {
  return options.engine === 'piper'
    ? `piper ${options.piperModel}`
    : `${options.voice} · ${options.model}`;
}

async function writePack(pack, stamp, count) {
  const bytes = Object.values(pack).reduce((sum, b64) => sum + b64.length, 0);
  const fingerprint = createHash('sha1')
    .update(JSON.stringify(Object.keys(pack)))
    .digest('hex')
    .slice(0, 8);

  const file = `/**
 * Pakiet lektora — wygenerowany przez tools/voice-build.mjs. Nie edytować ręcznie.
 * Głos: ${stamp} · kwestii: ${count} · sygnatura: ${fingerprint}
 */
export const VOICE_PACK = ${JSON.stringify(pack)};
export const VOICE_FORMAT = 'audio/mpeg';
`;

  const target = resolve(ROOT, options.out);
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, file);
  console.log(`${options.out} — ${(bytes / 1024 / 1024).toFixed(1)} MB, ${count} kwestii`);
}
