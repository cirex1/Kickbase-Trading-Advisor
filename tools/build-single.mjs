/**
 * Składa całą grę w jeden plik HTML.
 *
 * Po co, skoro gra i tak nie potrzebuje budowania? Bo wersja modułowa wymaga
 * serwera HTTP — przeglądarki blokują moduły ES otwierane przez file://.
 * Plik złożony w całość otwiera się podwójnym kliknięciem i działa również
 * offline. To wygoda, a nie warunek: źródła w src/ pozostają tym, co się
 * edytuje.
 *
 *   node tools/build-single.mjs              → dist/postaw-na-milion.html
 *   node tools/build-single.mjs --fragment   → dodatkowo dist/fragment.html
 *                                              (sama zawartość <body>)
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

/** Kolejność ma znaczenie: moduł musi stać po tych, z których korzysta. */
const MODULES = [
  'src/storage.js',
  'src/engine.js',
  'src/questions.js',
  'src/audio.js',
  'src/confetti.js',
  'src/app.js',
];

/**
 * Zdejmuje składnię modułów. Po sklejeniu wszystko jest w jednym zasięgu,
 * więc importy nie mają czego wnosić, a eksporty — komu oddawać.
 * `[^;]*` obejmuje też znaki nowej linii, więc importy wielolinijkowe
 * (a takie są w app.js) znikają w całości.
 */
function stripModuleSyntax(source) {
  return source
    .replace(/^import\b[^;]*;/gm, '')
    .replace(/^export\s+default\s+[^;]*;/gm, '')
    .replace(/^export\s+/gm, '');
}

const read = (relative) => readFile(join(ROOT, relative), 'utf8');

const [html, css, ...sources] = await Promise.all([
  read('index.html'),
  read('assets/css/style.css'),
  ...MODULES.map(read),
]);

const script = MODULES.map((name, i) => `/* ===== ${name} ===== */\n${stripModuleSyntax(sources[i])}`)
  .join('\n')
  .trim();

// Klasyczny <script> zamiast modułu — moduły nie działają przez file://.
// IIFE trzyma zmienne przy sobie, zamiast rozsypywać je po globalnym zasięgu.
const inlineScript = `<script>\n(function () {\n'use strict';\n${script}\n})();\n</script>`;
const inlineStyle = `<style>\n${css.trim()}\n</style>`;

const standalone = html
  .replace(/[ \t]*<link rel="stylesheet"[^>]*>\n?/, () => `    ${inlineStyle}\n`)
  .replace(/[ \t]*<script type="module"[^>]*><\/script>\n?/, () => `    ${inlineScript}\n`);

if (standalone.includes('<link rel="stylesheet"') || standalone.includes('type="module"')) {
  throw new Error('Nie udało się wstawić stylów albo skryptu — sprawdź index.html.');
}

await mkdir(join(ROOT, 'dist'), { recursive: true });
await writeFile(join(ROOT, 'dist/postaw-na-milion.html'), standalone);
console.log(`dist/postaw-na-milion.html — ${(standalone.length / 1024).toFixed(0)} kB`);

if (process.argv.includes('--fragment')) {
  const title = standalone.match(/<title>([^<]*)<\/title>/)?.[1] ?? 'Postaw na milion';
  const body = standalone.match(/<body>([\s\S]*)<\/body>/)?.[1];
  if (!body) throw new Error('Nie znalazłem zawartości <body>.');
  const fragment = `<title>${title}</title>\n${inlineStyle}\n${body.trim()}\n`;
  await writeFile(join(ROOT, 'dist/fragment.html'), fragment);
  console.log(`dist/fragment.html — ${(fragment.length / 1024).toFixed(0)} kB`);
}
