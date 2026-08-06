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
 *
 * Dwie rzeczy, które ten skrypt robi inaczej niż zwykłe sklejenie plików,
 * bo oba braki już raz wypuściły zepsutą grę:
 *
 * 1. Moduły odnajduje sam, idąc po importach od punktu wejścia. Ręcznie
 *    utrzymywana lista pominęła kiedyś speech.js — wersja modułowa działała,
 *    a jednoplikowa wywalała się przy pierwszym kliknięciu.
 * 2. Każdy moduł dostaje własne domknięcie zamiast trafiać do wspólnego
 *    zasięgu. Inaczej dwie jednostki z własną zmienną `enabled` dałyby błąd
 *    składni — a takie już w tym projekcie są.
 *
 * Na koniec gotowy skrypt jest sprawdzany składniowo. Jeśli się nie parsuje,
 * nic nie zostaje zapisane.
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

/** Punkt wejścia — resztę wyznacza graf importów. */
const ENTRY = 'src/app.js';

/** `import { a, b } from './x.js'` — obejmuje też zapis wielolinijkowy. */
const NAMED_IMPORT = /^import\s*\{([^}]*)\}\s*from\s*'([^']+)';/gm;
/** `import * as coś from './x.js'` */
const STAR_IMPORT = /^import\s*\*\s*as\s+(\w+)\s+from\s*'([^']+)';/gm;

const read = (relative) => readFile(join(ROOT, relative), 'utf8');

/** Nazwy, które moduł udostępnia na zewnątrz. */
function exportedNames(source) {
  const names = new Set();
  for (const m of source.matchAll(/^export\s+(?:async\s+)?function\s+(\w+)/gm)) names.add(m[1]);
  for (const m of source.matchAll(/^export\s+(?:const|let|var)\s+(\w+)/gm)) names.add(m[1]);
  return [...names];
}

/** Ścieżka importu sprowadzona do postaci liczonej od korzenia projektu. */
function resolveImport(fromFile, spec) {
  return join(dirname(fromFile), spec).split('\\').join('/');
}

/**
 * Przechodzi graf importów w głąb i zwraca moduły w kolejności, w której
 * wolno je skleić: zależność zawsze przed tym, kto z niej korzysta.
 */
async function collectModules(entry) {
  const order = [];
  const seen = new Set();

  async function visit(relative) {
    if (seen.has(relative)) return;
    seen.add(relative);
    const source = await read(relative);
    const deps = [
      ...[...source.matchAll(NAMED_IMPORT)].map((m) => m[2]),
      ...[...source.matchAll(STAR_IMPORT)].map((m) => m[2]),
    ].filter((spec) => spec.startsWith('.'));
    for (const spec of deps) await visit(resolveImport(relative, spec));
    order.push({ path: relative, source });
  }

  await visit(entry);
  return order;
}

/** Zamyka moduł w funkcji i oddaje jego eksporty do wspólnego rejestru. */
function wrapModule({ path, source }) {
  const exported = exportedNames(source);
  const prelude = [];

  let body = source
    .replace(NAMED_IMPORT, (all, names, spec) => {
      if (!spec.startsWith('.')) return all;
      prelude.push(
        `const { ${names.replace(/\s+/g, ' ').trim()} } = __mod['${resolveImport(path, spec)}'];`,
      );
      return '';
    })
    .replace(STAR_IMPORT, (all, alias, spec) => {
      if (!spec.startsWith('.')) return all;
      prelude.push(`const ${alias} = __mod['${resolveImport(path, spec)}'];`);
      return '';
    })
    .replace(/^export\s+default\s+[^;]*;/gm, '')
    .replace(/^export\s+/gm, '')
    .trim();

  return `/* ===== ${path} ===== */
__mod['${path}'] = (function () {
${prelude.join('\n')}
${body}
return { ${exported.join(', ')} };
})();`;
}

/* ------------------------------------------------------------------ *
 * Złożenie
 * ------------------------------------------------------------------ */

const [html, css] = await Promise.all([read('index.html'), read('assets/css/style.css')]);
const modules = await collectModules(ENTRY);
console.log(
  `modułów: ${modules.length} — ${modules.map((m) => m.path.replace('src/', '')).join(', ')}`,
);

const script = ['const __mod = {};', ...modules.map(wrapModule)].join('\n\n');

// Sprawdzenie składni PRZED zapisem — lepiej nie zbudować nic niż wypuścić
// plik, który wygląda dobrze, a w przeglądarce milczy.
try {
  new Function(script);
} catch (error) {
  console.error(`\nZłożony skrypt się nie parsuje: ${error.message}`);
  console.error('Nic nie zapisano.');
  process.exit(1);
}

// Klasyczny <script> zamiast modułu — moduły nie działają przez file://.
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
