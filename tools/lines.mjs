/**
 * Spis wszystkiego, co lektor ma do powiedzenia.
 *
 * Jedno miejsce dla dwóch narzędzi: `voice-build.mjs` zamawia z tej listy
 * syntezę, `build-studio.mjs` robi z niej stronę do nagrywania własnym głosem.
 * Gdyby każdy trzymał własną kopię, wystarczyłaby jedna literówka, żeby
 * nagranie nigdy nie trafiło do gry — klucz musi zgadzać się co do znaku
 * z tym, o który prosi `src/app.js`.
 */

import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

/**
 * Nastroje prowadzącego.
 *
 * Ten sam głos, kilka temperatur. `tag` trafia wyłącznie do syntezy — model
 * eleven_v3 czyta go jako wskazówkę aktorską i nie wypowiada na głos. Przy
 * nagraniu własnym głosem `hint` mówi to samo po ludzku.
 */
export const MOOD = {
  // rzeczowa zapowiedź: numer rundy, polecenia
  spokojnie: { stability: 0.6, style: 0.2, hint: 'spokojnie, rzeczowo' },
  // samo pytanie — czytane wyraźnie, ale bez emfazy
  pytanie: { stability: 0.5, style: 0.3, hint: 'wyraźnie, bez emfazy' },
  // chwila przed otwarciem zapadni
  napiecie: { stability: 0.55, style: 0.3, tag: '[whispers]', hint: 'ciszej, z napięciem' },
  // pieniądze zostały
  triumf: { stability: 0.25, style: 0.7, tag: '[excited]', hint: 'głośno, z radością' },
  // pieniądze poleciały w dół
  zawod: { stability: 0.45, style: 0.5, tag: '[disappointed]', hint: 'z żalem, ciężko' },
};

/**
 * Stałe kwestie prowadzącego. Klucz jest identyfikatorem nagrania w pakiecie.
 * „Pieniądze wracają do was” to zawołanie z teleturnieju — pada wtedy, gdy po
 * otwarciu zapadni cokolwiek zostało na stole.
 */
export const FIXED = {
  intro: [
    'Witamy w grze Postaw na milion. Przed wami milion złotych w czterdziestu paczkach.',
    'spokojnie',
  ],
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

/**
 * Grupy do nagrywania po kawałku.
 *
 * Kolejność nie jest przypadkowa: same kwestie prowadzącego to kilkanaście
 * minut pracy, a słychać je w każdej rundzie. Pytania i odpowiedzi to reszta
 * materiału — brak któregokolwiek nagrania niczego nie psuje, bo gra dopowiada
 * je syntezatorem.
 */
export const GROUPS = [
  { id: 'prowadzacy', name: 'Prowadzący', note: 'słychać w każdej rundzie — od tego zacznij' },
  { id: 'hasla', name: 'Hasła', note: 'nazwy na tablicy przed pytaniem' },
  { id: 'pytania', name: 'Pytania', note: 'treść pytań' },
  { id: 'odpowiedzi', name: 'Odpowiedzi', note: 'cztery warianty do każdego pytania' },
];

/** Identyfikator kwestii bywa z dwukropkiem, a ten nie wszędzie jest legalny w nazwie pliku. */
export const fileNameFor = (id, ext = 'mp3') => `${id.replace(/:/g, '__')}.${ext}`;

/** Tekst, który ma usłyszeć gracz — `spoken` bije `text`, gdy jest podany. */
const speakable = (item) => (item.spoken ?? item.text ?? '').trim();

export async function collectLines() {
  const { QUESTIONS } = await import(join(ROOT, 'src/questions.js'));
  const lines = new Map();

  for (const [key, [text, mood]] of Object.entries(FIXED)) {
    lines.set(key, { text, mood, group: 'prowadzacy' });
  }

  // „stan na rok …” prowadzący czyta razem z pytaniem o rekord
  for (const rok of [...new Set(QUESTIONS.map((q) => q.asOf).filter(Boolean))].sort()) {
    lines.set(`stan-${rok}`, { text: `Stan na rok ${rok}.`, mood: 'spokojnie', group: 'prowadzacy' });
  }

  for (const question of QUESTIONS) {
    const label = question.spokenLabel ?? question.label;
    if (label) {
      lines.set(`${question.id}:haslo`, { text: `${label}.`, mood: 'spokojnie', group: 'hasla' });
    }
    lines.set(`${question.id}:tresc`, {
      text: speakable(question),
      mood: 'pytanie',
      group: 'pytania',
    });
    question.answers.forEach((answer, i) => {
      const text = speakable(answer);
      if (text) {
        lines.set(`${question.id}:odp${i}`, { text, mood: 'spokojnie', group: 'odpowiedzi' });
      }
    });
  }

  return lines;
}
