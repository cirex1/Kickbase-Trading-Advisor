import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BUNDLE,
  CATEGORY_CHOICES,
  DOORS_PLAN,
  FINAL_DOORS,
  GRAB_SIZES,
  ROUND_COUNT,
  ROUND_SECONDS,
  START_BALANCE,
  advance,
  availableQuestions,
  canLock,
  canPlaceOn,
  chooseOffer,
  clearBets,
  createGame,
  currentQuestion,
  WARMUP_DIFFICULTY,
  difficultyFloor,
  doorsInRound,
  emptyDoors,
  isFinalRound,
  isWarmupRound,
  makeRng,
  offeredQuestions,
  place,
  placeBundles,
  placeRest,
  resolve,
  revealOrder,
  trimToDoors,
  take,
  undo,
  unplaced,
  unplacedBundles,
} from '../src/engine.js';
import { QUESTIONS } from '../src/questions.js';

const newGame = (seed = 'test') => createGame({ questions: QUESTIONS, seed });

/** Rozpoczyna rundę, biorąc pierwsze z proponowanych haseł. */
const enterRound = (state) => chooseOffer(state, offeredQuestions(state)[0].id);

const goodIndex = (state) => currentQuestion(state).answers.findIndex((a) => a.correct);
const badIndex = (state) => currentQuestion(state).answers.findIndex((a) => !a.correct);

/* -------------------------------------------------------------- */
/* Baza pytań                                                     */
/* -------------------------------------------------------------- */

test('każde pytanie ma unikalne id', () => {
  const ids = QUESTIONS.map((q) => q.id);
  assert.equal(new Set(ids).size, ids.length);
});

test('każde pytanie ma cztery odpowiedzi i dokładnie jedną poprawną', () => {
  for (const q of QUESTIONS) {
    assert.equal(q.answers.length, 4, `${q.id}: powinny być 4 zapadnie`);
    assert.equal(
      q.answers.filter((a) => a.correct).length,
      1,
      `${q.id}: dokładnie jedna odpowiedź ma być poprawna`,
    );
    assert.ok(q.difficulty >= 1 && q.difficulty <= 5, `${q.id}: zły poziom trudności`);
    assert.ok(q.category && q.text && q.note, `${q.id}: brakuje kategorii, treści lub wyjaśnienia`);
  }
});

test('starczy pytań na rozgrzewkę i na resztę partii', () => {
  const warmup = QUESTIONS.filter((q) => q.difficulty === WARMUP_DIFFICULTY);
  const rest = QUESTIONS.filter((q) => q.difficulty !== WARMUP_DIFFICULTY);

  // pierwsza runda: dwa hasła do wyboru, każde z innej dziedziny
  assert.ok(warmup.length >= CATEGORY_CHOICES, `tylko ${warmup.length} pytań na rozgrzewkę`);
  assert.ok(
    new Set(warmup.map((q) => q.category)).size >= CATEGORY_CHOICES,
    'pytania rozgrzewkowe pochodzą ze zbyt małej liczby dziedzin',
  );

  // pozostałe rundy: na każdą po dwa hasła, więc z zapasem
  const needed = (ROUND_COUNT - 1) * CATEGORY_CHOICES;
  assert.ok(rest.length >= needed, `${rest.length} pytań na ${ROUND_COUNT - 1} rund, potrzeba ${needed}`);
  assert.ok(
    new Set(rest.map((q) => q.category)).size >= CATEGORY_CHOICES,
    'za mało dziedzin poza rozgrzewką',
  );
});

/**
 * Pytanie o rekord bez podanego stanu i źródła to pytanie z terminem ważności —
 * dokładnie ten błąd, przez który teleturniej musiał kiedyś tłumaczyć się
 * z najdłuższego metra poza Chinami. Z rokiem i wykazem pytanie przestaje
 * dotyczyć świata, a zaczyna dotyczyć konkretnego zestawienia.
 */
const SUPERLATIVE =
  /\b(rekord\w*|naj(wię|mniej|dłuż|krót|wyż|niż|star|młod|szyb|wolniej|częś|licz|głęb|gęst|bogat|popularniej|cieplej|zimniej|lepsz)\w*)/i;

test('pytania o rekordy podają stan i źródło albo powód, dla którego ich nie ma', () => {
  const records = QUESTIONS.filter((q) => SUPERLATIVE.test(q.text));
  assert.ok(records.length > 0, 'żadne pytanie nie pyta o rekord — regexp przestał działać?');

  for (const q of records) {
    if (q.timeless) {
      // wyjście awaryjne dla stopni najwyższych, które rekordem nie są:
      // składu powietrza ani progu „co najmniej” nikt nie pobije
      assert.ok(
        typeof q.timeless === 'string' && q.timeless.length > 10,
        `${q.id}: timeless musi tłumaczyć, dlaczego rok jest zbędny`,
      );
      assert.ok(!q.asOf && !q.source, `${q.id}: albo timeless, albo stan i źródło`);
      continue;
    }
    assert.ok(
      Number.isInteger(q.asOf) && q.asOf >= 2000,
      `${q.id}: pytanie o rekord bez roku (asOf)`,
    );
    assert.ok(
      typeof q.source === 'string' && q.source.length > 2,
      `${q.id}: pytanie o rekord bez źródła (source)`,
    );
  }
});

test('stan i źródło chodzą parą, także poza pytaniami o rekordy', () => {
  for (const q of QUESTIONS) {
    if (q.asOf || q.source) {
      assert.ok(q.asOf && q.source, `${q.id}: podany tylko jeden z dwóch — stan albo źródło`);
      assert.ok(q.asOf <= new Date().getFullYear(), `${q.id}: stan z przyszłości`);
    }
  }
});

test('stałe gry trzymają się razem', () => {
  assert.equal(DOORS_PLAN.length, ROUND_COUNT);
  assert.equal(ROUND_SECONDS.length, ROUND_COUNT);
  assert.equal(DOORS_PLAN.at(-1), FINAL_DOORS);
  assert.ok(
    DOORS_PLAN.every((d, i) => i === 0 || d <= DOORS_PLAN[i - 1]),
    'liczba zapadni nie może rosnąć w trakcie gry',
  );
  assert.equal(START_BALANCE % BUNDLE, 0);
  assert.ok(GRAB_SIZES.every((n) => Number.isInteger(n) && n > 0));
});

/* -------------------------------------------------------------- */
/* Wybór kategorii                                                */
/* -------------------------------------------------------------- */

test('gra zaczyna się od wyboru hasła, nie od pytania', () => {
  const state = newGame();
  assert.equal(state.status, 'choosing');
  assert.equal(state.question, null);
  assert.equal(offeredQuestions(state).length, CATEGORY_CHOICES);
});

test('oferta to dwa różne pytania, z różnych dziedzin', () => {
  for (const seed of ['a', 'b', 'c', 'd']) {
    const state = newGame(seed);
    const offer = offeredQuestions(state);
    const ids = offer.map((o) => o.id);
    assert.equal(new Set(ids).size, ids.length, 'to samo pytanie dwa razy');
    const topics = ids.map((id) => QUESTIONS.find((q) => q.id === id).category);
    assert.equal(new Set(topics).size, topics.length, 'oba hasła z tej samej dziedziny');
    for (const { label } of offer) {
      assert.ok(label && label.length > 0, 'hasło bez nazwy');
    }
  }
});

test('wskazane hasło odsłania dokładnie to pytanie, które się za nim kryło', () => {
  const state = newGame('haslo');
  const [, second] = offeredQuestions(state);
  const started = chooseOffer(state, second.id);
  assert.equal(started.status, 'placing');
  assert.equal(currentQuestion(started).id, second.id);
  assert.equal(started.bets.length, currentQuestion(started).answers.length);
  assert.equal(unplaced(started), START_BALANCE);
});

test('hasło nie zdradza dziedziny — temat wychodzi dopiero z pytaniem', () => {
  const withLabel = QUESTIONS.filter((q) => q.label);
  for (const q of withLabel) {
    assert.notEqual(
      q.label.toLowerCase(),
      q.category.toLowerCase(),
      `${q.id}: hasło jest po prostu nazwą dziedziny`,
    );
  }
});

test('rozgrzewką jest tylko pierwsza runda', () => {
  let state = newGame('rozgrzewka');
  assert.equal(isWarmupRound(state), true);
  state = enterRound(state);
  assert.equal(currentQuestion(state).difficulty, WARMUP_DIFFICULTY);

  for (let round = 1; round < 4; round++) {
    state = advance(resolve(placeRest(state, goodIndex(state))));
    assert.equal(isWarmupRound(state), false);
    state = enterRound(state);
    assert.notEqual(
      currentQuestion(state).difficulty,
      WARMUP_DIFFICULTY,
      `runda ${round + 1} dostała pytanie rozgrzewkowe`,
    );
  }
});

test('im dalej w grę, tym trudniejsze pytania', () => {
  // Gracz, który zna odpowiedź, kładzie wszystko na jedną zapadnię i nie traci
  // ani grosza — trudność pytania jest tu jedynym źródłem ryzyka.
  for (const seed of ['trud1', 'trud2', 'trud3', 'trud4']) {
    let state = newGame(seed);
    for (let round = 0; round < ROUND_COUNT; round++) {
      const floor = difficultyFloor(state);
      state = enterRound(state);
      assert.ok(
        currentQuestion(state).difficulty >= floor,
        `${seed}, runda ${round + 1}: pytanie o trudności ${currentQuestion(state).difficulty}, ` +
          `a próg wynosi ${floor}`,
      );
      state = resolve(placeRest(state, goodIndex(state)));
      if (round < ROUND_COUNT - 1) state = advance(state);
    }
  }
});

test('próg trudności ustępuje, gdy pula się kończy', () => {
  // Dwa pytania, oba łatwe, próg nie do przejścia — gra musi mimo to działać.
  const cienka = [
    {
      id: 'a',
      label: 'A',
      category: 'Jedna',
      difficulty: 2,
      text: 'Pierwsze?',
      answers: [
        { text: 'tak', correct: true },
        { text: 'nie', rival: true },
        { text: 'może' },
        { text: 'nigdy' },
      ],
      note: 'Wyjaśnienie na potrzeby testu.',
    },
    { id: 'b', label: 'B', category: 'Druga', difficulty: 2, text: 'Drugie?', answers: [
      { text: 'tak', correct: true },
      { text: 'nie', rival: true },
      { text: 'może' },
      { text: 'nigdy' },
    ], note: 'Wyjaśnienie na potrzeby testu.' },
  ];
  const state = createGame({ questions: cienka, seed: 'cienko', difficultyFloor: [5] });
  assert.equal(offeredQuestions(state).length, CATEGORY_CHOICES);
  assert.equal(chooseOffer(state, offeredQuestions(state)[0].id).status, 'placing');
});

test('liczba zapadni maleje zgodnie z planem', () => {
  let state = newGame('zapadnie');
  for (let round = 0; round < ROUND_COUNT; round++) {
    assert.equal(doorsInRound(state), DOORS_PLAN[round], `runda ${round + 1}`);
    state = enterRound(state);
    assert.equal(currentQuestion(state).answers.length, DOORS_PLAN[round]);
    state = resolve(placeRest(state, goodIndex(state)));
    if (round < ROUND_COUNT - 1) state = advance(state);
  }
});

test('przy zwężaniu pytania zostaje najpoważniejszy kontrkandydat', () => {
  const question = {
    id: 'x',
    answers: [
      { text: 'dobra', correct: true },
      { text: 'groźna', rival: true },
      { text: 'odsiew A' },
      { text: 'odsiew B' },
    ],
  };
  for (const doors of [3, 2]) {
    const trimmed = trimToDoors(question, makeRng('trim'), doors);
    const texts = trimmed.answers.map((a) => a.text);
    assert.equal(trimmed.answers.length, doors);
    assert.ok(texts.includes('dobra'), 'zginęła poprawna odpowiedź');
    assert.ok(texts.includes('groźna'), `przy ${doors} zapadniach zginął kontrkandydat`);
  }
});

test('pytania się nie powtarzają w obrębie partii', () => {
  let state = newGame('powtorki');
  const asked = [];
  for (let round = 0; round < ROUND_COUNT; round++) {
    state = enterRound(state);
    asked.push(currentQuestion(state).id);
    state = resolve(placeRest(state, goodIndex(state)));
    if (round < ROUND_COUNT - 1) state = advance(state);
  }
  assert.equal(new Set(asked).size, ROUND_COUNT);
  assert.deepEqual(state.usedIds, asked);
});

test('hasła nie da się wybrać w złym momencie ani spoza oferty', () => {
  const started = enterRound(newGame());
  assert.throws(() => chooseOffer(started, started.question.id), /przed pytaniem/);
  assert.throws(() => chooseOffer(newGame(), 'nie-ma-takiego'), /nie jest w tej rundzie/);
});

test('to samo ziarno daje ten sam przebieg, inne — inny', () => {
  const run = (seed) => {
    let state = createGame({ questions: QUESTIONS, seed });
    const ids = [];
    for (let r = 0; r < 4; r++) {
      state = enterRound(state);
      ids.push(currentQuestion(state).id);
      state = advance(resolve(placeRest(state, goodIndex(state))));
    }
    return ids;
  };
  assert.deepEqual(run('ziarno-1'), run('ziarno-1'));
  assert.notDeepEqual(run('ziarno-1'), run('ziarno-2'));
});

/* -------------------------------------------------------------- */
/* Finał                                                          */
/* -------------------------------------------------------------- */

test('finał ma dwie zapadnie, w tym poprawną', () => {
  let state = newGame('final');
  for (let round = 0; round < ROUND_COUNT - 1; round++) {
    state = advance(resolve(placeRest(enterRound(state), goodIndex(enterRound(state)))));
  }
  assert.equal(isFinalRound(state), true);
  assert.equal(doorsInRound(state), FINAL_DOORS);
  state = enterRound(state);
  assert.equal(currentQuestion(state).answers.length, FINAL_DOORS);
  assert.equal(currentQuestion(state).answers.filter((a) => a.correct).length, 1);
  assert.equal(currentQuestion(state).isFinal, true);
});

test('w finale nie da się rozdzielić puli na obie zapadnie', () => {
  let state = newGame('finalowa');
  for (let round = 0; round < ROUND_COUNT - 1; round++) {
    const started = enterRound(state);
    state = advance(resolve(placeRest(started, goodIndex(started))));
  }
  state = placeBundles(enterRound(state), 0, 20);
  assert.equal(canPlaceOn(state, 1), false);
  assert.equal(canLock(state), false, 'reszta wciąż jest w rękach');
  assert.equal(canLock(placeRest(state, 0)), true);
});

/* -------------------------------------------------------------- */
/* Kładzenie paczek                                               */
/* -------------------------------------------------------------- */

test('runda startuje z pełną pulą w rękach', () => {
  const state = enterRound(newGame());
  assert.equal(state.balance, START_BALANCE);
  assert.equal(unplacedBundles(state), 40);
  assert.equal(emptyDoors(state), 4);
});

test('kłaść można tylko całe paczki i tylko tyle, ile się ma', () => {
  const state = enterRound(newGame());
  assert.throws(() => place(state, 0, START_BALANCE + BUNDLE), /Nie masz tylu pieniędzy/);
  assert.throws(() => place(state, 0, 10_000), /całe paczki/);
  assert.throws(() => place(state, 0, 0), /dodatnia/);
  assert.throws(() => place(state, 9, BUNDLE), /Nie ma takiej zapadni/);
});

test('place nie modyfikuje poprzedniego stanu', () => {
  const before = enterRound(newGame());
  const after = place(before, 0, 2 * BUNDLE);
  assert.equal(before.bets[0], 0);
  assert.equal(after.bets[0], 2 * BUNDLE);
});

test('placeBundles kładzie tyle paczek, ile zostało', () => {
  let state = placeBundles(enterRound(newGame()), 0, 10);
  assert.equal(state.bets[0], 10 * BUNDLE);
  state = placeBundles(state, 1, 100);
  assert.equal(unplaced(state), 0);
  assert.equal(state.bets[1], 30 * BUNDLE);
});

test('ostatniej pustej zapadni nie da się obstawić', () => {
  let state = enterRound(newGame());
  state = placeBundles(state, 0, 10);
  state = placeBundles(state, 1, 10);
  assert.equal(canPlaceOn(state, 2), true);

  state = placeBundles(state, 2, 10);
  assert.equal(emptyDoors(state), 1);
  assert.equal(canPlaceOn(state, 3), false, 'ostatnie puste pole jest zablokowane');
  assert.equal(canPlaceOn(state, 0), true, 'dokładać na zajęte pola wolno');
  assert.throws(() => place(state, 3, BUNDLE), /musi zostać pusta/);

  state = placeBundles(state, 0, 10);
  assert.equal(canLock(state), true);
});

test('zdjęcie paczek odblokowuje pole, które było zablokowane', () => {
  let state = enterRound(newGame());
  state = placeBundles(state, 0, 10);
  state = placeBundles(state, 1, 10);
  state = placeBundles(state, 2, 20);
  assert.equal(canPlaceOn(state, 3), false);
  state = take(state, 2, 20 * BUNDLE);
  assert.equal(canPlaceOn(state, 3), true);
});

test('undo cofa ostatni ruch, clearBets zdejmuje wszystko', () => {
  let state = enterRound(newGame());
  state = placeBundles(state, 0, 4);
  state = placeBundles(state, 1, 20);
  state = undo(state);
  assert.equal(state.bets[1], 0);
  assert.equal(state.bets[0], 4 * BUNDLE);
  state = clearBets(state);
  assert.equal(unplaced(state), START_BALANCE);
  assert.equal(undo(state), state, 'undo bez ruchów zwraca ten sam stan');
});

/* -------------------------------------------------------------- */
/* Otwieranie zapadni                                             */
/* -------------------------------------------------------------- */

test('zapadnie otwierają się po kolei: puste, potem rosnące stawki, poprawna na końcu', () => {
  let state = enterRound(newGame('kolejnosc'));
  const good = goodIndex(state);
  const wrong = [0, 1, 2, 3].filter((i) => i !== good);

  // duża stawka na pierwsze błędne pole, mała na drugie, trzecie puste
  state = placeBundles(state, wrong[1], 4);
  state = placeRest(state, wrong[0]);

  const order = revealOrder(state);
  assert.equal(order.length, 4);
  assert.equal(order.at(-1), good, 'poprawna zapadnia idzie na koniec');
  assert.equal(order[0], wrong[2], 'najpierw pole puste');
  assert.equal(order[1], wrong[1], 'potem mniejsza stawka');
  assert.equal(order[2], wrong[0], 'na koniec największa strata');
});

test('kolejność otwierania obejmuje każdą zapadnię dokładnie raz', () => {
  let state = enterRound(newGame('pokrycie'));
  state = placeRest(state, goodIndex(state));
  const order = revealOrder(state);
  assert.deepEqual([...order].sort((a, b) => a - b), [0, 1, 2, 3]);
});

test('zostaje tylko to, co leżało na poprawnej zapadni', () => {
  let state = enterRound(newGame());
  const good = goodIndex(state);
  const bad = badIndex(state);

  state = placeBundles(state, good, 12);
  state = placeRest(state, bad);
  state = resolve(state);

  assert.equal(state.status, 'revealed');
  assert.equal(state.outcome, 'continue');
  assert.equal(state.balance, 12 * BUNDLE);
  assert.equal(state.result.dropped, 28 * BUNDLE);
  assert.equal(state.result.forfeited, 0);
});

test('paczki trzymane w rękach lecą w dół razem z resztą', () => {
  let state = enterRound(newGame());
  state = resolve(placeBundles(state, goodIndex(state), 4));
  assert.equal(state.balance, 4 * BUNDLE);
  assert.equal(state.result.forfeited, 36 * BUNDLE);
});

test('utrata całej puli kończy grę porażką', () => {
  let state = enterRound(newGame());
  state = resolve(placeRest(state, badIndex(state)));
  assert.equal(state.outcome, 'lost');
  assert.equal(state.balance, 0);
  assert.equal(advance(state).status, 'over');
});

test('przejście przez wszystkie rundy z pełną pulą daje milion', () => {
  let state = newGame('milion');
  for (let round = 0; round < ROUND_COUNT; round++) {
    assert.equal(state.roundIndex, round);
    assert.equal(state.status, 'choosing');
    state = enterRound(state);
    assert.equal(emptyDoors(state), doorsInRound(state));
    state = resolve(placeRest(state, goodIndex(state)));
    if (round < ROUND_COUNT - 1) {
      assert.equal(state.outcome, 'continue');
      state = advance(state);
    }
  }
  assert.equal(state.outcome, 'won');
  assert.equal(state.balance, START_BALANCE);
  assert.equal(state.history.length, ROUND_COUNT);
  assert.ok(state.history.every((row) => row.category), 'historia zapamiętuje kategorię rundy');
  assert.equal(advance(state).status, 'over');
});

test('po otwarciu zapadni nie można ruszać paczek', () => {
  let state = enterRound(newGame());
  state = resolve(placeRest(state, goodIndex(state)));
  assert.throws(() => place(state, 0, BUNDLE), /tylko w trakcie rundy/);
  assert.throws(() => clearBets(state), /tylko w trakcie rundy/);
  assert.throws(() => resolve(state), /tylko w trakcie rundy/);
  assert.equal(canLock(state), false);
  assert.equal(canPlaceOn(state, 0), false);
});

test('nowa runda wraca do wyboru kategorii i czyści stół', () => {
  let state = enterRound(newGame('runda2'));
  state = advance(resolve(placeRest(state, goodIndex(state))));
  assert.equal(state.roundIndex, 1);
  assert.equal(state.status, 'choosing');
  assert.equal(state.question, null);
  assert.deepEqual(state.bets, []);
  assert.equal(state.moves.length, 0);
});
