import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BUNDLE,
  CATEGORY_CHOICES,
  DIFFICULTY_PLAN,
  FINAL_DOORS,
  GRAB_SIZES,
  ROUND_COUNT,
  ROUND_SECONDS,
  START_BALANCE,
  advance,
  availableQuestions,
  canLock,
  canPlaceOn,
  chooseCategory,
  clearBets,
  createGame,
  currentQuestion,
  difficultyFor,
  doorsInRound,
  emptyDoors,
  isFinalRound,
  offeredCategories,
  place,
  placeBundles,
  placeRest,
  resolve,
  revealOrder,
  take,
  undo,
  unplaced,
  unplacedBundles,
} from '../src/engine.js';
import { QUESTIONS } from '../src/questions.js';

const newGame = (seed = 'test') => createGame({ questions: QUESTIONS, seed });

/** Rozpoczyna rundę, biorąc pierwszą z proponowanych kategorii. */
const enterRound = (state) => chooseCategory(state, offeredCategories(state)[0]);

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

test('na każdym poziomie trudności starczy pytań i kategorii na cały plan', () => {
  for (const level of new Set(DIFFICULTY_PLAN)) {
    const rounds = DIFFICULTY_PLAN.filter((d) => d === level).length;
    const batch = QUESTIONS.filter((q) => q.difficulty === level);
    const categories = new Set(batch.map((q) => q.category));
    assert.ok(batch.length >= rounds, `poziom ${level}: ${batch.length} pytań, potrzeba ${rounds}`);
    // po zużyciu pytań z wcześniejszych rund wciąż musi zostać z czego wybierać
    assert.ok(
      categories.size >= CATEGORY_CHOICES + rounds - 1,
      `poziom ${level}: tylko ${categories.size} kategorii — za mało na wybór w ${rounds} rundach`,
    );
  }
});

test('stałe gry trzymają się razem', () => {
  assert.equal(DIFFICULTY_PLAN.length, ROUND_COUNT);
  assert.equal(ROUND_SECONDS.length, ROUND_COUNT);
  assert.equal(START_BALANCE % BUNDLE, 0);
  assert.ok(GRAB_SIZES.every((n) => Number.isInteger(n) && n > 0));
});

/* -------------------------------------------------------------- */
/* Wybór kategorii                                                */
/* -------------------------------------------------------------- */

test('gra zaczyna się od wyboru kategorii, nie od pytania', () => {
  const state = newGame();
  assert.equal(state.status, 'choosing');
  assert.equal(state.question, null);
  assert.equal(offeredCategories(state).length, CATEGORY_CHOICES);
});

test('proponowane kategorie są różne i mają pokrycie w pytaniach', () => {
  for (const seed of ['a', 'b', 'c', 'd']) {
    const state = newGame(seed);
    const offer = offeredCategories(state);
    assert.equal(new Set(offer).size, offer.length, 'kategorie się powtarzają');
    for (const category of offer) {
      assert.ok(
        availableQuestions(state).some((q) => q.category === category),
        `${category}: brak pytań`,
      );
    }
  }
});

test('wybór kategorii daje pytanie właśnie z niej', () => {
  const state = newGame('kategoria');
  const [, second] = offeredCategories(state);
  const started = chooseCategory(state, second);
  assert.equal(started.status, 'placing');
  assert.equal(currentQuestion(started).category, second);
  assert.equal(started.bets.length, currentQuestion(started).answers.length);
  assert.equal(unplaced(started), START_BALANCE);
});

test('pytanie o właściwym poziomie trudności dla rundy', () => {
  let state = newGame('trudnosc');
  for (let round = 0; round < 3; round++) {
    state = enterRound(state);
    assert.equal(currentQuestion(state).difficulty, difficultyFor(round));
    state = advance(resolve(placeRest(state, goodIndex(state))));
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

test('kategorii nie da się wybrać w złym momencie ani spoza oferty', () => {
  const state = enterRound(newGame());
  assert.throws(() => chooseCategory(state, 'Geografia'), /przed pytaniem/);
  assert.throws(() => chooseCategory(newGame(), 'Nie ma takiej'), /Brak pytań/);
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
