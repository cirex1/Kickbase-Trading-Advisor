import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BUNDLE,
  DIFFICULTY_PLAN,
  FINAL_DOORS,
  GRAB_SIZES,
  ROUND_COUNT,
  ROUND_SECONDS,
  START_BALANCE,
  advance,
  canLock,
  canPlaceOn,
  clearBets,
  createGame,
  currentQuestion,
  emptyDoors,
  makeRng,
  place,
  placeBundles,
  placeRest,
  resolve,
  selectQuestions,
  take,
  undo,
  unplaced,
  unplacedBundles,
} from '../src/engine.js';
import { QUESTIONS } from '../src/questions.js';

const newGame = (seed = 'test') => createGame({ questions: QUESTIONS, seed });

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

test('dla każdego poziomu trudności wystarcza pytań na plan rundy', () => {
  for (const level of new Set(DIFFICULTY_PLAN)) {
    const needed = DIFFICULTY_PLAN.filter((d) => d === level).length;
    const have = QUESTIONS.filter((q) => q.difficulty === level).length;
    assert.ok(have >= needed, `poziom ${level}: ${have} pytań, potrzeba ${needed}`);
  }
});

test('stałe gry trzymają się razem', () => {
  assert.equal(DIFFICULTY_PLAN.length, ROUND_COUNT);
  assert.equal(ROUND_SECONDS.length, ROUND_COUNT);
  assert.equal(START_BALANCE % BUNDLE, 0);
  assert.ok(GRAB_SIZES.every((n) => Number.isInteger(n) && n > 0));
});

/* -------------------------------------------------------------- */
/* Dobór pytań                                                    */
/* -------------------------------------------------------------- */

test('gra dobiera 8 pytań o rosnącej trudności, bez powtórek', () => {
  const state = newGame('abc');
  assert.equal(state.questions.length, ROUND_COUNT);
  assert.deepEqual(
    state.questions.map((q) => q.difficulty),
    DIFFICULTY_PLAN,
  );
  assert.equal(new Set(state.questions.map((q) => q.id)).size, ROUND_COUNT);
});

test('finał ma tylko dwie zapadnie, w tym poprawną', () => {
  const last = newGame('final').questions[ROUND_COUNT - 1];
  assert.equal(last.answers.length, FINAL_DOORS);
  assert.equal(last.answers.filter((a) => a.correct).length, 1);
  assert.equal(last.isFinal, true);
});

test('poprawna odpowiedź w finale nie ląduje zawsze po tej samej stronie', () => {
  const sides = new Set();
  for (const seed of ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']) {
    const last = newGame(seed).questions[ROUND_COUNT - 1];
    sides.add(last.answers.findIndex((a) => a.correct));
  }
  assert.equal(sides.size, 2);
});

test('to samo ziarno daje ten sam zestaw pytań, inne — inny', () => {
  const a = newGame('ziarno-1').questions.map((q) => q.id);
  const b = newGame('ziarno-1').questions.map((q) => q.id);
  const c = newGame('ziarno-2').questions.map((q) => q.id);
  assert.deepEqual(a, b);
  assert.notDeepEqual(a, c);
});

test('kolejność odpowiedzi jest losowana, ale zbiór pozostaje ten sam', () => {
  const picked = selectQuestions(QUESTIONS, makeRng('mieszanie'));
  for (const q of picked) {
    const original = QUESTIONS.find((o) => o.id === q.id);
    assert.deepEqual(
      q.answers.map((a) => a.text).sort(),
      original.answers.map((a) => a.text).sort(),
    );
  }
});

/* -------------------------------------------------------------- */
/* Kładzenie paczek                                               */
/* -------------------------------------------------------------- */

test('nowa gra startuje z pełną pulą w rękach', () => {
  const state = newGame();
  assert.equal(state.balance, START_BALANCE);
  assert.equal(unplaced(state), START_BALANCE);
  assert.equal(unplacedBundles(state), 40);
  assert.equal(emptyDoors(state), 4);
});

test('kłaść można tylko całe paczki i tylko tyle, ile się ma', () => {
  const state = newGame();
  assert.throws(() => place(state, 0, START_BALANCE + BUNDLE), /Nie masz tylu pieniędzy/);
  assert.throws(() => place(state, 0, 10_000), /całe paczki/);
  assert.throws(() => place(state, 0, 0), /dodatnia/);
  assert.throws(() => place(state, 9, BUNDLE), /Nie ma takiej zapadni/);
});

test('place nie modyfikuje poprzedniego stanu', () => {
  const before = newGame();
  const after = place(before, 0, 2 * BUNDLE);
  assert.equal(before.bets[0], 0);
  assert.equal(after.bets[0], 2 * BUNDLE);
  assert.equal(unplaced(after), START_BALANCE - 2 * BUNDLE);
});

test('placeBundles kładzie tyle paczek, ile zostało', () => {
  let state = newGame();
  state = placeBundles(state, 0, 10);
  assert.equal(state.bets[0], 10 * BUNDLE);
  state = placeBundles(state, 1, 100); // więcej, niż jest w rękach
  assert.equal(unplaced(state), 0);
  assert.equal(state.bets[1], 30 * BUNDLE);
});

/* --- zasada: jedna zapadnia musi zostać pusta -------------------------- */

test('ostatniej pustej zapadni nie da się obstawić', () => {
  let state = newGame();
  state = placeBundles(state, 0, 10);
  state = placeBundles(state, 1, 10);
  assert.equal(emptyDoors(state), 2);
  assert.equal(canPlaceOn(state, 2), true);

  state = placeBundles(state, 2, 10);
  assert.equal(emptyDoors(state), 1);
  assert.equal(canPlaceOn(state, 3), false, 'ostatnie puste pole jest zablokowane');
  assert.equal(canPlaceOn(state, 0), true, 'dokładać na zajęte pola wolno');
  assert.throws(() => place(state, 3, BUNDLE), /musi zostać pusta/);

  state = placeBundles(state, 0, 10);
  assert.equal(unplaced(state), 0);
  assert.equal(canLock(state), true);
});

test('gra nie pozwala zatwierdzić, dopóki paczki są w rękach', () => {
  let state = placeBundles(newGame(), 0, 10);
  assert.equal(canLock(state), false);
  state = placeRest(state, 1);
  assert.equal(canLock(state), true);
});

test('zdjęcie paczek odblokowuje pole, które było zablokowane', () => {
  let state = newGame();
  state = placeBundles(state, 0, 10);
  state = placeBundles(state, 1, 10);
  state = placeBundles(state, 2, 20);
  assert.equal(canPlaceOn(state, 3), false);
  state = take(state, 2, 20 * BUNDLE);
  assert.equal(emptyDoors(state), 2);
  assert.equal(canPlaceOn(state, 3), true);
});

test('take nie schodzi poniżej zera', () => {
  let state = placeBundles(newGame(), 0, 1);
  state = take(state, 0, 50 * BUNDLE);
  assert.equal(state.bets[0], 0);
  assert.equal(unplaced(state), START_BALANCE);
});

test('undo cofa ostatni ruch, clearBets zdejmuje wszystko', () => {
  let state = newGame();
  state = placeBundles(state, 0, 4);
  state = placeBundles(state, 1, 20);
  state = undo(state);
  assert.equal(state.bets[1], 0);
  assert.equal(state.bets[0], 4 * BUNDLE);
  state = clearBets(state);
  assert.equal(unplaced(state), START_BALANCE);
  assert.equal(state.moves.length, 0);
  assert.equal(undo(state), state, 'undo bez ruchów zwraca ten sam stan');
});

/* -------------------------------------------------------------- */
/* Otwieranie zapadni                                             */
/* -------------------------------------------------------------- */

function goodIndex(state) {
  return currentQuestion(state).answers.findIndex((a) => a.correct);
}

function badIndex(state, skip = -1) {
  return currentQuestion(state).answers.findIndex((a, i) => !a.correct && i !== skip);
}

test('zostaje tylko to, co leżało na poprawnej zapadni', () => {
  let state = newGame();
  const good = goodIndex(state);
  const bad = badIndex(state);

  state = placeBundles(state, good, 12);
  state = placeRest(state, bad);
  state = resolve(state);

  assert.equal(state.status, 'revealed');
  assert.equal(state.outcome, 'continue');
  assert.equal(state.balance, 12 * BUNDLE);
  assert.equal(state.result.kept, 12 * BUNDLE);
  assert.equal(state.result.dropped, 28 * BUNDLE);
  assert.equal(state.result.forfeited, 0);
});

test('paczki trzymane w rękach lecą w dół razem z resztą', () => {
  let state = newGame();
  state = resolve(placeBundles(state, goodIndex(state), 4));
  assert.equal(state.balance, 4 * BUNDLE);
  assert.equal(state.result.forfeited, 36 * BUNDLE);
  assert.equal(state.result.dropped, 36 * BUNDLE);
});

test('utrata całej puli kończy grę porażką', () => {
  let state = newGame();
  state = resolve(placeRest(state, badIndex(state)));
  assert.equal(state.outcome, 'lost');
  assert.equal(state.balance, 0);
  assert.equal(advance(state).status, 'over');
});

test('przejście przez wszystkie rundy z pełną pulą daje milion', () => {
  let state = newGame('milion');
  for (let round = 0; round < ROUND_COUNT; round++) {
    assert.equal(state.roundIndex, round);
    assert.equal(state.status, 'placing');
    assert.equal(emptyDoors(state), currentQuestion(state).answers.length);
    state = resolve(placeRest(state, goodIndex(state)));
    if (round < ROUND_COUNT - 1) {
      assert.equal(state.outcome, 'continue');
      state = advance(state);
    }
  }
  assert.equal(state.outcome, 'won');
  assert.equal(state.balance, START_BALANCE);
  assert.equal(state.history.length, ROUND_COUNT);
  assert.equal(advance(state).status, 'over');
});

test('gra kończy się po ostatniej rundzie także z częściową wygraną', () => {
  let state = newGame('czesciowa');
  for (let round = 0; round < ROUND_COUNT - 1; round++) {
    state = advance(resolve(placeRest(state, goodIndex(state))));
  }
  // finał: dwie zapadnie, więc wszystko musi wylądować na jednej
  assert.equal(state.bets.length, FINAL_DOORS);
  state = resolve(placeRest(state, goodIndex(state)));
  assert.equal(state.outcome, 'won');
  assert.equal(state.balance, START_BALANCE);
});

test('w finale nie da się rozdzielić puli na obie zapadnie', () => {
  let state = newGame('finalowa');
  for (let round = 0; round < ROUND_COUNT - 1; round++) {
    state = advance(resolve(placeRest(state, goodIndex(state))));
  }
  state = placeBundles(state, 0, 20);
  assert.equal(canPlaceOn(state, 1), false);
  assert.equal(canLock(state), false, 'reszta wciąż jest w rękach');
  state = placeRest(state, 0);
  assert.equal(canLock(state), true);
});

test('po otwarciu zapadni nie można ruszać paczek', () => {
  const state = resolve(placeRest(newGame(), goodIndex(newGame())));
  assert.throws(() => place(state, 0, BUNDLE), /tylko w trakcie rundy/);
  assert.throws(() => clearBets(state), /tylko w trakcie rundy/);
  assert.throws(() => resolve(state), /tylko w trakcie rundy/);
  assert.equal(canLock(state), false);
  assert.equal(canPlaceOn(state, 0), false);
});

test('nowa runda zeruje stawki i dopasowuje liczbę zapadni do pytania', () => {
  let state = newGame('runda2');
  state = advance(resolve(placeRest(state, goodIndex(state))));
  assert.equal(state.roundIndex, 1);
  assert.equal(state.status, 'placing');
  assert.equal(state.bets.length, currentQuestion(state).answers.length);
  assert.equal(unplaced(state), state.balance);
  assert.equal(state.moves.length, 0);
});
