import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DENOMINATIONS,
  DIFFICULTY_PLAN,
  ROUND_COUNT,
  ROUND_SECONDS,
  START_BALANCE,
  advance,
  availableChips,
  clearBets,
  createGame,
  currentQuestion,
  makeRng,
  place,
  placeRest,
  resolve,
  selectQuestions,
  take,
  undo,
  unplaced,
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

test('każde pytanie ma co najmniej jedną poprawną i jedną błędną odpowiedź', () => {
  for (const q of QUESTIONS) {
    const correct = q.answers.filter((a) => a.correct).length;
    assert.ok(correct >= 1, `${q.id}: brak poprawnej odpowiedzi`);
    assert.ok(correct < q.answers.length, `${q.id}: wszystkie odpowiedzi poprawne`);
    assert.ok(q.answers.length >= 3, `${q.id}: za mało odpowiedzi`);
    assert.ok(q.difficulty >= 1 && q.difficulty <= 5, `${q.id}: zły poziom trudności`);
  }
});

test('dla każdego poziomu trudności wystarcza pytań na plan rundy', () => {
  for (const level of new Set(DIFFICULTY_PLAN)) {
    const needed = DIFFICULTY_PLAN.filter((d) => d === level).length;
    const have = QUESTIONS.filter((q) => q.difficulty === level).length;
    assert.ok(have >= needed, `poziom ${level}: ${have} pytań, potrzeba ${needed}`);
  }
});

/* -------------------------------------------------------------- */
/* Dobór pytań                                                    */
/* -------------------------------------------------------------- */

test('gra dobiera 10 pytań o rosnącej trudności, bez powtórek', () => {
  const state = newGame('abc');
  assert.equal(state.questions.length, ROUND_COUNT);
  assert.deepEqual(
    state.questions.map((q) => q.difficulty),
    DIFFICULTY_PLAN,
  );
  const ids = state.questions.map((q) => q.id);
  assert.equal(new Set(ids).size, ROUND_COUNT);
});

test('to samo ziarno daje ten sam zestaw pytań, inne — inny', () => {
  const a = newGame('ziarno-1').questions.map((q) => q.id);
  const b = newGame('ziarno-1').questions.map((q) => q.id);
  const c = newGame('ziarno-2').questions.map((q) => q.id);
  assert.deepEqual(a, b);
  assert.notDeepEqual(a, c);
});

test('kolejność odpowiedzi jest losowana, ale zbiór pozostaje ten sam', () => {
  const rng = makeRng('mieszanie');
  const picked = selectQuestions(QUESTIONS, rng);
  for (const q of picked) {
    const original = QUESTIONS.find((o) => o.id === q.id);
    assert.deepEqual(
      q.answers.map((a) => a.text).sort(),
      original.answers.map((a) => a.text).sort(),
    );
  }
});

/* -------------------------------------------------------------- */
/* Układanie żetonów                                              */
/* -------------------------------------------------------------- */

test('nowa gra startuje z pełną pulą i pustymi polami', () => {
  const state = newGame();
  assert.equal(state.balance, START_BALANCE);
  assert.equal(unplaced(state), START_BALANCE);
  assert.deepEqual(state.bets, state.bets.map(() => 0));
  assert.equal(ROUND_SECONDS.length, ROUND_COUNT);
});

test('nie da się postawić więcej, niż się ma', () => {
  const state = newGame();
  assert.throws(() => place(state, 0, START_BALANCE + 1000), /Nie masz tylu pieniędzy/);
  assert.throws(() => place(state, 0, 0), /dodatnia/);
  assert.throws(() => place(state, 99, 1000), /Nie ma takiej odpowiedzi/);
});

test('place nie modyfikuje poprzedniego stanu', () => {
  const before = newGame();
  const after = place(before, 0, 50_000);
  assert.equal(before.bets[0], 0);
  assert.equal(after.bets[0], 50_000);
  assert.equal(unplaced(after), START_BALANCE - 50_000);
});

test('placeRest dokłada dokładnie resztę', () => {
  let state = newGame();
  state = place(state, 0, 250_000);
  state = placeRest(state, 1);
  assert.equal(unplaced(state), 0);
  assert.equal(state.bets[1], START_BALANCE - 250_000);
});

test('take zdejmuje żetony, ale nie schodzi poniżej zera', () => {
  let state = place(newGame(), 0, 10_000);
  state = take(state, 0, 50_000);
  assert.equal(state.bets[0], 0);
  assert.equal(unplaced(state), START_BALANCE);
});

test('undo cofa ostatni ruch, clearBets czyści wszystko', () => {
  let state = newGame();
  state = place(state, 0, 100_000);
  state = place(state, 1, 500_000);
  state = undo(state);
  assert.equal(state.bets[1], 0);
  assert.equal(state.bets[0], 100_000);
  state = clearBets(state);
  assert.equal(unplaced(state), START_BALANCE);
  assert.equal(state.moves.length, 0);
  assert.equal(undo(state), state, 'undo bez ruchów zwraca ten sam stan');
});

test('lista dostępnych nominałów kurczy się wraz z pulą', () => {
  let state = newGame();
  assert.deepEqual(availableChips(state), DENOMINATIONS);
  state = place(state, 0, 995_000);
  assert.deepEqual(availableChips(state), [1_000, 5_000]);
});

test('nominały pozwalają rozłożyć dowolną wielokrotność 1000', () => {
  for (const amount of [1_000, 7_000, 123_000, 999_000, 1_000_000]) {
    let left = amount;
    for (const d of [...DENOMINATIONS].sort((a, b) => b - a)) {
      left -= Math.floor(left / d) * d;
    }
    assert.equal(left, 0, `nie da się rozłożyć ${amount}`);
  }
});

/* -------------------------------------------------------------- */
/* Rozstrzyganie rundy                                            */
/* -------------------------------------------------------------- */

function betAllOnCorrect(state) {
  const idx = currentQuestion(state).answers.findIndex((a) => a.correct);
  return placeRest(state, idx);
}

function betAllOnWrong(state) {
  const idx = currentQuestion(state).answers.findIndex((a) => !a.correct);
  return placeRest(state, idx);
}

test('zostaje tylko to, co leży na poprawnej odpowiedzi', () => {
  let state = newGame();
  const q = currentQuestion(state);
  const good = q.answers.findIndex((a) => a.correct);
  const bad = q.answers.findIndex((a) => !a.correct);

  state = place(state, good, 300_000);
  state = placeRest(state, bad);
  state = resolve(state);

  assert.equal(state.status, 'revealed');
  assert.equal(state.outcome, 'continue');
  assert.equal(state.balance, 300_000);
  assert.equal(state.result.kept, 300_000);
  assert.equal(state.result.lost, 700_000);
  assert.equal(state.result.forfeited, 0);
});

test('pieniądze nierozłożone przepadają (np. po upływie czasu)', () => {
  let state = newGame();
  const good = currentQuestion(state).answers.findIndex((a) => a.correct);
  state = resolve(place(state, good, 100_000));
  assert.equal(state.balance, 100_000);
  assert.equal(state.result.forfeited, 900_000);
  assert.equal(state.result.lost, 900_000);
});

test('pytanie z kilkoma poprawnymi odpowiedziami sumuje stawki', () => {
  const multi = QUESTIONS.find((q) => q.answers.filter((a) => a.correct).length > 1);
  let state = createGame({ questions: [multi], seed: 'multi', plan: [multi.difficulty] });
  const correct = state.questions[0].answers
    .map((a, i) => (a.correct ? i : -1))
    .filter((i) => i >= 0);
  state = place(state, correct[0], 200_000);
  state = place(state, correct[1], 300_000);
  state = placeRest(state, state.questions[0].answers.findIndex((a) => !a.correct));
  state = resolve(state);
  assert.equal(state.balance, 500_000);
});

test('utrata całej puli kończy grę porażką', () => {
  let state = resolve(betAllOnWrong(newGame()));
  assert.equal(state.outcome, 'lost');
  assert.equal(state.balance, 0);
  state = advance(state);
  assert.equal(state.status, 'over');
});

test('przejście przez 10 rund z pełną pulą daje milion', () => {
  let state = newGame('milion');
  for (let round = 0; round < ROUND_COUNT; round++) {
    assert.equal(state.roundIndex, round);
    assert.equal(state.status, 'placing');
    state = resolve(betAllOnCorrect(state));
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

test('gra kończy się po 10 rundzie także z częściową wygraną', () => {
  let state = newGame('czesciowa');
  for (let round = 0; round < ROUND_COUNT - 1; round++) {
    state = advance(resolve(betAllOnCorrect(state)));
  }
  const q = currentQuestion(state);
  state = place(state, q.answers.findIndex((a) => a.correct), 400_000);
  state = placeRest(state, q.answers.findIndex((a) => !a.correct));
  state = resolve(state);
  assert.equal(state.outcome, 'won');
  assert.equal(state.balance, 400_000);
});

test('po odsłonięciu odpowiedzi nie można ruszać żetonów', () => {
  const state = resolve(betAllOnCorrect(newGame()));
  assert.throws(() => place(state, 0, 1000), /tylko w trakcie rundy/);
  assert.throws(() => clearBets(state), /tylko w trakcie rundy/);
  assert.throws(() => resolve(state), /tylko w trakcie rundy/);
});

test('nowa runda zeruje stawki i dopasowuje liczbę pól do pytania', () => {
  let state = advance(resolve(betAllOnCorrect(newGame('runda2'))));
  assert.equal(state.roundIndex, 1);
  assert.equal(state.status, 'placing');
  assert.equal(state.bets.length, currentQuestion(state).answers.length);
  assert.equal(unplaced(state), state.balance);
});
