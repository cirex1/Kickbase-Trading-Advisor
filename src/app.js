/**
 * Warstwa prezentacji: łączy silnik gry (src/engine.js) ze sceną w DOM-ie.
 *
 * Scena składa się z dwóch warstw ułożonych na tej samej siatce kolumn:
 *   .floor  — zapadnie odchylone w 3D (perspektywa studia),
 *   .lanes  — ekrany z odpowiedziami i stosy paczek, w płaszczyźnie ekranu.
 * Dzięki temu spadające paczki animujemy zwykłym translateY, bez walki
 * z układem współrzędnych obróconej podłogi.
 *
 * Runda ma trzy odsłony: plansza z kategoriami, pytanie z rozkładaniem paczek
 * i otwieranie zapadni — jedna po drugiej, w kolejności, którą wyznacza silnik.
 */

import {
  BUNDLE,
  GRAB_SIZES,
  ROUND_COUNT,
  START_BALANCE,
  advance,
  canLock,
  canPlaceOn,
  chooseCategory,
  clearBets,
  createGame,
  currentQuestion,
  doorsInRound,
  offeredCategories,
  placeBundles,
  placeRest,
  resolve,
  revealOrder,
  roundSeconds,
  take,
  undo,
  unplaced,
  unplacedBundles,
} from './engine.js';
import { QUESTIONS } from './questions.js';
import { isSoundOn, sfx, toggleSound } from './audio.js';
import { burst } from './confetti.js';
import { readSetting, writeSetting } from './storage.js';

const LETTERS = ['A', 'B', 'C', 'D', 'E', 'F'];
const BEST_KEY = 'pnm.best';

/** Maksymalna wysokość stosu w pikselach — na zapadni i w rękach. */
const STACK_HEIGHT = 104;
const HAND_STACK_HEIGHT = 34;

/** Rytm otwierania zapadni (ms). */
const BEAT = {
  bumper: 1500,
  tension: 1500, // cisza po zatwierdzeniu
  spotlight: 420, // podświetlenie pola tuż przed otwarciem
  emptyDoor: 900, // pusta zapadnia — otwiera się i nic nie spada
  loadedDoor: 1800, // zapadnia z pieniędzmi — paczki muszą dolecieć
  finish: 1300, // od poprawnej zapadni do podsumowania
};

const money = new Intl.NumberFormat('pl-PL', { maximumFractionDigits: 0 });
const zl = (value) => `${money.format(Math.round(value))} zł`;
const paczki = (n) =>
  `${n} ${n === 1 ? 'paczka' : n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 12 || n % 100 > 14) ? 'paczki' : 'paczek'}`;

const $ = (id) => document.getElementById(id);

const el = {
  screens: { start: $('screen-start'), game: $('screen-game'), end: $('screen-end') },
  best: $('best-score'),
  seedInput: $('seed-input'),
  btnStart: $('btn-start'),
  btnSound: $('btn-sound'),
  btnRules: $('btn-rules'),
  rulesDialog: $('rules-dialog'),
  btnRulesClose: $('btn-rules-close'),

  roundLabel: $('round-label'),
  balance: $('balance'),
  timerValue: $('timer-value'),
  timerRing: $('timer-ring'),
  timer: $('timer'),

  choice: $('choice'),
  choiceCards: $('choice-cards'),
  question: $('question'),
  category: $('q-category'),
  questionText: $('q-text'),
  finalHint: $('q-final'),

  stage: $('stage'),
  floor: $('floor'),
  lanes: $('lanes'),
  bumper: $('bumper'),
  bumperRound: $('bumper-round'),
  bumperStake: $('bumper-stake'),

  tray: $('tray'),
  unplaced: $('unplaced'),
  unplacedCount: $('unplaced-count'),
  handStack: $('hand-stack'),
  grabs: $('grabs'),
  btnUndo: $('btn-undo'),
  btnClear: $('btn-clear'),
  btnLock: $('btn-lock'),
  lockHint: $('lock-hint'),

  waiting: $('waiting'),
  waitingText: $('waiting-text'),
  btnSkip: $('btn-skip'),

  reveal: $('reveal'),
  revealHeadline: $('reveal-headline'),
  revealDetail: $('reveal-detail'),
  revealNote: $('reveal-note'),
  btnNext: $('btn-next'),

  confetti: $('confetti'),
  endEyebrow: $('end-eyebrow'),
  endAmount: $('end-amount'),
  endText: $('end-text'),
  endSeed: $('end-seed'),
  summary: $('summary'),
  btnAgain: $('btn-again'),
  btnHome: $('btn-home'),
};

let state = null;
let grabSize = 1;
let lanes = [];
let traps = [];
let timerId = null;
let deadline = 0;
let lastWholeSecond = null;
let stopConfetti = null;
let timers = [];
let shownBalance = START_BALANCE;
/** Kroki otwierania zapadni — trzymamy je, żeby dało się je przewinąć. */
let revealSteps = [];

/* ------------------------------------------------------------------ *
 * Pomocnicze
 * ------------------------------------------------------------------ */

function showScreen(name) {
  for (const [key, node] of Object.entries(el.screens)) {
    node.classList.toggle('is-active', key === name);
  }
}

function randomSeed() {
  const alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  return Array.from({ length: 6 }, () => alphabet[Math.floor(Math.random() * alphabet.length)]).join('');
}

function clearTimers() {
  timers.forEach(clearTimeout);
  timers = [];
}

function later(fn, delay) {
  timers.push(setTimeout(fn, delay));
}

/** Płynne przeliczanie kwoty na liczniku. */
function animateMoney(to, duration = 700) {
  const from = shownBalance;
  shownBalance = to;
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.balance.textContent = zl(from + (to - from) * eased);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function setBalance(value) {
  shownBalance = value;
  el.balance.textContent = zl(value);
}

function readBest() {
  const raw = Number(readSetting(BEST_KEY, 0));
  return Number.isFinite(raw) && raw > 0 ? raw : 0;
}

function refreshBestLabel() {
  const best = readBest();
  el.best.textContent = best
    ? `Twój najlepszy wynik: ${zl(best)}`
    : 'Jeszcze nie masz wyniku — czas to zmienić.';
}

/* ------------------------------------------------------------------ *
 * Stosy paczek
 * ------------------------------------------------------------------ */

/**
 * Rysuje stos banknotów o zadanej wartości. Paczki układane są od dołu,
 * z lekkim, ale powtarzalnym rozchwianiem — żeby stos wyglądał jak ułożony
 * ręką, a nie wygenerowany.
 */
function renderStack(node, amount, laneIndex = 0, fallbackHeight = STACK_HEIGHT) {
  const count = Math.round(amount / BUNDLE);
  // wysokość bierzemy z CSS (--stack-max), żeby oba układy — biurkowy
  // i telefonowy — miały jedno źródło prawdy
  const declared = parseFloat(getComputedStyle(node).getPropertyValue('--stack-max'));
  const maxHeight = Number.isFinite(declared) && declared > 0 ? declared : fallbackHeight;

  const key = `${count}:${maxHeight}`;
  if (node.dataset.stack === key) return;
  node.dataset.stack = key;
  node.innerHTML = '';

  const step = count > 0 ? Math.min(7, maxHeight / count) : 0;
  for (let i = 0; i < count; i++) {
    const bundle = document.createElement('i');
    bundle.className = 'bundle';
    // deterministyczne "rozchwianie" — te same paczki zawsze wyglądają tak samo
    const wobble = ((i * 37 + laneIndex * 11) % 7) - 3;
    const tilt = ((i * 53 + laneIndex * 17) % 5) - 2;
    bundle.style.setProperty('--i', String(i));
    bundle.style.setProperty('--x', `${wobble}px`);
    bundle.style.setProperty('--tilt', `${tilt}deg`);
    bundle.style.setProperty('--drift', `${((i * 29 + laneIndex * 7) % 60) - 30}px`);
    bundle.style.bottom = `${i * step}px`;
    node.append(bundle);
  }
}

/* ------------------------------------------------------------------ *
 * Scena
 * ------------------------------------------------------------------ */

/** Buduje puste zapadnie i ekrany — jeszcze zanim padnie pytanie. */
function buildStage(doors) {
  el.stage.style.setProperty('--doors', String(doors));
  el.stage.classList.remove('is-revealed');
  el.stage.classList.add('is-idle');

  el.floor.innerHTML = '';
  traps = Array.from({ length: doors }, () => {
    const trap = document.createElement('div');
    trap.className = 'trap';
    trap.innerHTML = `
      <div class="trap__pit"></div>
      <div class="trap__flap trap__flap--l"></div>
      <div class="trap__flap trap__flap--r"></div>
    `;
    el.floor.append(trap);
    return trap;
  });

  el.lanes.innerHTML = '';
  lanes = Array.from({ length: doors }, (_, index) => {
    const lane = document.createElement('div');
    lane.className = 'lane';
    lane.dataset.index = String(index);
    lane.innerHTML = `
      <button class="lane__hit" type="button" data-act="place" disabled aria-label="Pole ${LETTERS[index]}"></button>
      <div class="lane__panel">
        <span class="lane__letter">${LETTERS[index]}</span>
        <span class="lane__text"></span>
      </div>
      <div class="lane__amount"></div>
      <div class="lane__stack"></div>
      <button class="lane__minus" type="button" data-act="take" title="Zabierz paczkę" aria-label="Zabierz paczkę z pola ${LETTERS[index]}">−</button>
    `;
    el.lanes.append(lane);
    return {
      lane,
      text: lane.querySelector('.lane__text'),
      amount: lane.querySelector('.lane__amount'),
      stack: lane.querySelector('.lane__stack'),
      hit: lane.querySelector('.lane__hit'),
      minus: lane.querySelector('.lane__minus'),
    };
  });
}

/* ------------------------------------------------------------------ *
 * Odsłona 1 — wybór kategorii
 * ------------------------------------------------------------------ */

function renderChoice() {
  clearTimers();
  idleTimer();

  el.roundLabel.textContent = `${state.roundIndex + 1} / ${ROUND_COUNT}`;
  el.question.hidden = true;
  el.choice.hidden = true;
  el.tray.hidden = true;
  el.reveal.hidden = true;
  el.waiting.hidden = true;
  setBalance(state.balance);
  buildStage(doorsInRound(state));

  // plansza rundy — krótka zapowiedź, jak przed przerwą na antenie
  el.bumperRound.textContent = `Runda ${state.roundIndex + 1}`;
  el.bumperStake.textContent = zl(state.balance);
  el.bumper.hidden = false;
  el.bumper.classList.remove('is-running');
  void el.bumper.offsetWidth;
  el.bumper.classList.add('is-running');
  sfx.bumper();

  later(() => {
    el.bumper.hidden = true;
    const offer = offeredCategories(state);
    el.choiceCards.innerHTML = '';
    offer.forEach((category, i) => {
      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'choice__card';
      card.dataset.category = category;
      card.style.setProperty('--i', String(i));
      card.innerHTML = `
        <span class="choice__label">Kategoria ${i + 1}</span>
        <span class="choice__name"></span>
      `;
      card.querySelector('.choice__name').textContent = category;
      el.choiceCards.append(card);
    });
    el.choice.hidden = false;
    el.choiceCards.querySelector('.choice__card')?.focus();
  }, BEAT.bumper);
}

function pickCategory(category) {
  if (!state || state.status !== 'choosing') return;
  clearTimers();
  sfx.pick();
  state = chooseCategory(state, category);
  renderQuestion();
}

/* ------------------------------------------------------------------ *
 * Odsłona 2 — pytanie i rozkładanie paczek
 * ------------------------------------------------------------------ */

function renderQuestion() {
  const question = currentQuestion(state);

  el.choice.hidden = true;
  el.question.hidden = false;
  el.category.textContent = question.category;
  el.questionText.textContent = question.text;
  el.finalHint.hidden = !question.isFinal;

  el.stage.classList.remove('is-idle');
  lanes.forEach((node, index) => {
    node.text.textContent = question.answers[index].text;
  });

  el.tray.hidden = false;
  el.reveal.hidden = true;
  el.waiting.hidden = true;
  renderStakes();
  startTimer();
}

function renderGrabs() {
  const left = unplacedBundles(state);
  el.grabs.innerHTML = '';

  GRAB_SIZES.forEach((size, i) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'grab';
    button.dataset.size = String(size);
    button.disabled = state.status !== 'placing' || left <= 0;
    button.classList.toggle('is-selected', size === grabSize);
    button.setAttribute('aria-pressed', String(size === grabSize));
    button.innerHTML = `
      <span class="grab__stack" style="--n:${Math.min(size, 4)}"></span>
      <span class="grab__label">${zl(size * BUNDLE)}</span>
    `;
    button.title = `Bierz ${paczki(size)} naraz (klawisz ${i + 1})`;
    el.grabs.append(button);
  });
}

function renderStakes() {
  const left = unplaced(state);

  lanes.forEach((node, index) => {
    const value = state.bets[index] ?? 0;
    node.amount.textContent = value > 0 ? zl(value) : '';
    node.lane.classList.toggle('has-money', value > 0);
    node.lane.classList.toggle('is-blocked', !canPlaceOn(state, index));
    node.hit.disabled = !canPlaceOn(state, index);
    node.minus.disabled = value <= 0 || state.status !== 'placing';
    renderStack(node.stack, value, index);
  });

  renderStack(el.handStack, left, 9, HAND_STACK_HEIGHT);
  el.unplaced.textContent = zl(left);
  el.unplacedCount.textContent = left > 0 ? paczki(unplacedBundles(state)) : 'ręce puste';
  el.unplaced.classList.toggle('is-zero', left === 0);

  const ready = canLock(state);
  el.btnLock.disabled = !ready;
  el.btnUndo.disabled = !state.moves.length || state.status !== 'placing';
  el.btnClear.disabled = left === state.balance || state.status !== 'placing';

  if (ready || state.status !== 'placing') el.lockHint.textContent = '';
  else if (left > 0) el.lockHint.textContent = 'Rozłóż wszystkie paczki.';
  else el.lockHint.textContent = 'Jedna zapadnia musi zostać pusta.';

  renderGrabs();
}

/* ------------------------------------------------------------------ *
 * Zegar
 * ------------------------------------------------------------------ */

const RING_LENGTH = 2 * Math.PI * 26;

function startTimer() {
  stopTimer();
  const seconds = roundSeconds(state);
  deadline = performance.now() + seconds * 1000;
  lastWholeSecond = null;
  el.timer.classList.remove('is-urgent', 'is-off');
  el.timerRing.style.strokeDasharray = String(RING_LENGTH);
  tick(seconds);
  timerId = setInterval(() => tick(seconds), 100);
}

function stopTimer() {
  if (timerId) clearInterval(timerId);
  timerId = null;
}

/** Zegar poza rundą: nie odlicza, więc nie udaje, że odlicza. */
function idleTimer() {
  stopTimer();
  el.timer.classList.remove('is-urgent');
  el.timer.classList.add('is-off');
  el.timerValue.textContent = '–';
  el.timerRing.style.strokeDashoffset = String(RING_LENGTH);
}

function tick(seconds) {
  const left = Math.max(0, (deadline - performance.now()) / 1000);
  const whole = Math.ceil(left);

  el.timerValue.textContent = String(whole);
  el.timerRing.style.strokeDashoffset = String(RING_LENGTH * (1 - left / seconds));
  el.timer.classList.toggle('is-urgent', left <= 10);

  if (whole !== lastWholeSecond) {
    if (lastWholeSecond !== null && whole <= 10 && whole > 0) sfx.hurry();
    lastWholeSecond = whole;
  }

  if (left <= 0) {
    stopTimer();
    lockIn(true);
  }
}

/* ------------------------------------------------------------------ *
 * Odsłona 3 — zapadnie otwierają się po kolei
 * ------------------------------------------------------------------ */

function lockIn(byTimeout = false) {
  if (!state || state.status !== 'placing') return;
  if (!byTimeout && !canLock(state)) return;
  idleTimer();
  clearTimers();

  const bets = state.bets.slice();
  state = resolve(state);
  const { correct, forfeited } = state.result;
  const correctSet = new Set(correct);
  const order = revealOrder(state);

  el.tray.hidden = true;
  el.waiting.hidden = false;
  el.waitingText.textContent = byTimeout ? 'Czas minął.' : 'Zapadnie otwierają się…';
  el.stage.classList.add('is-revealed');
  lanes.forEach((node) => {
    node.hit.disabled = true;
    node.minus.disabled = true;
  });

  sfx.lock();

  // Kroki układamy z góry — dzięki temu „Pokaż wynik” po prostu wykonuje
  // wszystkie pozostałe naraz, zamiast osobno odtwarzać całą sekwencję.
  let running = state.result.balanceBefore;
  revealSteps = [];

  if (forfeited > 0) {
    running -= forfeited;
    const to = running;
    revealSteps.push({
      wait: BEAT.tension,
      run: () => {
        sfx.fall();
        animateMoney(to, 600);
        el.waitingText.textContent = `Czas minął — ${zl(forfeited)} zostaje w rękach i przepada.`;
      },
    });
  }

  order.forEach((index, position) => {
    const stake = bets[index];
    const isCorrect = correctSet.has(index);
    const isFirst = position === 0 && forfeited === 0;

    revealSteps.push({
      wait: isFirst ? BEAT.tension : BEAT.spotlight,
      run: () => {
        lanes[index].lane.classList.add('is-next');
        if (!isCorrect) sfx.tension();
      },
    });

    if (isCorrect) {
      revealSteps.push({
        wait: BEAT.spotlight,
        run: () => {
          lanes[index].lane.classList.remove('is-next');
          traps[index].classList.add('is-safe');
          lanes[index].lane.classList.add('is-safe');
          if (stake > 0) sfx.safe();
          else sfx.doors();
          el.waitingText.textContent =
            stake > 0 ? `Poprawna odpowiedź — ${zl(stake)} zostaje.` : 'Poprawna odpowiedź.';
        },
      });
    } else {
      if (stake > 0) running -= stake;
      const to = running;
      revealSteps.push({
        wait: stake > 0 ? BEAT.loadedDoor : BEAT.emptyDoor,
        run: () => {
          lanes[index].lane.classList.remove('is-next');
          traps[index].classList.add('is-open');
          lanes[index].lane.classList.add('is-doomed');
          sfx.doors();
          if (stake > 0) {
            lanes[index].stack.classList.add('is-falling');
            sfx.fall();
            animateMoney(to, 900);
            el.waitingText.textContent = `${LETTERS[index]} — w dół leci ${zl(stake)}.`;
          } else {
            el.waitingText.textContent = `${LETTERS[index]} — puste pole.`;
          }
        },
      });
    }
  });

  revealSteps.push({ wait: BEAT.finish, run: showRoundSummary });
  runNextStep();
}

function runNextStep() {
  const step = revealSteps.shift();
  if (!step) return;
  later(() => {
    step.run();
    runNextStep();
  }, step.wait);
}

/** „Pokaż wynik” — wykonuje wszystko, co zostało, bez czekania. */
function skipReveal() {
  if (!revealSteps.length) return;
  clearTimers();
  const rest = revealSteps;
  revealSteps = [];
  rest.forEach((step) => step.run());
}

function showRoundSummary() {
  const { correct, kept, dropped, forfeited } = state.result;
  const question = currentQuestion(state);

  setBalance(kept);
  el.waiting.hidden = true;

  el.revealHeadline.textContent = kept === 0 ? 'Wszystko przepadło' : `Zostaje ${zl(kept)}`;
  el.revealHeadline.classList.toggle('is-bad', kept === 0);

  const parts = [];
  if (kept === 0) parts.push('Cała kwota poleciała w dół.');
  else if (dropped === 0) parts.push('Ani jedna paczka nie spadła.');
  else parts.push(`W dół poleciało ${zl(dropped)}.`);
  if (forfeited > 0) parts.push(`W tym ${zl(forfeited)}, które zostało w rękach.`);
  el.revealDetail.textContent = parts.join(' ');

  const names = correct.map((i) => `${LETTERS[i]}: ${question.answers[i].text}`).join(' • ');
  el.revealNote.textContent = `Poprawna odpowiedź — ${names}. ${question.note ?? ''}`.trim();

  el.btnNext.textContent = state.outcome === 'continue' ? 'Następna runda' : 'Podsumowanie';
  el.reveal.hidden = false;
  el.btnNext.focus();
}

function goNext() {
  clearTimers();
  revealSteps = [];
  state = advance(state);
  if (state.status === 'over') showEnd();
  else renderChoice();
}

/* ------------------------------------------------------------------ *
 * Ekran końcowy
 * ------------------------------------------------------------------ */

function showEnd() {
  stopTimer();
  const won = state.balance > 0;
  const jackpot = state.balance >= START_BALANCE;

  if (state.balance > readBest()) writeSetting(BEST_KEY, state.balance);
  refreshBestLabel();

  el.endEyebrow.textContent = jackpot
    ? 'MILION!'
    : won
      ? `Koniec gry po ${ROUND_COUNT} pytaniach`
      : `Koniec gry w rundzie ${state.history.length}`;
  el.endAmount.textContent = zl(state.balance);
  el.endAmount.classList.toggle('is-bad', !won);
  el.endText.textContent = jackpot
    ? 'Osiem pytań, ani jedna paczka w dół. Lepiej się nie da.'
    : won
      ? 'To, co zostało na zapadni, jedzie do domu.'
      : 'Wszystkie zapadnie się otworzyły. Następnym razem warto rozłożyć paczki szerzej.';
  el.endSeed.textContent = `Kod tej gry: ${state.seed}`;

  el.summary.innerHTML = '';
  state.history.forEach((row) => {
    const item = document.createElement('li');
    item.className = row.kept > 0 ? 'summary__row' : 'summary__row is-bad';
    item.innerHTML = `
      <span class="summary__round">Runda ${row.round}</span>
      <span class="summary__category"></span>
      <span class="summary__kept">${zl(row.kept)}</span>
      <span class="summary__lost${row.dropped > 0 ? '' : ' is-none'}">${
        row.dropped > 0 ? `−${zl(row.dropped)}` : '—'
      }</span>
    `;
    item.querySelector('.summary__category').textContent = row.category;
    el.summary.append(item);
  });

  showScreen('end');
  if (stopConfetti) stopConfetti();
  if (won) {
    sfx.win();
    stopConfetti = burst(el.confetti, { count: jackpot ? 260 : 150 });
  } else {
    sfx.gameOver();
  }
  el.btnAgain.focus();
}

/* ------------------------------------------------------------------ *
 * Sterowanie
 * ------------------------------------------------------------------ */

function startGame(seed) {
  clearTimers();
  revealSteps = [];
  if (stopConfetti) {
    stopConfetti();
    stopConfetti = null;
  }
  state = createGame({ questions: QUESTIONS, seed: seed || randomSeed() });
  grabSize = 1;
  showScreen('game');
  renderChoice();
}

function putOn(index, all = false) {
  if (!canPlaceOn(state, index)) {
    sfx.blocked();
    return;
  }
  state = all ? placeRest(state, index) : placeBundles(state, index, grabSize);
  sfx.place();
  const node = lanes[index];
  node.lane.classList.remove('is-bumped');
  void node.lane.offsetWidth;
  node.lane.classList.add('is-bumped');
  renderStakes();
}

function onLaneClick(event) {
  const button = event.target.closest('button[data-act]');
  if (!button || state?.status !== 'placing') return;
  const index = Number(button.closest('.lane').dataset.index);

  if (button.dataset.act === 'place') {
    putOn(index, event.shiftKey);
  } else if (button.dataset.act === 'take') {
    if (!state.bets[index]) return;
    state = take(state, index, Math.min(grabSize * BUNDLE, state.bets[index]));
    sfx.remove();
    renderStakes();
  }
}

function onGrabClick(event) {
  const button = event.target.closest('.grab');
  if (!button || button.disabled) return;
  grabSize = Number(button.dataset.size);
  sfx.grab();
  renderGrabs();
}

function onKeyDown(event) {
  if (event.target.matches('input, textarea')) return;
  if (el.rulesDialog.open || !state) return;

  if (state.status === 'choosing') {
    const digit = Number(event.key);
    const cards = [...el.choiceCards.querySelectorAll('.choice__card')];
    if (digit >= 1 && digit <= cards.length) {
      event.preventDefault();
      pickCategory(cards[digit - 1].dataset.category);
    }
    return;
  }

  if (state.status === 'revealed') {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (revealSteps.length) skipReveal();
      else if (!el.reveal.hidden) goNext();
    }
    return;
  }
  if (state.status !== 'placing') return;

  const digit = Number(event.key);
  if (digit >= 1 && digit <= GRAB_SIZES.length) {
    grabSize = GRAB_SIZES[digit - 1];
    sfx.grab();
    renderGrabs();
    return;
  }

  const letter = LETTERS.indexOf(event.key.toUpperCase());
  if (letter >= 0 && letter < state.bets.length) {
    event.preventDefault();
    putOn(letter, event.shiftKey);
    return;
  }

  if (event.key === 'Backspace') {
    event.preventDefault();
    state = undo(state);
    sfx.remove();
    renderStakes();
  } else if (event.key === 'Enter' && canLock(state)) {
    event.preventDefault();
    lockIn();
  }
}

function syncSoundButton() {
  const on = isSoundOn();
  el.btnSound.textContent = on ? '🔊 Dźwięk' : '🔇 Cisza';
  el.btnSound.setAttribute('aria-pressed', String(on));
}

/* ------------------------------------------------------------------ *
 * Podpięcie zdarzeń
 * ------------------------------------------------------------------ */

el.choiceCards.addEventListener('click', (event) => {
  const card = event.target.closest('.choice__card');
  if (card) pickCategory(card.dataset.category);
});
el.lanes.addEventListener('click', onLaneClick);
el.grabs.addEventListener('click', onGrabClick);
el.btnLock.addEventListener('click', () => lockIn(false));
el.btnSkip.addEventListener('click', skipReveal);
el.btnNext.addEventListener('click', goNext);
el.btnUndo.addEventListener('click', () => {
  state = undo(state);
  sfx.remove();
  renderStakes();
});
el.btnClear.addEventListener('click', () => {
  state = clearBets(state);
  sfx.remove();
  renderStakes();
});

el.btnStart.addEventListener('click', () => startGame(el.seedInput.value.trim()));
el.seedInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') startGame(el.seedInput.value.trim());
});
el.btnAgain.addEventListener('click', () => startGame(''));
el.btnHome.addEventListener('click', () => {
  clearTimers();
  revealSteps = [];
  if (stopConfetti) stopConfetti();
  stopTimer();
  refreshBestLabel();
  showScreen('start');
});

el.btnSound.addEventListener('click', () => {
  toggleSound();
  syncSoundButton();
});
el.btnRules.addEventListener('click', () => el.rulesDialog.showModal());
el.btnRulesClose.addEventListener('click', () => el.rulesDialog.close());

document.addEventListener('keydown', onKeyDown);

// Gdy karta zejdzie na drugi plan, zatrzymujemy odliczanie — inaczej gracz
// wracałby do gry z zerowym zegarem.
let hiddenAt = 0;
document.addEventListener('visibilitychange', () => {
  if (!state || state.status !== 'placing') return;
  if (document.hidden) {
    hiddenAt = performance.now();
    stopTimer();
  } else if (hiddenAt) {
    deadline += performance.now() - hiddenAt;
    hiddenAt = 0;
    timerId = setInterval(() => tick(roundSeconds(state)), 100);
  }
});

/* ------------------------------------------------------------------ *
 * Start
 * ------------------------------------------------------------------ */

const seedFromUrl = new URLSearchParams(location.search).get('kod');
if (seedFromUrl) el.seedInput.value = seedFromUrl;
syncSoundButton();
refreshBestLabel();
showScreen('start');

// podgląd stanu w konsoli przeglądarki — przydaje się przy dopisywaniu pytań
window.__pnm = {
  get state() {
    return state;
  },
  skipReveal,
};
