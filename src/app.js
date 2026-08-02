/**
 * Warstwa prezentacji: łączy silnik gry (src/engine.js) z DOM-em.
 * Stan gry trzymamy w jednej zmiennej `state` i po każdej zmianie
 * odświeżamy odpowiedni fragment interfejsu.
 */

import {
  DENOMINATIONS,
  ROUND_COUNT,
  START_BALANCE,
  advance,
  clearBets,
  createGame,
  currentQuestion,
  place,
  placeRest,
  resolve,
  roundSeconds,
  take,
  undo,
  unplaced,
} from './engine.js';
import { QUESTIONS } from './questions.js';
import { isSoundOn, sfx, toggleSound } from './audio.js';
import { burst } from './confetti.js';

const LETTERS = ['A', 'B', 'C', 'D', 'E', 'F'];
const BEST_KEY = 'pnm.best';

const money = new Intl.NumberFormat('pl-PL', { maximumFractionDigits: 0 });
const zl = (value) => `${money.format(Math.round(value))} zł`;

const $ = (id) => document.getElementById(id);

const el = {
  screens: {
    start: $('screen-start'),
    game: $('screen-game'),
    end: $('screen-end'),
  },
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

  category: $('q-category'),
  questionText: $('q-text'),
  multiHint: $('q-multi'),
  answers: $('answers'),

  tray: $('tray'),
  unplaced: $('unplaced'),
  chips: $('chips'),
  btnUndo: $('btn-undo'),
  btnClear: $('btn-clear'),
  btnLock: $('btn-lock'),

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
let selectedChip = DENOMINATIONS[DENOMINATIONS.length - 1];
let answerNodes = [];
let timerId = null;
let deadline = 0;
let lastWholeSecond = null;
let stopConfetti = null;

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

/** Płynne przeliczanie kwoty na liczniku. */
function animateMoney(node, from, to, duration = 700) {
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    node.textContent = zl(from + (to - from) * eased);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function readBest() {
  const raw = Number(localStorage.getItem(BEST_KEY));
  return Number.isFinite(raw) && raw > 0 ? raw : 0;
}

function writeBest(value) {
  if (value > readBest()) localStorage.setItem(BEST_KEY, String(value));
}

function refreshBestLabel() {
  const best = readBest();
  el.best.textContent = best ? `Twój najlepszy wynik: ${zl(best)}` : 'Jeszcze nie masz wyniku — czas to zmienić.';
}

/* ------------------------------------------------------------------ *
 * Rysowanie rundy
 * ------------------------------------------------------------------ */

function renderRound() {
  const question = currentQuestion(state);
  const correctCount = question.answers.filter((a) => a.correct).length;

  el.roundLabel.textContent = `${state.roundIndex + 1} / ${ROUND_COUNT}`;
  el.category.textContent = question.category;
  el.questionText.textContent = question.text;
  el.multiHint.hidden = correctCount < 2;
  el.multiHint.textContent = `Uwaga: poprawne odpowiedzi: ${correctCount}`;

  el.reveal.hidden = true;
  el.tray.hidden = false;
  el.answers.classList.remove('is-revealed');
  el.answers.innerHTML = '';
  answerNodes = [];

  question.answers.forEach((answer, index) => {
    const card = document.createElement('div');
    card.className = 'answer';
    card.dataset.index = String(index);
    card.innerHTML = `
      <button class="answer__main" type="button" data-act="place">
        <span class="answer__letter">${LETTERS[index]}</span>
        <span class="answer__text"></span>
      </button>
      <div class="answer__footer">
        <span class="answer__amount">0 zł</span>
        <span class="answer__tools">
          <button class="mini" type="button" data-act="take" title="Zdejmij żeton">−</button>
          <button class="mini" type="button" data-act="rest" title="Dołóż całą resztę">Reszta</button>
        </span>
      </div>
      <div class="answer__bar"><i></i></div>
    `;
    card.querySelector('.answer__text').textContent = answer.text;
    el.answers.append(card);
    answerNodes.push({
      card,
      amount: card.querySelector('.answer__amount'),
      bar: card.querySelector('.answer__bar i'),
      main: card.querySelector('.answer__main'),
      buttons: [...card.querySelectorAll('button')],
    });
  });

  el.balance.textContent = zl(state.balance);
  renderStakes();
  startTimer();
}

function renderChips() {
  const left = unplaced(state);
  const affordable = DENOMINATIONS.filter((d) => d <= left);
  if (affordable.length && !affordable.includes(selectedChip)) {
    selectedChip = affordable[affordable.length - 1];
  }

  el.chips.innerHTML = '';
  DENOMINATIONS.forEach((value, i) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = `chip chip--${i}`;
    chip.dataset.value = String(value);
    chip.disabled = value > left || state.status !== 'placing';
    chip.setAttribute('aria-pressed', String(value === selectedChip));
    chip.classList.toggle('is-selected', value === selectedChip);
    chip.innerHTML = `<span class="chip__value">${value >= 1000 ? `${value / 1000}k` : value}</span>`;
    chip.title = `Żeton ${zl(value)} (klawisz ${i + 1})`;
    el.chips.append(chip);
  });
}

function renderStakes() {
  const left = unplaced(state);
  const total = Math.max(state.balance, 1);

  answerNodes.forEach((node, index) => {
    const value = state.bets[index] ?? 0;
    node.amount.textContent = zl(value);
    node.bar.style.width = `${(value / total) * 100}%`;
    node.card.classList.toggle('has-stake', value > 0);
  });

  el.unplaced.textContent = zl(left);
  el.unplaced.classList.toggle('is-zero', left === 0);
  el.btnLock.disabled = left !== 0 || state.status !== 'placing';
  el.btnUndo.disabled = !state.moves.length || state.status !== 'placing';
  el.btnClear.disabled = left === state.balance || state.status !== 'placing';
  renderChips();
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
  el.timer.classList.remove('is-urgent');
  el.timerRing.style.strokeDasharray = String(RING_LENGTH);
  tick(seconds);
  timerId = setInterval(() => tick(seconds), 100);
}

function stopTimer() {
  if (timerId) clearInterval(timerId);
  timerId = null;
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
 * Rozstrzygnięcie rundy
 * ------------------------------------------------------------------ */

function lockIn(byTimeout = false) {
  if (!state || state.status !== 'placing') return;
  stopTimer();

  const before = state.balance;
  state = resolve(state);
  const { correct, kept, lost, forfeited } = state.result;

  sfx.lock();
  el.tray.hidden = true;
  el.answers.classList.add('is-revealed');

  const correctSet = new Set(correct);
  answerNodes.forEach((node, index) => {
    node.buttons.forEach((b) => (b.disabled = true));
    node.card.classList.add(correctSet.has(index) ? 'is-correct' : 'is-wrong');
    if (!correctSet.has(index) && state.bets[index] > 0) {
      node.card.classList.add('is-swept');
    }
  });

  setTimeout(() => {
    if (kept > 0) sfx.correct();
    else sfx.wrong();
    if (lost > 0) sfx.sweep();
    animateMoney(el.balance, before, kept);
  }, 450);

  el.revealHeadline.textContent = kept === 0 ? 'Koniec drogi' : `Zostaje ${zl(kept)}`;
  el.revealHeadline.classList.toggle('is-bad', kept === 0);

  const parts = [];
  if (kept === 0) parts.push('Wszystkie żetony leżały na złych polach.');
  else if (lost === 0) parts.push('Cała kwota jedzie dalej — ani złotówki straty.');
  else parts.push(`Ze stołu znika ${zl(lost)}.`);
  if (forfeited > 0) {
    parts.push(
      byTimeout
        ? `Czas minął — ${zl(forfeited)} nie trafiło na żadne pole.`
        : `${zl(forfeited)} nie zostało rozłożone i przepada.`,
    );
  }
  el.revealDetail.textContent = parts.join(' ');

  const question = currentQuestion(state);
  const names = correct.map((i) => `${LETTERS[i]}: ${question.answers[i].text}`).join(' • ');
  el.revealNote.textContent = `${correct.length > 1 ? 'Poprawne' : 'Poprawna'} ${
    correct.length > 1 ? 'odpowiedzi' : 'odpowiedź'
  } — ${names}. ${question.note ?? ''}`.trim();

  el.btnNext.textContent = state.outcome === 'continue' ? 'Następne pytanie' : 'Podsumowanie';
  el.reveal.hidden = false;
  el.btnNext.focus();
}

function goNext() {
  state = advance(state);
  if (state.status === 'over') {
    showEnd();
  } else {
    renderRound();
  }
}

/* ------------------------------------------------------------------ *
 * Ekran końcowy
 * ------------------------------------------------------------------ */

function showEnd() {
  stopTimer();
  const won = state.balance > 0;
  const jackpot = state.balance >= START_BALANCE;

  writeBest(state.balance);
  refreshBestLabel();

  el.endEyebrow.textContent = jackpot
    ? 'MILION!'
    : won
      ? `Koniec gry po ${ROUND_COUNT} pytaniach`
      : `Koniec gry w rundzie ${state.history.length}`;
  el.endAmount.textContent = zl(state.balance);
  el.endAmount.classList.toggle('is-bad', !won);
  el.endText.textContent = jackpot
    ? 'Komplet dziesięciu pytań i pełna pula. Lepiej się nie da.'
    : won
      ? 'Kwota, która została na stole, jedzie do domu.'
      : 'Na stole nie został ani grosz. Następnym razem warto rozłożyć żetony szerzej.';
  el.endSeed.textContent = `Kod tej gry: ${state.seed}`;

  el.summary.innerHTML = '';
  state.history.forEach((row) => {
    const item = document.createElement('li');
    item.className = row.kept > 0 ? 'summary__row' : 'summary__row is-bad';
    item.innerHTML = `
      <span class="summary__round">Runda ${row.round}</span>
      <span class="summary__kept">${zl(row.kept)}</span>
      <span class="summary__lost">${row.lost > 0 ? `−${zl(row.lost)}` : '—'}</span>
    `;
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
  if (stopConfetti) {
    stopConfetti();
    stopConfetti = null;
  }
  state = createGame({ questions: QUESTIONS, seed: seed || randomSeed() });
  selectedChip = DENOMINATIONS[DENOMINATIONS.length - 1];
  showScreen('game');
  renderRound();
}

function onAnswerClick(event) {
  const button = event.target.closest('button[data-act]');
  if (!button || state?.status !== 'placing') return;
  const card = button.closest('.answer');
  const index = Number(card.dataset.index);

  if (button.dataset.act === 'place') {
    const left = unplaced(state);
    if (left <= 0) return;
    const amount = Math.min(selectedChip, left);
    state = place(state, index, amount);
    sfx.place();
    pulse(card);
  } else if (button.dataset.act === 'take') {
    if (!state.bets[index]) return;
    state = take(state, index, Math.min(selectedChip, state.bets[index]));
    sfx.remove();
  } else if (button.dataset.act === 'rest') {
    if (unplaced(state) <= 0) return;
    state = placeRest(state, index);
    sfx.place();
    pulse(card);
  }
  renderStakes();
}

function pulse(card) {
  card.classList.remove('is-pulsing');
  void card.offsetWidth;
  card.classList.add('is-pulsing');
}

function onChipClick(event) {
  const chip = event.target.closest('.chip');
  if (!chip || chip.disabled) return;
  selectedChip = Number(chip.dataset.value);
  sfx.chip();
  renderChips();
}

function onKeyDown(event) {
  if (event.target.matches('input, textarea')) return;
  if (el.rulesDialog.open) return;
  if (!state) return;

  if (state.status === 'revealed' && (event.key === 'Enter' || event.key === ' ')) {
    event.preventDefault();
    goNext();
    return;
  }
  if (state.status !== 'placing') return;

  const digit = Number(event.key);
  if (digit >= 1 && digit <= DENOMINATIONS.length) {
    const value = DENOMINATIONS[digit - 1];
    if (value <= unplaced(state)) {
      selectedChip = value;
      sfx.chip();
      renderChips();
    }
    return;
  }

  const letter = LETTERS.indexOf(event.key.toUpperCase());
  if (letter >= 0 && letter < state.bets.length) {
    event.preventDefault();
    const left = unplaced(state);
    if (left <= 0) return;
    state = place(state, letter, Math.min(selectedChip, left));
    sfx.place();
    pulse(answerNodes[letter].card);
    renderStakes();
    return;
  }

  if (event.key === 'Backspace') {
    event.preventDefault();
    state = undo(state);
    sfx.remove();
    renderStakes();
  } else if (event.key === 'Enter' && unplaced(state) === 0) {
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

el.answers.addEventListener('click', onAnswerClick);
el.chips.addEventListener('click', onChipClick);
el.btnLock.addEventListener('click', () => lockIn(false));
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
