/**
 * Silnik gry "Postaw na milion".
 *
 * Zasady wzorowane na teleturnieju: pieniądze leżą w paczkach na zapadniach.
 * Po zatwierdzeniu zapadnie pod błędnymi odpowiedziami otwierają się i paczki
 * lecą w dół. Zostaje tylko to, co leżało na poprawnym polu.
 *
 * Moduł jest czysty (bez DOM, bez efektów ubocznych) i deterministyczny przy
 * podanym ziarnie losowości — dzięki temu da się go testować w Node.
 * Każda funkcja zwraca NOWY stan, nigdy nie modyfikuje przekazanego.
 */

/** Wartość jednej paczki banknotów. Wszystkie kwoty są jej wielokrotnością. */
export const BUNDLE = 25_000;

export const START_BALANCE = 1_000_000; // 40 paczek
export const ROUND_COUNT = 8;

/** Ile paczek naraz można chwycić (przyciski w tacy). */
export const GRAB_SIZES = [1, 2, 4, 10];

/** Czas na rozłożenie paczek w kolejnych rundach (w sekundach). */
export const ROUND_SECONDS = [90, 85, 80, 75, 70, 65, 60, 50];

/** Poziom trudności pytania w kolejnych rundach. */
export const DIFFICULTY_PLAN = [1, 2, 2, 3, 3, 4, 4, 5];

/** W ostatniej rundzie zostają tylko dwie zapadnie — wszystko na jedną. */
export const FINAL_DOORS = 2;

/* ------------------------------------------------------------------ *
 * Losowość
 * ------------------------------------------------------------------ */

/** Generator mulberry32 — szybki, deterministyczny, wystarczający do gry. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Zamienia dowolny tekst na 32-bitowe ziarno (FNV-1a). */
export function hashSeed(text) {
  let h = 2166136261;
  const s = String(text);
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function makeRng(seed) {
  return mulberry32(hashSeed(seed));
}

/** Tasowanie Fishera-Yatesa na kopii tablicy. */
export function shuffle(list, rng) {
  const out = list.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * Dobór pytań
 * ------------------------------------------------------------------ */

/**
 * Wybiera po jednym pytaniu na rundę zgodnie z DIFFICULTY_PLAN.
 * Gdy w danym poziomie zabraknie pytań, sięga po najbliższy dostępny poziom.
 * Kolejność odpowiedzi w każdym pytaniu jest losowana.
 */
export function selectQuestions(questions, rng, plan = DIFFICULTY_PLAN) {
  const pools = new Map();
  for (const q of questions) {
    if (!pools.has(q.difficulty)) pools.set(q.difficulty, []);
    pools.get(q.difficulty).push(q);
  }
  for (const [level, pool] of pools) pools.set(level, shuffle(pool, rng));

  const levels = [...pools.keys()].sort((a, b) => a - b);
  const picked = [];

  for (const wanted of plan) {
    const order = levels
      .slice()
      .sort((a, b) => Math.abs(a - wanted) - Math.abs(b - wanted) || a - b);
    let question = null;
    for (const level of order) {
      const pool = pools.get(level);
      if (pool && pool.length) {
        question = pool.pop();
        break;
      }
    }
    if (!question) throw new Error('Za mało pytań w bazie, aby rozpocząć grę.');
    picked.push(prepareQuestion(question, rng));
  }
  return picked;
}

/** Kopiuje pytanie z potasowanymi odpowiedziami. */
export function prepareQuestion(question, rng) {
  return {
    ...question,
    answers: shuffle(question.answers, rng).map((a) => ({ ...a })),
  };
}

/**
 * Finał: zostaje poprawna odpowiedź i jeden losowy dystraktor.
 * Kolejność znowu losowa, żeby poprawna nie lądowała zawsze po tej samej stronie.
 */
export function trimToFinal(question, rng, doors = FINAL_DOORS) {
  const correct = question.answers.filter((a) => a.correct);
  const wrong = shuffle(
    question.answers.filter((a) => !a.correct),
    rng,
  );
  const kept = [...correct, ...wrong].slice(0, Math.max(doors, correct.length + 1));
  return { ...question, answers: shuffle(kept, rng), isFinal: true };
}

/* ------------------------------------------------------------------ *
 * Stan gry
 * ------------------------------------------------------------------ */

export function createGame({
  questions,
  seed = 'domyslne',
  startBalance = START_BALANCE,
  plan = DIFFICULTY_PLAN,
} = {}) {
  const rng = makeRng(seed);
  const selected = selectQuestions(questions, rng, plan);
  selected[selected.length - 1] = trimToFinal(selected[selected.length - 1], rng);

  return {
    seed: String(seed),
    startBalance,
    balance: startBalance,
    roundIndex: 0,
    questions: selected,
    bets: new Array(selected[0].answers.length).fill(0),
    moves: [],
    status: 'placing', // 'placing' | 'revealed' | 'over'
    outcome: null, // 'continue' | 'lost' | 'won'
    result: null,
    history: [],
  };
}

export function currentQuestion(state) {
  return state.questions[state.roundIndex];
}

export function roundSeconds(state) {
  return ROUND_SECONDS[state.roundIndex] ?? ROUND_SECONDS[ROUND_SECONDS.length - 1];
}

/** Kwota, która nie została jeszcze położona na żadnej zapadni. */
export function unplaced(state) {
  return state.balance - state.bets.reduce((sum, v) => sum + v, 0);
}

/** Liczba paczek pozostałych w ręce. */
export function unplacedBundles(state) {
  return unplaced(state) / BUNDLE;
}

/** Ile zapadni jest jeszcze pustych. */
export function emptyDoors(state) {
  return state.bets.filter((v) => v === 0).length;
}

/**
 * Zasada teleturnieju: co najmniej jedna zapadnia musi zostać pusta.
 * Pilnujemy jej już przy kładzeniu paczek, więc stan nigdy nie jest niezgodny —
 * także wtedy, gdy rundę rozliczy upływ czasu.
 */
export function canPlaceOn(state, index) {
  if (state.status !== 'placing') return false;
  if (unplaced(state) <= 0) return false;
  if (state.bets[index] > 0) return true;
  return emptyDoors(state) > 1;
}

/** Czy rundę można zatwierdzić. */
export function canLock(state) {
  return state.status === 'placing' && unplaced(state) === 0 && emptyDoors(state) >= 1;
}

function assertPlacing(state) {
  if (state.status !== 'placing') {
    throw new Error('Paczki można przekładać tylko w trakcie rundy.');
  }
}

export function place(state, index, amount) {
  assertPlacing(state);
  if (!Number.isInteger(index) || index < 0 || index >= state.bets.length) {
    throw new Error('Nie ma takiej zapadni.');
  }
  if (!(amount > 0)) throw new Error('Kwota musi być dodatnia.');
  if (amount % BUNDLE !== 0) throw new Error('Kłaść można tylko całe paczki.');
  if (amount > unplaced(state)) throw new Error('Nie masz tylu pieniędzy.');
  if (state.bets[index] === 0 && emptyDoors(state) <= 1) {
    throw new Error('Jedna zapadnia musi zostać pusta.');
  }

  const bets = state.bets.slice();
  bets[index] += amount;
  return { ...state, bets, moves: [...state.moves, { index, amount }] };
}

/** Kładzie n paczek; jeśli tyle nie zostało, kładzie tyle, ile jest. */
export function placeBundles(state, index, count) {
  const available = Math.min(count, unplacedBundles(state));
  return available > 0 ? place(state, index, available * BUNDLE) : state;
}

export function take(state, index, amount) {
  assertPlacing(state);
  const current = state.bets[index] ?? 0;
  const value = Math.min(amount, current);
  if (value <= 0) return state;
  const bets = state.bets.slice();
  bets[index] -= value;
  return { ...state, bets, moves: [...state.moves, { index, amount: -value }] };
}

/** Dokłada całą resztę na wskazaną zapadnię. */
export function placeRest(state, index) {
  const left = unplaced(state);
  return left > 0 ? place(state, index, left) : state;
}

export function undo(state) {
  assertPlacing(state);
  if (!state.moves.length) return state;
  const moves = state.moves.slice();
  const last = moves.pop();
  const bets = state.bets.slice();
  bets[last.index] -= last.amount;
  return { ...state, bets, moves };
}

export function clearBets(state) {
  assertPlacing(state);
  return { ...state, bets: state.bets.map(() => 0), moves: [] };
}

export function correctIndexes(question) {
  return question.answers
    .map((a, i) => (a.correct ? i : -1))
    .filter((i) => i >= 0);
}

/**
 * Otwiera zapadnie: zostaje tylko to, co leży na poprawnej odpowiedzi.
 * Pieniądze wciąż trzymane w ręce (np. po upływie czasu) też lecą w dół.
 */
export function resolve(state) {
  assertPlacing(state);
  const question = currentQuestion(state);
  const correct = correctIndexes(question);
  const correctSet = new Set(correct);

  let kept = 0;
  let dropped = 0;
  state.bets.forEach((value, i) => {
    if (correctSet.has(i)) kept += value;
    else dropped += value;
  });
  const forfeited = unplaced(state);

  const isLast = state.roundIndex >= state.questions.length - 1;
  const outcome = kept === 0 ? 'lost' : isLast ? 'won' : 'continue';

  const result = {
    correct,
    kept,
    dropped: dropped + forfeited,
    forfeited,
    balanceBefore: state.balance,
  };

  return {
    ...state,
    balance: kept,
    status: 'revealed',
    outcome,
    result,
    history: [
      ...state.history,
      {
        round: state.roundIndex + 1,
        questionId: question.id,
        bets: state.bets.slice(),
        kept,
        dropped: result.dropped,
      },
    ],
  };
}

/** Przechodzi do kolejnej rundy albo kończy grę. */
export function advance(state) {
  if (state.status !== 'revealed') return state;
  if (state.outcome !== 'continue') {
    return { ...state, status: 'over' };
  }
  const roundIndex = state.roundIndex + 1;
  return {
    ...state,
    roundIndex,
    bets: new Array(state.questions[roundIndex].answers.length).fill(0),
    moves: [],
    status: 'placing',
    outcome: null,
    result: null,
  };
}

export function isOver(state) {
  return state.status === 'over';
}
