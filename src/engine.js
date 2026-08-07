/**
 * Silnik gry "Postaw na milion".
 *
 * Przebieg rundy:
 *   'choosing' → gracz dostaje dwa hasła do wyboru; co się za nimi kryje,
 *                 okazuje się dopiero po wskazaniu jednego z nich
 *   'placing'  → pada pytanie, gracz rozkłada paczki na zapadniach
 *   'revealed' → zapadnie otwierają się po kolei, pieniądze lecą w dół
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

/**
 * Czas na rozłożenie paczek (w sekundach). W teleturnieju jest to około
 * minuty na pytanie — tyle zostawiamy i my, skracając tylko finał.
 */
export const ROUND_SECONDS = [60, 60, 60, 60, 60, 60, 60, 45];

/**
 * Pytania nie są ustawione w drabinkę trudności — w teleturnieju napięcie
 * bierze się z malejącej liczby zapadni i rosnącej stawki, a nie z coraz
 * trudniejszych pytań. Wyjątkiem jest pierwsza runda: to rozgrzewka, więc
 * losujemy do niej wyłącznie pytania oznaczone jako łatwe.
 *
 * `difficulty: 1` znaczy zatem „nadaje się na rozgrzewkę”, a nie „pierwszy
 * szczebel z pięciu”. Reszta puli jest jednym workiem.
 */
export const WARMUP_DIFFICULTY = 1;

/**
 * Najniższa trudność, po jaką sięga losowanie w kolejnych rundach.
 *
 * To nie jest drabinka: pula pozostaje jednym workiem, a `difficulty` nadal nie
 * ustawia pytań w kolejności. Zmienia się tylko to, z jakiego wycinka worka
 * wolno losować — bo gracz, który zna odpowiedź, kładzie wszystko na jedną
 * zapadnię i nie traci ani grosza. Przy czterech polach trudność pytania jest
 * jedynym źródłem ryzyka, jakie w tej grze istnieje.
 *
 * Gdyby próg okazał się nie do przejścia, wystarczy tu wpisać same dwójki —
 * dobór pytań wróci do jednego wielkiego worka.
 */
export const DIFFICULTY_FLOOR = [1, 2, 2, 3, 3, 4, 4, 5];

/** Ile haseł dostaje gracz do wyboru przed pytaniem. */
export const CATEGORY_CHOICES = 2;

/**
 * Ile zapadni stoi w kolejnych rundach.
 *
 * Tak jest w teleturnieju: cztery pola do czwartego pytania, potem trzy,
 * a w finale dwa. Pytanie robi się mechanicznie łatwiejsze dokładnie wtedy,
 * gdy ryzyko rośnie — bo przy dwóch polach reguła pustej zapadni oznacza,
 * że cała pula musi wylądować na jednej odpowiedzi.
 */
export const DOORS_PLAN = [4, 4, 4, 4, 3, 3, 3, 2];

/** Ile zapadni zostaje w finale. */
export const FINAL_DOORS = DOORS_PLAN[DOORS_PLAN.length - 1];

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

/**
 * Losowość wywodzona z ziarna i etapu gry, zamiast jednego generatora
 * przesuwanego w czasie. Dzięki temu stan pozostaje zwykłym obiektem —
 * da się go porównać, zapisać i odtworzyć.
 */
function stageRng(state, stage) {
  return makeRng(`${state.seed}#${state.roundIndex}#${stage}`);
}

/* ------------------------------------------------------------------ *
 * Dobór pytań
 * ------------------------------------------------------------------ */

/** Czy w tej rundzie gramy pytaniem rozgrzewkowym. */
export function isWarmupRound(state) {
  return state.roundIndex === 0;
}

/** Ile najmniej trudne pytanie może mieć w tej rundzie. */
export function difficultyFloor(state) {
  const plan = state.difficultyFloor ?? DIFFICULTY_FLOOR;
  return plan[Math.min(state.roundIndex, plan.length - 1)] ?? 2;
}

/**
 * Czy ten wycinek puli wystarczy, żeby zbudować ofertę.
 *
 * Same dwa pytania to za mało — hasła mają pochodzić z różnych dziedzin,
 * inaczej wybór sprowadza się do dwóch pytań o to samo.
 */
function enoughToOffer(list) {
  return list.length >= CATEGORY_CHOICES && new Set(list.map((q) => q.category)).size >= CATEGORY_CHOICES;
}

/**
 * Pytania dostępne w danej rundzie: jeszcze niezadane, a poza tym mieszczące się
 * w progu trudności. Próg obniżamy stopniowo, dopóki nie zostanie z czego wybrać —
 * brak pytania musiałby przerwać grę, a to gorsze niż zadanie łatwiejszego.
 *
 * Pierwsza runda rządzi się swoim prawem: to rozgrzewka, więc losujemy do niej
 * wyłącznie pytania oznaczone jako łatwe.
 */
export function availableQuestions(state) {
  const unused = state.pool.filter((q) => !state.usedIds.includes(q.id));
  const warmup = (q) => q.difficulty === WARMUP_DIFFICULTY;

  if (isWarmupRound(state)) {
    const easy = unused.filter(warmup);
    return enoughToOffer(easy) ? easy : unused;
  }

  const rest = unused.filter((q) => !warmup(q));
  for (let floor = difficultyFloor(state); floor > WARMUP_DIFFICULTY + 1; floor--) {
    const hard = rest.filter((q) => q.difficulty >= floor);
    if (enoughToOffer(hard)) return hard;
  }
  return enoughToOffer(rest) ? rest : unused;
}

/**
 * Dwa hasła do wyboru.
 *
 * W teleturnieju kategoria nie zdradza tematu — nosi przewrotną nazwę, której
 * sens rozumie się dopiero po pytaniu. Dlatego wybór dotyczy konkretnych pytań,
 * a nie dziedzin: gracz widzi wyłącznie `label`, a `category` (prawdziwy temat)
 * wychodzi na jaw dopiero przy rozwiązaniu.
 *
 * Staramy się dobrać hasła z różnych dziedzin, żeby wybór nie sprowadzał się
 * do dwóch pytań o to samo.
 */
export function offeredQuestions(state) {
  const pool = shuffle(availableQuestions(state), stageRng(state, 'oferta'));
  const picked = [];
  for (const question of pool) {
    if (picked.length >= CATEGORY_CHOICES) break;
    if (picked.some((q) => q.category === question.category)) continue;
    picked.push(question);
  }
  // gdy w puli zostały już tylko pytania z jednej dziedziny, bierzemy je mimo to
  for (const question of pool) {
    if (picked.length >= CATEGORY_CHOICES) break;
    if (!picked.includes(question)) picked.push(question);
  }
  // `category` jako zapasowe hasło — pytanie bez wymyślonej nazwy nadal działa
  return picked.map((q) => ({ id: q.id, label: q.label ?? q.category }));
}

/**
 * Kopiuje pytanie z potasowanymi odpowiedziami.
 *
 * Każda odpowiedź zapamiętuje przy okazji swoje miejsce w bazie (`at`).
 * Po tasowaniu i zwężaniu nie da się go już odtworzyć, a jest potrzebne:
 * nagrania lektora są podpisane numerem z bazy, nie numerem zapadni.
 */
export function prepareQuestion(question, rng) {
  const numbered = question.answers.map((a, at) => ({ ...a, at }));
  return { ...question, answers: shuffle(numbered, rng) };
}

/**
 * Ogranicza pytanie do liczby zapadni w danej rundzie.
 *
 * Zostaje poprawna odpowiedź i — koniecznie — ta oznaczona jako `rival`, czyli
 * najpoważniejszy kontrkandydat. Gdyby wypadała losowo, pytanie robiłoby się
 * banalne dokładnie wtedy, gdy na placu boju zostałby oczywisty odsiew.
 * Resztę miejsc, jeśli jakieś zostały, dobieramy losowo.
 */
export function trimToDoors(question, rng, doors) {
  if (doors >= question.answers.length) return question;
  const correct = question.answers.filter((a) => a.correct);
  const wrong = question.answers.filter((a) => !a.correct);
  const rivals = wrong.filter((a) => a.rival);
  const rest = shuffle(
    wrong.filter((a) => !a.rival),
    rng,
  );
  const kept = [...correct, ...rivals, ...rest].slice(0, Math.max(doors, correct.length + 1));
  return { ...question, answers: shuffle(kept, rng) };
}

/* ------------------------------------------------------------------ *
 * Stan gry
 * ------------------------------------------------------------------ */

export function createGame({
  questions,
  seed = 'domyslne',
  startBalance = START_BALANCE,
  doorsPlan = DOORS_PLAN,
  difficultyFloor: floorPlan = DIFFICULTY_FLOOR,
} = {}) {
  return {
    seed: String(seed),
    doorsPlan,
    difficultyFloor: floorPlan,
    pool: questions,
    startBalance,
    balance: startBalance,
    roundIndex: 0,
    usedIds: [],
    question: null,
    bets: [],
    moves: [],
    status: 'choosing', // 'choosing' | 'placing' | 'revealed' | 'over'
    outcome: null, // 'continue' | 'lost' | 'won'
    result: null,
    history: [],
  };
}

/** Ile zapadni stoi w tej rundzie — jeszcze zanim padnie pytanie. */
export function doorsInRound(state) {
  const plan = state.doorsPlan ?? DOORS_PLAN;
  return plan[Math.min(state.roundIndex, plan.length - 1)];
}

export function isFinalRound(state) {
  const plan = state.doorsPlan ?? DOORS_PLAN;
  return state.roundIndex >= plan.length - 1;
}

/**
 * Gracz wskazuje hasło — dopiero teraz odsłania się pytanie, które się za nim kryło.
 */
export function chooseOffer(state, id) {
  if (state.status !== 'choosing') {
    throw new Error('Hasło wybiera się przed pytaniem.');
  }
  if (!offeredQuestions(state).some((o) => o.id === id)) {
    throw new Error(`Hasło ${id} nie jest w tej rundzie do wyboru.`);
  }
  const picked = state.pool.find((q) => q.id === id);

  const rng = stageRng(state, `pytanie:${id}`);
  const doors = doorsInRound(state);
  const question = {
    ...trimToDoors(prepareQuestion(picked, rng), rng, doors),
    isFinal: isFinalRound(state),
  };

  return {
    ...state,
    question,
    usedIds: [...state.usedIds, picked.id],
    bets: new Array(question.answers.length).fill(0),
    moves: [],
    status: 'placing',
  };
}

export function currentQuestion(state) {
  return state.question;
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
 * Kolejność otwierania zapadni.
 *
 * Prowadzący nigdy nie otwiera wszystkich naraz. Najpierw idą pola puste —
 * nic się nie dzieje, a napięcie rośnie. Potem te z pieniędzmi, od najmniejszej
 * stawki. Poprawna zapadnia zostaje na koniec, więc im więcej gracz na niej
 * postawił, tym dłużej czeka na wiadomość, że pieniądze zostały.
 */
export function revealOrder(state) {
  const correct = new Set(correctIndexes(state.question));
  const wrong = state.bets
    .map((stake, index) => ({ stake, index }))
    .filter(({ index }) => !correct.has(index))
    .sort((a, b) => a.stake - b.stake || a.index - b.index)
    .map(({ index }) => index);
  return [...wrong, ...correct];
}

/**
 * Zamyka rundę: na koncie zostaje tylko to, co leży na poprawnej zapadni.
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

  const outcome = kept === 0 ? 'lost' : isFinalRound(state) ? 'won' : 'continue';

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
        category: question.category,
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
  return {
    ...state,
    roundIndex: state.roundIndex + 1,
    question: null,
    bets: [],
    moves: [],
    status: 'choosing',
    outcome: null,
    result: null,
  };
}

export function isOver(state) {
  return state.status === 'over';
}
