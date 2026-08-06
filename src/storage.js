/**
 * Bezpieczny dostęp do localStorage.
 *
 * W trybie prywatnym, w osadzonej ramce albo przy zablokowanych ciasteczkach
 * samo sięgnięcie po localStorage rzuca wyjątkiem. Bez tej osłony gra nie
 * wystartowałaby w ogóle — a przecież chodzi tylko o zapamiętanie wyniku
 * i ustawienia dźwięku.
 */

export function readSetting(key, fallback = null) {
  try {
    const value = localStorage.getItem(key);
    return value === null ? fallback : value;
  } catch {
    return fallback;
  }
}

export function writeSetting(key, value) {
  try {
    localStorage.setItem(key, String(value));
    return true;
  } catch {
    return false;
  }
}
