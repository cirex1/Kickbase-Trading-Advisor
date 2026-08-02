/**
 * Minimalne konfetti na <canvas>. Bez bibliotek, bez zależności.
 */

const COLORS = ['#ffd76a', '#ffb02e', '#fff3c4', '#7cf7d2', '#8ab6ff', '#ff7ba9'];

export function burst(canvas, { count = 180, duration = 4200 } = {}) {
  if (!canvas) return () => {};
  const ctx = canvas.getContext('2d');
  if (!ctx) return () => {};

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const resize = () => {
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
  };
  resize();
  window.addEventListener('resize', resize);

  const w = () => canvas.width;
  const h = () => canvas.height;

  const pieces = Array.from({ length: count }, () => ({
    x: Math.random() * w(),
    y: -Math.random() * h() * 0.5,
    vx: (Math.random() - 0.5) * 2.2 * dpr,
    vy: (1.4 + Math.random() * 2.6) * dpr,
    size: (4 + Math.random() * 7) * dpr,
    spin: (Math.random() - 0.5) * 0.28,
    angle: Math.random() * Math.PI * 2,
    color: COLORS[Math.floor(Math.random() * COLORS.length)],
  }));

  let raf = 0;
  const started = performance.now();

  const frame = (now) => {
    const life = now - started;
    ctx.clearRect(0, 0, w(), h());
    const fade = Math.max(0, 1 - Math.max(0, life - duration * 0.6) / (duration * 0.4));

    for (const p of pieces) {
      p.x += p.vx;
      p.y += p.vy;
      p.vy += 0.02 * dpr;
      p.angle += p.spin;
      if (p.y > h() + 20) {
        p.y = -20;
        p.x = Math.random() * w();
      }
      ctx.save();
      ctx.globalAlpha = fade;
      ctx.translate(p.x, p.y);
      ctx.rotate(p.angle);
      ctx.fillStyle = p.color;
      ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
      ctx.restore();
    }

    if (life < duration) {
      raf = requestAnimationFrame(frame);
    } else {
      ctx.clearRect(0, 0, w(), h());
    }
  };
  raf = requestAnimationFrame(frame);

  return function stop() {
    cancelAnimationFrame(raf);
    window.removeEventListener('resize', resize);
    ctx.clearRect(0, 0, w(), h());
  };
}
