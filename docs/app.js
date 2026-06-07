/* app.js — the browser side: smooth drawing, turning the drawing into a
   28x28 image, and rendering the results. The actual recognition happens in
   Python (main.py). */

const SIZE = 280;                       // on-screen canvas is 280x280
const canvas = document.getElementById("draw");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const seenCanvas = document.getElementById("seen");
const seenCtx = seenCanvas.getContext("2d");

let drawing = false;
let lastX = 0, lastY = 0;
let pyReady = false;
let examples = null;

/* ---------- loading feedback (the one-time Python download can be slow) ---------- */
const statusEl = () => document.getElementById("status");
const slowTimer = setTimeout(() => {
  if (!pyReady) statusEl().innerHTML = "⏳ Downloading Python + NumPy… first load can take 20–40s on a slower connection (it's cached after this).";
}, 12000);
const verySlowTimer = setTimeout(() => {
  if (!pyReady) statusEl().innerHTML = "⏳ Still loading… if it never finishes, an ad-blocker or restrictive network may be blocking the CDN — try another browser or network.";
}, 40000);
// main.py calls this if Python itself fails to start, so errors are visible instead of a frozen screen.
window.onPyError = function (msg) {
  pyReady = true;
  clearTimeout(slowTimer); clearTimeout(verySlowTimer);
  const s = statusEl();
  s.innerHTML = "⚠️ Python failed to start: " + msg;
  s.style.color = "#ff6b6b";
};

/* ---------- canvas setup ---------- */
function resetCanvas() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#fff";
}
resetCanvas();

function clearAll() {
  resetCanvas();
  document.getElementById("pred").textContent = "–";
  document.getElementById("sure").textContent = pyReady ? "Draw a digit!" : "Booting Python…";
  buildBars();             // reset bars to empty
  seenCtx.clearRect(0, 0, 28, 28);
}

/* ---------- drawing with mouse + touch (Pointer Events) ---------- */
function pos(e) {
  const r = canvas.getBoundingClientRect();
  return [
    (e.clientX - r.left) * (SIZE / r.width),
    (e.clientY - r.top) * (SIZE / r.height),
  ];
}
canvas.addEventListener("pointerdown", (e) => {
  drawing = true;
  [lastX, lastY] = pos(e);
  ctx.beginPath();
  ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const [x, y] = pos(e);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  [lastX, lastY] = [x, y];
});
function endStroke() {
  if (!drawing) return;
  drawing = false;
  predict();
}
canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointerleave", endStroke);

/* ---------- turn the drawing into a 28x28 image (MNIST-style) ---------- */
function getDrawing() {
  const src = ctx.getImageData(0, 0, SIZE, SIZE).data;
  // Find the bounding box of the ink (white pixels on black).
  let minX = SIZE, minY = SIZE, maxX = 0, maxY = 0, found = false;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      if (src[(y * SIZE + x) * 4] > 30) {       // R channel
        found = true;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (!found) return null;

  const bw = maxX - minX + 1, bh = maxY - minY + 1;
  const longSide = Math.max(bw, bh);
  // MNIST fits the digit into a 20px box centered in a 28x28 frame.
  const scale = 20 / longSide;
  const dw = bw * scale, dh = bh * scale;

  const tmp = document.createElement("canvas");
  tmp.width = 28; tmp.height = 28;
  const tctx = tmp.getContext("2d", { willReadFrequently: true });
  tctx.fillStyle = "#000";
  tctx.fillRect(0, 0, 28, 28);
  tctx.imageSmoothingEnabled = true;
  tctx.drawImage(canvas, minX, minY, bw, bh, (28 - dw) / 2, (28 - dh) / 2, dw, dh);

  const d = tctx.getImageData(0, 0, 28, 28).data;
  const out = new Array(784);
  for (let i = 0; i < 784; i++) out[i] = d[i * 4];
  return out;
}

/* ---------- ask Python, then render ---------- */
function predict() {
  if (!pyReady || !window.pyPredict) return;
  const arr = getDrawing();
  if (!arr) return;
  const res = JSON.parse(window.pyPredict(arr));
  render(res);
}

function render(res) {
  document.getElementById("pred").textContent = res.pred;
  const pct = Math.round(res.probs[res.pred] * 100);
  document.getElementById("sure").textContent = `I'm ${pct}% sure`;
  buildBars(res.probs, res.pred);
  drawSeen(res.seen);
}

/* ---------- confidence bars ---------- */
function buildBars(probs, pred) {
  const wrap = document.getElementById("bars");
  wrap.innerHTML = "";
  for (let d = 0; d < 10; d++) {
    const p = probs ? probs[d] : 0;
    const row = document.createElement("div");
    row.className = "bar-row" + (pred === d ? " win" : "");
    row.innerHTML =
      `<span class="bar-label">${d}</span>` +
      `<span class="bar-track"><span class="bar-fill" style="width:${(p * 100).toFixed(1)}%"></span></span>` +
      `<span class="bar-pct">${Math.round(p * 100)}</span>`;
    wrap.appendChild(row);
  }
}

/* ---------- "what the network sees" preview ---------- */
function drawSeen(seen) {
  const id = seenCtx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = seen[i];
    id.data[i * 4] = v; id.data[i * 4 + 1] = v; id.data[i * 4 + 2] = v; id.data[i * 4 + 3] = 255;
  }
  seenCtx.putImageData(id, 0, 0);
}

/* ---------- "Try an example" (real held-out MNIST test digits) ---------- */
async function tryExample() {
  if (!examples) {
    const r = await fetch("./examples.json");
    examples = (await r.json()).digits;
  }
  const pick = examples[Math.floor(Math.random() * examples.length)];
  const tmp = document.createElement("canvas");
  tmp.width = 28; tmp.height = 28;
  const tctx = tmp.getContext("2d");
  const id = tctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = pick.pixels[i];
    id.data[i * 4] = v; id.data[i * 4 + 1] = v; id.data[i * 4 + 2] = v; id.data[i * 4 + 3] = 255;
  }
  tctx.putImageData(id, 0, 0);
  resetCanvas();
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, 0, 0, 28, 28, 0, 0, SIZE, SIZE);
  predict();
}

/* ---------- buttons ---------- */
document.getElementById("clear").addEventListener("click", clearAll);
document.getElementById("example").addEventListener("click", tryExample);
buildBars();                            // show empty bars at startup

/* ---------- Python readiness handshake (called from main.py) ---------- */
window.onPyReady = function (acc) {
  pyReady = true;
  clearTimeout(slowTimer); clearTimeout(verySlowTimer);
  const status = document.getElementById("status");
  status.innerHTML = `🐍 Python + NumPy ready — model is ${(acc * 100).toFixed(1)}% accurate`;
  status.classList.add("ready");
  document.getElementById("sure").textContent = "Draw a digit!";
  document.getElementById("example").removeAttribute("disabled");
  document.getElementById("clear").removeAttribute("disabled");
  // Show it working immediately with a real test digit.
  tryExample();
};
