const SVGNS = "http://www.w3.org/2000/svg";
const tip = document.getElementById("tip");
function el(tag, attrs, parent) { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function fmt(v, unit) { return v == null ? "—" : (Math.round(v * 10) / 10) + unit; }

function showTip(evt, title, rows) {
  tip.textContent = "";
  const t = document.createElement("div"); t.className = "tt"; t.textContent = title; tip.appendChild(t);
  rows.forEach(r => {
    const row = document.createElement("div"); row.className = "row";
    const i = document.createElement("i"); i.style.background = "var(" + r.c + ")";
    const b = document.createElement("b"); b.textContent = r.value;
    const s = document.createElement("span"); s.textContent = r.name;
    row.append(i, b, s); tip.appendChild(row);
  });
  tip.hidden = false;
  let x, y;
  if (evt && evt.clientX != null && evt.type.startsWith("pointer")) { x = evt.clientX + 14; y = evt.clientY + 14; }
  else { const r = evt.target.getBoundingClientRect(); x = r.left + r.width / 2; y = r.top - 10; }
  const w = tip.offsetWidth, h = tip.offsetHeight;
  if (x + w > window.innerWidth - 8) x = Math.max(8, window.innerWidth - w - 8);
  if (y + h > window.innerHeight - 8) y = Math.max(8, y - h - 28);
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.hidden = true; }

function legend(id, series, kind) {
  const box = document.getElementById(id); box.textContent = "";
  series.forEach(s => {
    const sp = document.createElement("span"); const i = document.createElement("i");
    i.className = kind === "line" ? "ln" : "sw"; i.style.background = "var(" + s.c + ")";
    sp.append(i, document.createTextNode(s.name)); box.appendChild(sp);
  });
}

function table(id, head, rows, textCols) {
  const nText = textCols || 1;
  const box = document.getElementById(id); box.textContent = "";
  const t = document.createElement("table"); const th = document.createElement("thead"); const tr = document.createElement("tr");
  head.forEach((h, i) => { const c = document.createElement("th"); c.textContent = h; if (i >= nText) c.className = "n"; tr.appendChild(c); });
  th.appendChild(tr); t.appendChild(th);
  const tb = document.createElement("tbody");
  rows.forEach(r => { const row = document.createElement("tr"); r.forEach((v, i) => { const c = document.createElement("td"); c.textContent = (typeof v === "number") ? String(v) : v; if (i >= nText) c.className = "n"; row.appendChild(c); }); tb.appendChild(row); });
  t.appendChild(tb); box.appendChild(t);
}

function yAxis(svg, m, iw, ih, yMax, unit) {
  [0, 25, 50, 75, 100].filter(t => t <= yMax).forEach(t => {
    const y = m.t + ih - t / yMax * ih;
    el("line", { x1: m.l, x2: m.l + iw, y1: y, y2: y, style: "stroke: var(" + (t === 0 ? "--axis" : "--rule") + "); stroke-width: 1" }, svg);
    const tx = el("text", { x: m.l - 8, y: y + 4, "text-anchor": "end", style: "fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums" }, svg);
    tx.textContent = t + (unit.trim() === "%" ? "%" : "");
  });
}

function lineChart(id, spec, height) {
  const box = document.getElementById(id); const H = height || 260; box.style.height = H + "px";
  let idx = null;
  function draw() {
    const W = box.clientWidth; if (!W) return; box.textContent = "";
    const m = { t: 12, r: 18, b: 40, l: 44 }; const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const xs = spec.x, x0 = xs[0], x1 = xs[xs.length - 1];
    const X = v => m.l + (v - x0) / (x1 - x0) * iw, Y = v => m.t + ih - v / spec.yMax * ih;
    const svg = el("svg", { width: W, height: H, role: "img", tabindex: 0, "aria-label": T.lineAria }, box);
    yAxis(svg, m, iw, ih, spec.yMax, spec.unit);
    xs.forEach(v => { const t = el("text", { x: X(v), y: m.t + ih + 18, "text-anchor": "middle", style: "fill: var(--muted); font-size: 11px" }, svg); t.textContent = v; });
    const xl = el("text", { x: m.l + iw / 2, y: H - 4, "text-anchor": "middle", style: "fill: var(--ink-2); font-size: 11.5px" }, svg); xl.textContent = spec.xLabel;
    spec.series.forEach(s => {
      let d = ""; s.v.forEach((v, i) => { if (v == null) return; d += (d ? "L" : "M") + X(xs[i]) + "," + Y(v); });
      el("path", { d, style: "fill: none; stroke: var(" + s.c + "); stroke-width: " + (s.thin ? 1.5 : 2) + "; stroke-linejoin: round; stroke-linecap: round" }, svg);
      if (!s.thin) s.v.forEach((v, i) => { if (v == null) return; el("circle", { cx: X(xs[i]), cy: Y(v), r: 3.5, style: "fill: var(" + s.c + "); stroke: var(--surface); stroke-width: 2" }, svg); });
    });
    const cross = el("line", { y1: m.t, y2: m.t + ih, style: "stroke: var(--axis); stroke-width: 1; visibility: hidden" }, svg);
    const hit = el("rect", { x: m.l - 10, y: m.t, width: iw + 20, height: ih, style: "fill: transparent" }, svg);
    function at(i, evt) {
      idx = i; const x = X(xs[i]); cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.style.visibility = "visible";
      showTip(evt, spec.xLabel + " = " + xs[i], spec.series.map(s => ({ c: s.c, name: s.name, value: fmt(s.v[i], spec.unit) })));
    }
    hit.addEventListener("pointermove", e => { const r = svg.getBoundingClientRect(); const px = e.clientX - r.left; let best = 0; xs.forEach((v, i) => { if (Math.abs(X(v) - px) < Math.abs(X(xs[best]) - px)) best = i; }); at(best, e); });
    hit.addEventListener("pointerleave", () => { cross.style.visibility = "hidden"; hideTip(); });
    svg.addEventListener("focus", e => at(idx == null ? 0 : idx, e));
    svg.addEventListener("blur", () => { cross.style.visibility = "hidden"; hideTip(); });
    svg.addEventListener("keydown", e => { if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); const cur = idx == null ? 0 : idx; const n = Math.min(xs.length - 1, Math.max(0, cur + (e.key === "ArrowRight" ? 1 : -1))); at(n, { type: "key", target: svg }); } });
  }
  new ResizeObserver(draw).observe(box); draw();
}

function barChart(id, spec, height) {
  const box = document.getElementById(id); const H = height || 260; box.style.height = H + "px";
  function draw() {
    const W = box.clientWidth; if (!W) return; box.textContent = "";
    const m = { t: 18, r: 8, b: 44, l: 44 }; const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const cats = spec.cats, series = spec.series;
    const nc = cats.length, ns = series.length; const slot = iw / nc;
    const bw = Math.max(6, Math.min(24, (slot * 0.72 - (ns - 1) * 2) / ns)); const gw = ns * bw + (ns - 1) * 2;
    const Y = v => m.t + ih - v / spec.yMax * ih;
    const svg = el("svg", { width: W, height: H, role: "img", "aria-label": T.barAria }, box);
    yAxis(svg, m, iw, ih, spec.yMax, spec.unit);
    cats.forEach((c, ci) => {
      const cx = m.l + slot * ci + slot / 2;
      c.split(" · ").forEach((p, pi) => { const t = el("text", { x: cx, y: m.t + ih + 16 + pi * 14, "text-anchor": "middle", style: "fill: var(--ink-2); font-size: 11px" }, svg); t.textContent = p; });
      series.forEach((s, si) => {
        const v = s.v[ci]; const x = cx - gw / 2 + si * (bw + 2); const y = Y(v), h = Math.max(0, m.t + ih - y); const r = Math.min(4, h);
        const d = "M" + x + "," + (m.t + ih) + "V" + (y + r) + "Q" + x + "," + y + " " + (x + r) + "," + y + "H" + (x + bw - r) + "Q" + (x + bw) + "," + y + " " + (x + bw) + "," + (y + r) + "V" + (m.t + ih) + "Z";
        const bar = el("path", { d, style: "fill: var(" + s.c + ")" }, svg);
        if (bw >= 14) { const lab = el("text", { x: x + bw / 2, y: y - 5, "text-anchor": "middle", style: "fill: var(--ink-2); font-size: 10.5px; font-variant-numeric: tabular-nums" }, svg); lab.textContent = (Math.round(v * 10) / 10); }
        const hit = el("rect", { x: x - 1, y: m.t, width: bw + 2, height: ih, tabindex: 0, style: "fill: transparent", "aria-label": c + T.sep + s.name + ": " + fmt(v, spec.unit) }, svg);
        const on = e => { bar.style.opacity = "0.82"; showTip(e, c, [{ c: s.c, name: s.name, value: fmt(v, spec.unit) }]); };
        const off = () => { bar.style.opacity = "1"; hideTip(); };
        hit.addEventListener("pointermove", on); hit.addEventListener("pointerleave", off); hit.addEventListener("focus", on); hit.addEventListener("blur", off);
      });
    });
  }
  new ResizeObserver(draw).observe(box); draw();
  return { redraw: draw };
}

function hbarChart(id, rows, colors, maxV, minV) {
  const box = document.getElementById(id); const rowH = 22; const H = rows.length * rowH + 34; box.style.height = H + "px";
  function draw() {
    const W = box.clientWidth; if (!W) return; box.textContent = "";
    const lab = Math.min(170, W * 0.42); const m = { t: 4, r: 40, b: 26, l: lab + 8 }; const iw = W - m.l - m.r;
    const X = v => m.l + (v - minV) / (maxV - minV) * iw;
    const svg = el("svg", { width: W, height: H, role: "img", "aria-label": T.hbarAria }, box);
    (minV < 0 ? [minV, 0, 25, 50, 75, 100] : [0, 25, 50, 75, 100]).filter(t => t >= minV && t <= maxV).forEach(t => {
      el("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, style: "stroke: var(" + (t === 0 ? "--axis" : "--rule") + "); stroke-width: 1" }, svg);
      const tx = el("text", { x: X(t), y: H - 8, "text-anchor": "middle", style: "fill: var(--muted); font-size: 11px" }, svg); tx.textContent = t + "%";
    });
    rows.forEach((r, i) => {
      const y = m.t + i * rowH + 4; const bh = 14;
      const t = el("text", { x: m.l - 8, y: y + 11, "text-anchor": "end", style: "fill: var(--ink-2); font-size: 11px" }, svg); t.textContent = r.name;
      const x0 = X(Math.min(0, r.v)), x1 = X(Math.max(0, r.v));
      el("rect", { x: x0, y, width: Math.max(1, x1 - x0), height: bh, rx: 3, style: "fill: var(" + colors[r.label] + ")" }, svg);
      const vt = el("text", { x: X(r.v) + (r.v >= 0 ? 5 : -5), y: y + 11, "text-anchor": r.v >= 0 ? "start" : "end", style: "fill: var(--ink-2); font-size: 10.5px; font-variant-numeric: tabular-nums" }, svg);
      vt.textContent = (Math.round(r.v * 10) / 10);
      const hit = el("rect", { x: 0, y: y - 3, width: W, height: rowH, tabindex: 0, style: "fill: transparent", "aria-label": r.name + T.sep + T.failRate + " " + r.v + "%" }, svg);
      const on = e => showTip(e, r.name, [{ c: colors[r.label], name: T.failRate, value: (Math.round(r.v * 10) / 10) + "%" }, { c: "--neutral-mark", name: T.fullAcc, value: r.full + "%" }, { c: "--neutral-mark", name: T.nIncl, value: String(r.n) }]);
      hit.addEventListener("pointermove", on); hit.addEventListener("pointerleave", hideTip); hit.addEventListener("focus", on); hit.addEventListener("blur", hideTip);
    });
  }
  new ResizeObserver(draw).observe(box); draw();
}

// figure 1
legend("lg-f1", DATA.f1.series, "line"); lineChart("f1", DATA.f1);
table("tb-f1", [T.f1Head, ...DATA.f1.x.map(k => "k=" + k)], DATA.f1.series.map(s => [s.name, ...s.v.map(v => v == null ? "—" : v)])
  .concat([[T.pickLast, "—", 72.0, 93.5, 97.0, 98.5], [T.ordered, 100, 100, 100, 99.5, 100]]));
// figure 2
legend("lg-f2", DATA.f2.series, "bar"); barChart("f2", DATA.f2);
table("tb-f2", [T.f2Head[0], ...DATA.f2.series.map(s => s.name)], DATA.f2.cats.map((c, i) => [c, ...DATA.f2.series.map(s => s.v[i])]));
table("tb-f2b", T.f2bHead, DATA.f2b);
// figure 3 with k switch
let k3 = 8;
legend("lg-f3", DATA.f3.byK[8], "bar");
const f3 = barChart("f3", { cats: DATA.f3.cats, yMax: 100, unit: "%", get series() { return DATA.f3.byK[k3]; } });
document.querySelectorAll("#k-seg button").forEach(b => b.addEventListener("click", () => {
  k3 = Number(b.dataset.k); document.querySelectorAll("#k-seg button").forEach(x => x.setAttribute("aria-pressed", String(x === b))); f3.redraw();
}));
table("tb-f3", T.f3Head, DATA.f3.table, 2);
// figure 4
legend("lg-f4", DATA.f4.series, "bar"); barChart("f4", DATA.f4);
table("tb-f4", T.f4Head, DATA.f4full, 2);
// figure 5
legend("lg-f5", DATA.f5.series, "line"); lineChart("f5", DATA.f5, 280);
table("tb-f5", [T.f5Head, ...DATA.f5.x.map(x => x + "%")], DATA.f5.series.map(s => [s.name, ...s.v]).concat(DATA.f5extra));
// figure 6
legend("lg-f6a", DATA.f6a.series, "line"); lineChart("f6a", DATA.f6a, 240);
legend("lg-f6b", DATA.f6b.series, "line"); lineChart("f6b", DATA.f6b, 240);
table("tb-f6", T.f6Head, [[T.f6Rows[0], ...DATA.f6a.series[0].v], [T.f6Rows[1], ...DATA.f6a.series[1].v],
  [T.f6Rows[2], ...DATA.f6b.series[0].v], [T.f6Rows[3], ...DATA.f6b.series[1].v]]);
// figure 8
legend("lg-f8", DATA.f8.series, "bar"); barChart("f8", DATA.f8, 240);
table("tb-f8", T.f8Head, DATA.f8full, 2);
// figure 7 (BBH-27)
if (DATA.f7) {
  document.getElementById("f7-fig").hidden = false;
  const colors = { state: "--s1", partial: "--s3", none: "--neutral-mark" };
  legend("lg-f7", [{ name: T.f7Legend.state, c: "--s1" }, { name: T.f7Legend.partial, c: "--s3" }, { name: T.f7Legend.none, c: "--neutral-mark" }], "bar");
  const rows = DATA.f7.rows.slice().sort((a, b) => b.v - a.v);
  hbarChart("f7", rows, colors, DATA.f7.max, DATA.f7.min);
  table("tb-f7", T.f7Head, rows.map(r => [r.name, T.f7Legend[r.label], r.full, r.shuf, r.v, r.n]), 2);
}
// experiment index
const ix = document.getElementById("exp-index");
DATA.index.forEach(r => { const tr = document.createElement("tr"); r.forEach((v, i) => { const td = document.createElement("td"); td.textContent = v; if (i === 0) td.className = "mono"; tr.appendChild(td); }); ix.appendChild(tr); });
window.addEventListener("scroll", hideTip, { passive: true });
