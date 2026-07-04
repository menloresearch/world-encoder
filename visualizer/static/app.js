"use strict";

const $ = (sel) => document.querySelector(sel);
const SERIES_VARS = ["--series-1", "--series-2", "--series-3", "--series-4"];
const seriesColor = (i) => `var(${SERIES_VARS[i % SERIES_VARS.length]})`;
const SVG_NS = "http://www.w3.org/2000/svg";

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function svgEl(tag, attrs = {}) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") e.style.cssText = v;
    else e.setAttribute(k, v);
  }
  return e;
}

/* ================= tabs ================= */

const tabs = { data: $("#tab-data"), metrics: $("#tab-metrics") };
const views = { data: $("#view-data"), metrics: $("#view-metrics") };
let metricsLoaded = false;

function showTab(name) {
  for (const k of Object.keys(tabs)) {
    tabs[k].classList.toggle("active", k === name);
    tabs[k].setAttribute("aria-selected", String(k === name));
    views[k].hidden = k !== name;
  }
  if (name === "metrics" && !metricsLoaded) {
    metricsLoaded = true;
    loadMetrics();
  }
}
tabs.data.addEventListener("click", () => showTab("data"));
tabs.metrics.addEventListener("click", () => showTab("metrics"));

/* ================= data browser ================= */

const state = {
  cfg: null,
  scenes: [],
  scene: null,
  cams: [],
  cam: null,
  stream: "color",
  timestamps: [],
  idx: 0,
  playing: false,
  playTimer: null,
};

function frameURL(t) {
  const { cfg, scene, cam, stream } = state;
  return `/frames/${cfg}/${scene}/${cam}/${stream}/${t}.jpg`;
}

async function init() {
  const summary = await getJSON("/api/summary");
  $("#root-info").textContent = summary.frames_root;
  const pills = $("#cfg-pills");
  pills.textContent = "";
  if (!summary.cfgs.length) {
    pills.append(el("span", "muted", "No cfg dirs found under the frames root."));
    return;
  }
  for (const cfg of summary.cfgs) {
    const b = el("button", "pill");
    b.append(el("span", "", cfg.name), document.createTextNode(" "),
             el("span", "n", `(${cfg.scenes})`));
    b.addEventListener("click", () => selectCfg(cfg.name, b));
    pills.append(b);
  }
  // default to the first cfg that has scenes
  const first = summary.cfgs.find((c) => c.scenes > 0) || summary.cfgs[0];
  const idx = summary.cfgs.indexOf(first);
  selectCfg(first.name, pills.children[idx]);
}

async function selectCfg(name, pill) {
  for (const p of $("#cfg-pills").children) p.classList.toggle("active", p === pill);
  state.cfg = name;
  const res = await getJSON(`/api/scenes?cfg=${encodeURIComponent(name)}`);
  state.scenes = res.scenes;
  renderSceneList();
}

// "task_0001_user_0016_scene_0001_cfg_0003" -> "task 0001 · user 0016 · scene 0001"
function prettyScene(name) {
  const m = name.match(/^task_(\w+?)_user_(\w+?)_scene_(\w+?)_cfg_\w+$/);
  return m ? `task ${m[1]} · user ${m[2]} · scene ${m[3]}` : name;
}

function renderSceneList() {
  const q = $("#scene-search").value.trim().toLowerCase();
  const list = $("#scene-list");
  list.textContent = "";
  const matches = state.scenes.filter((s) => s.toLowerCase().includes(q));
  $("#scene-count").textContent =
    `${matches.length} of ${state.scenes.length} scenes`;
  const frag = document.createDocumentFragment();
  for (const s of matches) {
    const b = el("button", "scene-item", prettyScene(s));
    b.title = s;
    if (s === state.scene) b.classList.add("active");
    b.addEventListener("click", () => selectScene(s));
    frag.append(b);
  }
  list.append(frag);
}
$("#scene-search").addEventListener("input", renderSceneList);

async function selectScene(name) {
  stopPlayback();
  state.scene = name;
  renderSceneList();
  const res = await getJSON(
    `/api/scene?cfg=${encodeURIComponent(state.cfg)}&name=${encodeURIComponent(name)}`);
  state.cams = res.cams;
  $("#player-empty").hidden = true;
  $("#player").hidden = false;
  $("#player-title").textContent = `${prettyScene(name)} — ${state.cfg}`;
  const pills = $("#cam-pills");
  pills.textContent = "";
  res.cams.forEach((cam, i) => {
    const streams = Object.keys(cam.streams);
    const stream = streams.includes("color") ? "color" : streams[0];
    const b = el("button", "pill");
    b.append(el("span", "", cam.name.replace(/^cam_/, "cam ")),
             document.createTextNode(" "),
             el("span", "n", `(${cam.streams[stream] ?? 0})`));
    b.addEventListener("click", () => selectCam(cam.name, stream, b));
    pills.append(b);
    if (i === 0) selectCam(cam.name, stream, b);
  });
}

async function selectCam(cam, stream, pill) {
  stopPlayback();
  for (const p of $("#cam-pills").children) p.classList.toggle("active", p === pill);
  state.cam = cam;
  state.stream = stream;
  const res = await getJSON(
    `/api/frames?cfg=${encodeURIComponent(state.cfg)}&scene=${encodeURIComponent(state.scene)}` +
    `&cam=${encodeURIComponent(cam)}&stream=${encodeURIComponent(stream)}`);
  state.timestamps = res.timestamps;
  state.idx = 0;
  const scrub = $("#scrub");
  scrub.max = Math.max(0, state.timestamps.length - 1);
  scrub.value = 0;
  showFrame(0);
}

function showFrame(i) {
  const ts = state.timestamps;
  if (!ts.length) return;
  state.idx = Math.max(0, Math.min(i, ts.length - 1));
  $("#frame-img").src = frameURL(ts[state.idx]);
  $("#scrub").value = state.idx;
  const t0 = ts[0];
  const rel = ((ts[state.idx] - t0) / 1000).toFixed(2);
  $("#frame-label").textContent =
    `frame ${state.idx + 1}/${ts.length} · t=${rel}s`;
  // warm the cache a few frames ahead so playback doesn't stutter
  for (let k = 1; k <= 4 && state.idx + k < ts.length; k++) {
    new Image().src = frameURL(ts[state.idx + k]);
  }
}

function stopPlayback() {
  state.playing = false;
  clearTimeout(state.playTimer);
  $("#btn-play").textContent = "▶";
}

function stepPlayback() {
  if (!state.playing) return;
  const ts = state.timestamps;
  if (state.idx >= ts.length - 1) { stopPlayback(); return; }
  showFrame(state.idx + 1);
  const speed = parseFloat($("#speed").value);
  const next = state.idx + 1 < ts.length ? ts[state.idx + 1] - ts[state.idx] : 100;
  const dt = Math.max(15, Math.min(1000, next)) / speed;
  state.playTimer = setTimeout(stepPlayback, dt);
}

$("#btn-play").addEventListener("click", () => {
  if (state.playing) { stopPlayback(); return; }
  if (!state.timestamps.length) return;
  if (state.idx >= state.timestamps.length - 1) state.idx = 0;
  state.playing = true;
  $("#btn-play").textContent = "❚❚";
  stepPlayback();
});
$("#scrub").addEventListener("input", (e) => {
  stopPlayback();
  showFrame(parseInt(e.target.value, 10));
});
document.addEventListener("keydown", (e) => {
  if (views.data.hidden || e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === " ") { e.preventDefault(); $("#btn-play").click(); }
  else if (e.key === "ArrowRight") { stopPlayback(); showFrame(state.idx + 1); }
  else if (e.key === "ArrowLeft") { stopPlayback(); showFrame(state.idx - 1); }
});

/* ================= metrics ================= */

async function loadMetrics() {
  const body = $("#metrics-body");
  body.textContent = "";
  let res;
  try {
    res = await getJSON("/api/metrics");
  } catch (err) {
    body.append(el("p", "muted", `Failed to load metrics: ${err.message}`));
    return;
  }
  if (!res.files.length) {
    const empty = el("div", "empty-state");
    empty.append(
      el("p", "", "No metrics yet."),
      el("p", "muted",
         `When a training/eval run finishes, drop a JSON file into ${res.dir} ` +
         `(schema in visualizer/README.md) and reload this page.`));
    body.append(empty);
    return;
  }
  for (const f of res.files) body.append(renderRunCard(f));
}

function renderRunCard(f) {
  const card = el("section", "run-card");
  if (f.error) {
    card.classList.add("error-card");
    card.append(el("h2", "", f.file),
                el("p", "run-note", `Could not parse: ${f.error}`));
    return card;
  }
  const d = f.data || {};
  card.append(el("h2", "", d.run || f.file));
  if (d.note) card.append(el("p", "run-note", d.note));
  card.append(el("p", "run-file",
    `${f.file} · updated ${new Date(f.mtime * 1000).toLocaleString()}`));

  if (d.scalars && Object.keys(d.scalars).length) {
    const row = el("div", "kpi-row");
    for (const [label, value] of Object.entries(d.scalars)) {
      const tile = el("div", "stat-tile");
      tile.append(el("div", "label", label), el("div", "value", fmtNum(value)));
      row.append(tile);
    }
    card.append(row);
  }
  for (const spec of d.charts || []) card.append(lineChart(spec));
  for (const spec of d.bars || []) card.append(barChart(spec));
  for (const spec of d.tables || []) card.append(tableBlock(spec));
  return card;
}

function fmtNum(v) {
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v) && Math.abs(v) < 100000) return v.toLocaleString();
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return v.toPrecision(3);
}

// clean tick values covering [min, max]
function niceTicks(min, max, n = 5) {
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const lo = Math.floor(min / step) * step;
  const ticks = [];
  for (let v = lo; v <= max + step * 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

const tooltip = $("#tooltip");
function showTooltip(x, y, build) {
  tooltip.textContent = "";
  build(tooltip);
  tooltip.hidden = false;
  const pad = 12;
  const r = tooltip.getBoundingClientRect();
  let left = x + pad, top = y + pad;
  if (left + r.width > window.innerWidth - 8) left = x - r.width - pad;
  if (top + r.height > window.innerHeight - 8) top = y - r.height - pad;
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}
function hideTooltip() { tooltip.hidden = true; }

function chartBlock(title) {
  const block = el("div", "chart-block");
  if (title) block.append(el("h3", "", title));
  return block;
}

// collapsible table view — the no-hover / relief path for every chart
function dataTable(columns, rows) {
  const det = el("details", "data-table");
  det.append(el("summary", "", "Data table"));
  const table = el("table", "metrics-table");
  const trh = el("tr");
  for (const c of columns) trh.append(el("th", "", String(c)));
  table.append(trh);
  for (const row of rows) {
    const tr = el("tr");
    for (const v of row) tr.append(el("td", "", typeof v === "number" ? fmtNum(v) : String(v)));
    table.append(tr);
  }
  det.append(table);
  return det;
}

/* spec: {title, x_label?, y_label?, x: [..], series: {name: [..], ...}} */
function lineChart(spec) {
  const block = chartBlock(spec.title);
  const names = Object.keys(spec.series || {});
  const x = spec.x || [];
  if (!names.length || !x.length) {
    block.append(el("p", "muted", "chart has no data"));
    return block;
  }

  if (names.length >= 2) {
    const legend = el("div", "legend");
    names.forEach((name, i) => {
      const key = el("span", "key");
      const sw = el("span", "swatch-line");
      sw.style.borderTopColor = seriesColor(i);
      key.append(sw, el("span", "", name));
      legend.append(key);
    });
    block.append(legend);
  }

  const W = 720, H = 280, m = { l: 52, r: 20, t: 12, b: 34 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const allY = names.flatMap((n) => spec.series[n]).filter((v) => v != null);
  const yTicks = niceTicks(Math.min(...allY), Math.max(...allY));
  const y0 = yTicks[0], y1 = yTicks[yTicks.length - 1];
  const x0 = Math.min(...x), x1 = Math.max(...x);
  const px = (v) => m.l + (x1 === x0 ? iw / 2 : ((v - x0) / (x1 - x0)) * iw);
  const py = (v) => m.t + ih - ((v - y0) / (y1 - y0)) * ih;

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });

  for (const t of yTicks) {
    svg.append(svgEl("line", {
      x1: m.l, x2: W - m.r, y1: py(t), y2: py(t),
      style: "stroke: var(--grid); stroke-width: 1",
    }));
    const lbl = svgEl("text", {
      x: m.l - 8, y: py(t) + 4, "text-anchor": "end",
      style: "fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums",
    });
    lbl.textContent = fmtNum(t);
    svg.append(lbl);
  }
  svg.append(svgEl("line", {
    x1: m.l, x2: W - m.r, y1: py(y0), y2: py(y0),
    style: "stroke: var(--baseline); stroke-width: 1",
  }));
  for (const t of niceTicks(x0, x1, 6)) {
    if (t < x0 || t > x1) continue;
    const lbl = svgEl("text", {
      x: px(t), y: H - m.b + 18, "text-anchor": "middle",
      style: "fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums",
    });
    lbl.textContent = fmtNum(t);
    svg.append(lbl);
  }
  if (spec.x_label) {
    const lbl = svgEl("text", {
      x: m.l + iw / 2, y: H - 2, "text-anchor": "middle",
      style: "fill: var(--ink-2); font-size: 11px",
    });
    lbl.textContent = spec.x_label;
    svg.append(lbl);
  }

  names.forEach((name, i) => {
    const ys = spec.series[name];
    const pts = x.map((xv, k) => [xv, ys[k]]).filter((p) => p[1] != null);
    const dAttr = pts.map((p, k) =>
      `${k ? "L" : "M"}${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`).join("");
    svg.append(svgEl("path", {
      d: dAttr, fill: "none",
      style: `stroke: ${seriesColor(i)}; stroke-width: 2; ` +
             "stroke-linecap: round; stroke-linejoin: round",
    }));
    const last = pts[pts.length - 1];
    if (last) {
      svg.append(svgEl("circle", {
        cx: px(last[0]), cy: py(last[1]), r: 4,
        style: `fill: ${seriesColor(i)}; stroke: var(--surface); stroke-width: 2`,
      }));
    }
  });

  // crosshair + all-series tooltip: snap the pointer to the nearest x position
  const cross = svgEl("line", {
    y1: m.t, y2: m.t + ih, x1: 0, x2: 0, visibility: "hidden",
    style: "stroke: var(--baseline); stroke-width: 1",
  });
  svg.append(cross);
  const hit = svgEl("rect", {
    x: m.l, y: m.t, width: iw, height: ih, fill: "transparent",
  });
  hit.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const mx = ((e.clientX - r.left) / r.width) * W;
    let best = 0;
    x.forEach((xv, k) => { if (Math.abs(px(xv) - mx) < Math.abs(px(x[best]) - mx)) best = k; });
    cross.setAttribute("x1", px(x[best]));
    cross.setAttribute("x2", px(x[best]));
    cross.setAttribute("visibility", "visible");
    showTooltip(e.clientX, e.clientY, (tt) => {
      tt.append(el("div", "tt-title", `${spec.x_label || "x"} = ${fmtNum(x[best])}`));
      names.forEach((name, i) => {
        const row = el("div", "tt-row");
        const key = el("span", "tt-key");
        key.style.borderTopColor = seriesColor(i);
        row.append(key, el("span", "tt-val", fmtNum(spec.series[name][best])),
                   el("span", "tt-name", name));
        tt.append(row);
      });
    });
  });
  hit.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden");
    hideTooltip();
  });
  svg.append(hit);

  const wrap = el("div", "chart-wrap");
  wrap.append(svg);
  block.append(wrap);
  block.append(dataTable(
    [spec.x_label || "x", ...names],
    x.map((xv, k) => [xv, ...names.map((n) => spec.series[n][k])])));
  return block;
}

/* spec: {title, y_label?, labels: [..], values: [..], errors?: [..]} */
function barChart(spec) {
  const block = chartBlock(spec.title);
  const labels = spec.labels || [], values = spec.values || [];
  if (!labels.length) {
    block.append(el("p", "muted", "chart has no data"));
    return block;
  }
  const errs = spec.errors || [];
  const W = 720, m = { l: 52, r: 20, t: 24, b: 40 };
  const rowH = Math.max(52, 300 / labels.length);
  const H = m.t + m.b + Math.min(300, rowH * labels.length);
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const hi = Math.max(...values.map((v, i) => v + (errs[i] || 0)), 0);
  const lo = Math.min(...values.map((v, i) => v - (errs[i] || 0)), 0);
  const yTicks = niceTicks(lo, hi);
  const y0 = yTicks[0], y1 = yTicks[yTicks.length - 1];
  const py = (v) => m.t + ih - ((v - y0) / (y1 - y0)) * ih;
  const band = iw / labels.length;
  const barW = Math.min(24, band * 0.5);

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  for (const t of yTicks) {
    svg.append(svgEl("line", {
      x1: m.l, x2: W - m.r, y1: py(t), y2: py(t),
      style: "stroke: var(--grid); stroke-width: 1",
    }));
    const lbl = svgEl("text", {
      x: m.l - 8, y: py(t) + 4, "text-anchor": "end",
      style: "fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums",
    });
    lbl.textContent = fmtNum(t);
    svg.append(lbl);
  }

  values.forEach((v, i) => {
    const cx = m.l + band * i + band / 2;
    const yTop = py(Math.max(v, 0)), yBase = py(Math.max(y0, 0));
    const h = Math.max(1, Math.abs(yBase - yTop));
    const rr = Math.min(4, barW / 2, h);
    const xL = cx - barW / 2;
    // rounded at the data end, square at the baseline
    const dAttr = `M${xL},${yBase} V${yTop + rr} Q${xL},${yTop} ${xL + rr},${yTop} ` +
      `H${xL + barW - rr} Q${xL + barW},${yTop} ${xL + barW},${yTop + rr} V${yBase} Z`;
    const bar = svgEl("path", { d: dAttr, style: `fill: ${seriesColor(0)}` });
    svg.append(bar);
    if (errs[i]) {
      const e1 = py(v - errs[i]), e2 = py(v + errs[i]);
      svg.append(svgEl("line", {
        x1: cx, x2: cx, y1: e1, y2: e2,
        style: "stroke: var(--ink-2); stroke-width: 1.5",
      }));
      for (const ey of [e1, e2]) {
        svg.append(svgEl("line", {
          x1: cx - 4, x2: cx + 4, y1: ey, y2: ey,
          style: "stroke: var(--ink-2); stroke-width: 1.5",
        }));
      }
    }
    // direct value label on the cap (relief for the sub-3:1 hues), in ink not series color
    const vl = svgEl("text", {
      x: cx, y: (errs[i] ? py(v + errs[i]) : yTop) - 6, "text-anchor": "middle",
      style: "fill: var(--ink-1); font-size: 12px; font-weight: 600",
    });
    vl.textContent = fmtNum(v) + (errs[i] ? ` ±${fmtNum(errs[i])}` : "");
    svg.append(vl);
    const cl = svgEl("text", {
      x: cx, y: H - m.b + 18, "text-anchor": "middle",
      style: "fill: var(--ink-2); font-size: 11px",
    });
    cl.textContent = truncate(String(labels[i]), Math.floor(band / 6.5));
    svg.append(cl);

    // the whole band is the hit target, not just the painted bar
    const hitArea = svgEl("rect", {
      x: m.l + band * i, y: m.t, width: band, height: ih, fill: "transparent",
    });
    hitArea.addEventListener("pointermove", (e) => {
      bar.style.opacity = "0.8";
      showTooltip(e.clientX, e.clientY, (tt) => {
        tt.append(el("div", "tt-title", String(labels[i])));
        const row = el("div", "tt-row");
        row.append(el("span", "tt-val", fmtNum(v)),
                   el("span", "tt-name", errs[i] ? `± ${fmtNum(errs[i])}` : ""));
        tt.append(row);
      });
    });
    hitArea.addEventListener("pointerleave", () => {
      bar.style.opacity = "";
      hideTooltip();
    });
    svg.append(hitArea);
  });
  svg.append(svgEl("line", {
    x1: m.l, x2: W - m.r, y1: py(Math.max(y0, 0)), y2: py(Math.max(y0, 0)),
    style: "stroke: var(--baseline); stroke-width: 1",
  }));

  const wrap = el("div", "chart-wrap");
  wrap.append(svg);
  block.append(wrap);
  const cols = [spec.y_label || "value"];
  if (errs.length) cols.push("±");
  block.append(dataTable(["", ...cols],
    labels.map((l, i) => errs.length ? [l, values[i], errs[i] ?? ""] : [l, values[i]])));
  return block;
}

function truncate(s, n) {
  return s.length > n ? s.slice(0, Math.max(1, n - 1)) + "…" : s;
}

/* spec: {title, columns: [..], rows: [[..], ..]} */
function tableBlock(spec) {
  const block = chartBlock(spec.title);
  const table = el("table", "metrics-table");
  const trh = el("tr");
  for (const c of spec.columns || []) trh.append(el("th", "", String(c)));
  table.append(trh);
  for (const row of spec.rows || []) {
    const tr = el("tr");
    for (const v of row) tr.append(el("td", "", typeof v === "number" ? fmtNum(v) : String(v)));
    table.append(tr);
  }
  block.append(table);
  return block;
}

init().catch((err) => {
  $("#cfg-pills").textContent = "";
  $("#cfg-pills").append(el("span", "muted", `Failed to reach server: ${err.message}`));
});
