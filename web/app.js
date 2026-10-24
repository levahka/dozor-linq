/* ДОЗОР — интерфейс оператора. Без сборки и внешних CDN: работает в закрытом контуре предприятия. */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const api = async (url, opt = {}) => {
  const r = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opt });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
};
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtT = (ts) => (ts ? new Date(ts * 1000).toLocaleTimeString("ru-RU") : "—");
const fmtDT = (ts) => (ts ? new Date(ts * 1000).toLocaleString("ru-RU") : "—");
const fmtDur = (s) => (s == null ? "идёт" : s < 60 ? `${s.toFixed(0)} с` : `${Math.floor(s / 60)} мин ${Math.round(s % 60)} с`);
const STATUS = { new: "новое", confirmed: "подтверждено", forwarded: "передано мастеру", false: "ложное" };

const state = { cameras: [], rules: [], events: [], open: new Set(), sound: true, metrics: null, jOffset: 0 };

/* ---------------------------------------------------------------- вкладки */
$$("#tabs button").forEach((b) =>
  b.addEventListener("click", () => {
    $$("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
    $$(".tab").forEach((t) => t.classList.toggle("on", t.id === "tab-" + b.dataset.tab));
    const tab = b.dataset.tab;
    if (tab === "journal") loadJournal();
    if (tab === "analytics") loadStats();
    if (tab === "setup") setupInit();
    if (tab === "perf") { renderPerf(); renderEcon(); }
    location.hash = tab;
  })
);
setInterval(() => ($("#clock").textContent = new Date().toLocaleTimeString("ru-RU")), 1000);

/* ---------------------------------------------------------------- мониторинг */
function renderCams() {
  $("#camGrid").innerHTML = state.cameras
    .map(
      (c) => `
    <div class="cam" id="cam-${c.id}">
      <img src="/api/cameras/${c.id}/stream" alt="${esc(c.name)}">
      <span class="live">LIVE</span>
      <div class="cap">
        <div><b>${esc(c.name)}</b><br><small>${esc(c.area || "")} · ${c.id}</small></div>
        <div class="badges"><span class="b" data-m="fps">— к/с</span><span class="b" data-m="lat">— мс</span><span class="b" data-m="ppl">👤 —</span></div>
      </div>
    </div>`
    )
    .join("");
}

function updateCamBadges() {
  const m = state.metrics;
  if (!m) return;
  for (const [id, cm] of Object.entries(m.cameras)) {
    const el = $(`#cam-${id}`);
    if (!el) continue;
    $('[data-m="fps"]', el).textContent = `${cm.fps.toFixed(1)} к/с`;
    $('[data-m="lat"]', el).textContent = `${cm.latency_ms.toFixed(0)} мс`;
    $('[data-m="ppl"]', el).textContent = `👤 ${cm.persons}`;
  }
  const linq = ["linq", "remote", "rabbit", "relay"].includes(m.backend);
  const pb = $("#pBackend");
  pb.textContent = linq ? (m.backend === "relay" || m.backend === "rabbit" ? "LinQ TPU · облачный стенд" : "LinQ TPU") : m.backend === "split" ? "CPU (разрез head/tail)" : "CPU";
  pb.classList.toggle("linq", linq);
  $("#pFps").textContent = `${m.total_fps.toFixed(1)} к/с всего`;
}

function camAlertState() {
  const byCam = {};
  for (const e of state.events) {
    if (!state.open.has(e.id)) continue;
    const cur = byCam[e.camera_id];
    if (!cur || e.severity === "critical") byCam[e.camera_id] = e;
  }
  for (const c of state.cameras) {
    const el = $(`#cam-${c.id}`);
    if (!el) continue;
    const e = byCam[c.id];
    el.classList.toggle("alert-critical", !!e && e.severity === "critical");
    el.classList.toggle("alert-warning", !!e && e.severity !== "critical");
    let ban = $(".banner", el);
    if (e && !ban) {
      ban = document.createElement("span");
      ban.className = "banner";
      el.appendChild(ban);
    }
    if (ban) e ? (ban.textContent = e.title) : ban.remove();
  }
}

function evCard(e, fresh = false) {
  const open = state.open.has(e.id);
  return `<div class="ev ${e.severity} st-${e.status}${fresh ? " fresh" : ""}" data-id="${e.id}">
    <img src="${e.snapshot_url || ""}" loading="lazy" alt="">
    <div>
      <div class="t">${esc(e.title)}</div>
      <div class="m">${esc(e.camera_name)} · ${fmtT(e.alert_ts)}${e.zone ? " · " + esc(e.zone) : ""}</div>
      <div class="m">${open ? '<span class="tag open">идёт</span> ' : ""}<span class="tag ${e.status}">${STATUS[e.status]}</span> ${esc(e.info || "")}</div>
      ${e.status === "new" ? `<div class="acts"><button data-st="confirmed">✓ Подтвердить</button><button data-st="forwarded">→ Мастеру</button><button data-st="false">✕ Ложное</button></div>` : ""}
    </div></div>`;
}

function renderFeed(freshId) {
  const only = $("#onlyNew").checked;
  const list = state.events.filter((e) => !only || e.status === "new").slice(0, 100);
  $("#feed").innerHTML = list.length ? list.map((e) => evCard(e, e.id === freshId)).join("") : '<div class="empty">Нарушений нет</div>';
  camAlertState();
}
$("#onlyNew").addEventListener("change", () => renderFeed());
$("#feed").addEventListener("click", async (ev) => {
  const card = ev.target.closest(".ev");
  if (!card) return;
  const id = +card.dataset.id;
  const st = ev.target.dataset.st;
  if (st) {
    ev.stopPropagation();
    await setStatus(id, st);
  } else openEvent(id);
});

async function setStatus(id, status, comment) {
  const body = { status, by: "оператор" };
  if (comment != null) body.comment = comment;
  const rec = await api(`/api/events/${id}`, { method: "PATCH", body: JSON.stringify(body) });
  upsert(rec);
  renderFeed();
  return rec;
}

function upsert(rec) {
  const i = state.events.findIndex((e) => e.id === rec.id);
  if (i >= 0) state.events[i] = rec;
  else state.events.unshift(rec);
}

/* звук тревоги — WebAudio, без файлов */
let actx;
function beep(critical) {
  if (!state.sound) return;
  try {
    actx = actx || new AudioContext();
    const n = critical ? 3 : 1;
    for (let i = 0; i < n; i++) {
      const o = actx.createOscillator(), g = actx.createGain();
      o.frequency.value = critical ? 880 : 660;
      o.connect(g); g.connect(actx.destination);
      const t = actx.currentTime + i * 0.25;
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.25, t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.18);
      o.start(t); o.stop(t + 0.2);
    }
  } catch (_) {}
}
$("#soundBtn").addEventListener("click", () => {
  state.sound = !state.sound;
  $("#soundBtn").textContent = state.sound ? "🔔" : "🔕";
});

function showAlarm(e) {
  const a = $("#alarm");
  a.innerHTML = `<img src="${e.snapshot_url || ""}" alt=""><div><b>${esc(e.title)}</b>${esc(e.camera_name)} · ${fmtT(e.alert_ts)}<br><small>${esc(e.info || "")}</small></div>
    <button id="alarmOpen">Открыть</button><button id="alarmAck" style="background:transparent;color:#fff;border:1px solid #fff">Принято</button>`;
  a.hidden = false;
  $("#alarmOpen").onclick = () => { a.hidden = true; openEvent(e.id); };
  $("#alarmAck").onclick = () => (a.hidden = true);
  clearTimeout(showAlarm.t);
  showAlarm.t = setTimeout(() => (a.hidden = true), 10000);
}

/* ---------------------------------------------------------------- WebSocket */
function connectWs() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => $("#pWs").classList.add("on");
  ws.onclose = () => {
    $("#pWs").classList.remove("on");
    setTimeout(connectWs, 2000);
  };
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.type === "metrics") {
      state.metrics = msg.metrics;
      updateCamBadges();
      if ($("#tab-perf").classList.contains("on")) renderPerf();
    } else if (msg.type === "event_open") {
      const e = msg.event;
      state.open.add(e.id);
      upsert(e);
      renderFeed(e.id);
      beep(e.severity === "critical");
      // всплывающее окно — только для угрозы жизни; остальные тревоги видны в ленте и рамкой камеры
      if (["fall", "immobile", "proximity"].includes(e.type)) showAlarm(e);
    } else if (msg.type === "event_close") {
      state.open.delete(msg.id);
      const e = state.events.find((x) => x.id === msg.id);
      if (e) e.end_ts = msg.end_ts;
      renderFeed();
    } else if (msg.type === "event_update") {
      upsert(msg.event);
      renderFeed();
    }
  };
}

/* ---------------------------------------------------------------- карточка события */
let curEv = null;
async function openEvent(id) {
  const e = await api(`/api/events/${id}`);
  curEv = e;
  $("#evSev").textContent = e.severity === "critical" ? "КРИТИЧНО" : "ПРЕДУПРЕЖДЕНИЕ";
  $("#evSev").className = "sev " + e.severity;
  $("#evTitle").textContent = `#${e.id} · ${e.title}`;
  const react = e.alert_ts - e.onset_ts;
  $("#evDl").innerHTML = `
    <dt>Камера</dt><dd>${esc(e.camera_name)} (${e.camera_id})</dd>
    <dt>Участок / зона</dt><dd>${esc(e.area || "—")}${e.zone ? " / " + esc(e.zone) : ""}</dd>
    <dt>Начало нарушения</dt><dd>${fmtDT(e.onset_ts)}</dd>
    <dt>Тревога поднята</dt><dd>${fmtDT(e.alert_ts)} <span class="hint">(через ${react.toFixed(1)} с)</span></dd>
    <dt>Окончание</dt><dd>${e.end_ts ? fmtDT(e.end_ts) + ` <span class="hint">(${fmtDur(e.end_ts - e.onset_ts)})</span>` : "продолжается"}</dd>
    <dt>Объект</dt><dd>человек #${e.track_id} · ${esc(e.info || "")}</dd>
    <dt>Статус</dt><dd><span class="tag ${e.status}">${STATUS[e.status]}</span>${e.status_by ? ` <span class="hint">${esc(e.status_by)}, ${fmtT(e.status_ts)}</span>` : ""}</dd>`;
  $("#evComment").value = e.comment || "";
  setMedia("snap");
  $("#evModal").hidden = false;
  history.replaceState(null, "", `#event=${id}`);
}
function setMedia(m) {
  $$(".media-switch button").forEach((b) => b.classList.toggle("on", b.dataset.m === m));
  $("#evImg").src = (m === "clip" ? curEv.clip_url : curEv.snapshot_url) || curEv.snapshot_url || "";
}
$$(".media-switch button").forEach((b) => b.addEventListener("click", () => setMedia(b.dataset.m)));
$("#evClose").onclick = () => { $("#evModal").hidden = true; history.replaceState(null, "", "#"); };
$("#evModal").addEventListener("click", (e) => { if (e.target.id === "evModal") $("#evClose").click(); });
$$(".actions button").forEach((b) =>
  b.addEventListener("click", async () => {
    const rec = await setStatus(curEv.id, b.dataset.st, $("#evComment").value);
    openEvent(rec.id);
    if ($("#tab-journal").classList.contains("on")) loadJournal();
  })
);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#evClose").click(); });

/* ---------------------------------------------------------------- журнал */
function journalQuery() {
  const p = new URLSearchParams();
  const v = (id) => $(id).value;
  if (v("#fCam")) p.set("camera", v("#fCam"));
  if (v("#fType")) p.set("type", v("#fType"));
  if (v("#fZone")) p.set("zone", v("#fZone"));
  if (v("#fStatus")) p.set("status", v("#fStatus"));
  if (v("#fSev")) p.set("severity", v("#fSev"));
  if (v("#fFrom")) p.set("ts_from", new Date(v("#fFrom")).getTime() / 1000);
  if (v("#fTo")) p.set("ts_to", new Date(v("#fTo")).getTime() / 1000);
  return p;
}
async function loadJournal() {
  const p = journalQuery();
  $("#csvBtn").href = "/api/events.csv?" + p.toString();
  p.set("limit", 50);
  p.set("offset", state.jOffset);
  const rows = await api("/api/events?" + p.toString());
  $("#jPage").textContent = state.jOffset / 50 + 1;
  $("#jBody").innerHTML = rows.length
    ? rows
        .map(
          (e) => `<tr data-id="${e.id}">
      <td>${e.id}</td><td><img src="${e.snapshot_url || ""}" loading="lazy" alt=""></td>
      <td>${fmtDT(e.alert_ts)}</td>
      <td><span class="sevdot ${e.severity}"></span>${esc(e.title)}</td>
      <td>${esc(e.camera_name)}<br><span class="hint">${esc(e.area || "")}</span></td>
      <td>${esc(e.zone || "—")}</td>
      <td>${(e.alert_ts - e.onset_ts).toFixed(1)} с</td>
      <td>${e.end_ts ? fmtDur(e.end_ts - e.onset_ts) : "идёт"}</td>
      <td><span class="tag ${e.status}">${STATUS[e.status]}</span></td></tr>`
        )
        .join("")
    : `<tr><td colspan="9" class="empty">Нет событий по фильтру</td></tr>`;
}
$("#jBody").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-id]"); if (tr) openEvent(+tr.dataset.id); });
$("#fApply").onclick = () => { state.jOffset = 0; loadJournal(); };
$("#jPrev").onclick = () => { state.jOffset = Math.max(0, state.jOffset - 50); loadJournal(); };
$("#jNext").onclick = () => { state.jOffset += 50; loadJournal(); };

/* ---------------------------------------------------------------- аналитика (SVG без библиотек) */
function hbars(rows, label, val, crit) {
  if (!rows.length) return '<div class="empty">Нет данных</div>';
  const max = Math.max(...rows.map(val), 1);
  return rows
    .map((r) => `<div class="hbar"><span>${esc(label(r))}</span><div class="bar ${crit && crit(r) ? "crit" : ""}" style="width:${(100 * val(r)) / max}%"></div><span class="n">${val(r)}</span></div>`)
    .join("");
}
function vbars(labels, values, h = 160) {
  const W = 900, pad = 24, n = labels.length, bw = (W - pad * 2) / n, max = Math.max(...values, 1);
  let s = `<svg viewBox="0 0 ${W} ${h + 30}" width="100%" preserveAspectRatio="none">`;
  values.forEach((v, i) => {
    const bh = (h * v) / max, x = pad + i * bw;
    s += `<rect x="${x + 2}" y="${h - bh + 4}" width="${bw - 4}" height="${bh}" rx="3" fill="var(--accent)" opacity="${v ? 1 : 0.15}"><title>${labels[i]}: ${v}</title></rect>`;
    if (v) s += `<text x="${x + bw / 2}" y="${h - bh}" text-anchor="middle">${v}</text>`;
    if (n <= 31 || i % Math.ceil(n / 31) === 0) s += `<text x="${x + bw / 2}" y="${h + 22}" text-anchor="middle">${labels[i]}</text>`;
  });
  return s + "</svg>";
}
async function loadStats() {
  const st = await api(`/api/stats?days=${$("#aDays").value}`);
  const crit = st.by_type.filter((r) => ["fall", "immobile", "proximity", "zone"].includes(r.type)).reduce((a, r) => a + r.n, 0);
  const lat = st.latency.length ? st.latency.reduce((a, r) => a + r.avg_s, 0) / st.latency.length : null;
  const top = st.by_zone.find((z) => z.zone !== "—");
  $("#kpis").innerHTML = `
    <div class="kpi acc"><div class="v">${st.total}</div><div class="l">нарушений за период</div></div>
    <div class="kpi crit"><div class="v">${crit}</div><div class="l">из них критичных</div></div>
    <div class="kpi"><div class="v">${lat == null ? "—" : lat.toFixed(1) + " с"}</div><div class="l">ср. время от начала нарушения до тревоги</div></div>
    <div class="kpi"><div class="v">${st.precision == null ? "—" : Math.round(st.precision * 100) + "%"}</div><div class="l">точность (по отметкам оператора)</div></div>
    <div class="kpi"><div class="v" style="font-size:18px">${top ? esc(top.zone) : "—"}</div><div class="l">самая проблемная зона</div></div>`;
  $("#chType").innerHTML = hbars(st.by_type, (r) => r.title || r.type, (r) => r.n, (r) => ["fall", "immobile", "proximity", "zone"].includes(r.type));
  $("#chCam").innerHTML = hbars(st.by_camera, (r) => r.camera_name, (r) => r.n);
  const hours = Array.from({ length: 24 }, (_, h) => (st.by_hour.find((r) => r.h === h) || { n: 0 }).n);
  $("#chHour").innerHTML = vbars(Array.from({ length: 24 }, (_, h) => `${h}`), hours);
  $("#chDay").innerHTML = st.by_day.length ? vbars(st.by_day.map((r) => r.d.slice(5)), st.by_day.map((r) => r.n)) : '<div class="empty">Нет данных</div>';
  $("#chZone").innerHTML = hbars(st.by_zone, (r) => r.zone, (r) => r.n);
  const tt = Object.fromEntries(state.rules.map((r) => [r.type, r.title]));
  $("#chPrec").innerHTML = st.precision_by_type.length
    ? `<table class="grid"><tr><th>Нарушение</th><th>Верно</th><th>Ложно</th><th>Точность</th></tr>${st.precision_by_type
        .map((r) => `<tr><td>${esc(tt[r.type] || r.type)}</td><td>${r.ok}</td><td>${r.bad}</td><td>${r.precision == null ? "—" : Math.round(r.precision * 100) + "%"}</td></tr>`)
        .join("")}</table><p class="hint">Оператор отмечает события «подтверждено / ложное» — так система сама меряет точность и копит данные для дообучения.</p>`
    : '<div class="empty">Нет отметок</div>';
}
$("#aDays").onchange = loadStats;

/* ---------------------------------------------------------------- настройка камер и зон */
const DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"];
const PARAM_LABELS = {
  confirm_s: "Подтверждать через, с", clear_s: "Закрывать через, с", cooldown_s: "Пауза повтора, с",
  speed_thr: "Порог скорости, роста/с", window_s: "Окно неподвижности, с", margin: "Дистанция, ростов",
  angle_thr: "Наклон корпуса, °", near: "Телефон у руки/уха, роста",
};
let S = null; // редактируемая копия камеры
let selZone = -1, drag = null;

async function setupInit() {
  if (!$("#sCam").options.length) {
    $("#sCam").innerHTML = state.cameras.map((c) => `<option value="${c.id}">${esc(c.name)}</option>`).join("");
    $("#sCam").onchange = () => setupLoad();
  }
  setupLoad();
}
function setupLoad() {
  const cam = state.cameras.find((c) => c.id === $("#sCam").value);
  S = JSON.parse(JSON.stringify(cam));
  S.zones = S.zones || [];
  S.rules = S.rules || {};
  selZone = S.zones.length ? 0 : -1;
  $("#sName").value = S.name || "";
  $("#sArea").value = S.area || "";
  $("#sEvery").value = S.detect_every || 2;
  $("#sImg").src = `/api/cameras/${S.id}/frame.jpg?raw=1&t=${Date.now()}`;
  $("#sMsg").textContent = "";
  renderZones();
  renderRules();
}
$("#sReload").onclick = () => ($("#sImg").src = `/api/cameras/${S.id}/frame.jpg?raw=1&t=${Date.now()}`);
$("#sImg").onload = drawEditor;
window.addEventListener("resize", drawEditor);

function renderZones() {
  $("#zList").innerHTML = S.zones.length
    ? S.zones
        .map((z, i) => {
          const sc = z.schedule || {};
          return `<div class="zone-item ${i === selZone ? "sel" : ""}" data-i="${i}">
      <div class="zr"><input type="text" data-k="name" value="${esc(z.name)}">
        <select data-k="kind"><option value="danger" ${z.kind === "danger" ? "selected" : ""}>Опасная (вход запрещён)</option><option value="area" ${z.kind === "area" ? "selected" : ""}>Участок (для правил)</option></select>
        <button class="sm ghost" data-act="del" title="Удалить">🗑</button></div>
      <div class="zr"><select data-k="severity"><option value="critical" ${z.severity === "critical" ? "selected" : ""}>критично</option><option value="warning" ${z.severity === "warning" ? "selected" : ""}>предупреждение</option></select>
        <label class="hint">активна с <input type="time" data-k="from" value="${sc.from || ""}"> по <input type="time" data-k="to" value="${sc.to || ""}"></label></div>
      <div class="days">${DAYS.map((d, k) => `<label><input type="checkbox" data-day="${k + 1}" ${!sc.days || sc.days.includes(k + 1) ? "checked" : ""}>${d}</label>`).join("")}
        <span class="hint">· точек: ${z.polygon.length}${i === selZone ? " (редактируется)" : ""}</span></div></div>`;
        })
        .join("")
    : '<div class="hint">Зон нет. Нажмите «+ новая зона» и обведите область на кадре.</div>';
  drawEditor();
}
$("#zList").addEventListener("click", (e) => {
  const it = e.target.closest(".zone-item");
  if (!it) return;
  const i = +it.dataset.i;
  if (e.target.dataset.act === "del") {
    S.zones.splice(i, 1);
    selZone = Math.min(selZone, S.zones.length - 1);
    return renderZones();
  }
  if (selZone !== i && !["INPUT", "SELECT"].includes(e.target.tagName)) {
    selZone = i;
    renderZones();
  }
});
$("#zList").addEventListener("change", (e) => {
  const it = e.target.closest(".zone-item");
  const z = S.zones[+it.dataset.i];
  const k = e.target.dataset.k;
  if (k === "name" || k === "kind" || k === "severity") z[k] = e.target.value;
  const from = $('[data-k="from"]', it).value, to = $('[data-k="to"]', it).value;
  const days = $$("[data-day]", it).filter((x) => x.checked).map((x) => +x.dataset.day);
  z.schedule = from || to || days.length < 7 ? { from: from || "00:00", to: to || "23:59", days } : null;
  drawEditor();
});
$("#zAdd").onclick = () => {
  S.zones.push({ id: "z" + Date.now().toString(36), name: `Зона ${S.zones.length + 1}`, kind: "danger", severity: "critical", polygon: [], schedule: null });
  selZone = S.zones.length - 1;
  renderZones();
};

function drawEditor() {
  const cv = $("#sCanvas"), img = $("#sImg");
  if (!S || !img.clientWidth) return;
  cv.width = img.clientWidth;
  cv.height = img.clientHeight;
  const g = cv.getContext("2d"), W = cv.width, H = cv.height;
  g.clearRect(0, 0, W, H);
  S.zones.forEach((z, i) => {
    if (!z.polygon.length) return;
    const sel = i === selZone;
    g.beginPath();
    z.polygon.forEach(([x, y], k) => (k ? g.lineTo(x * W, y * H) : g.moveTo(x * W, y * H)));
    g.closePath();
    g.fillStyle = z.kind === "danger" ? "rgba(255,77,79,.25)" : "rgba(64,169,255,.2)";
    g.fill();
    g.lineWidth = sel ? 3 : 2;
    g.strokeStyle = sel ? "#c8f031" : z.kind === "danger" ? "#ff4d4f" : "#40a9ff";
    g.stroke();
    if (sel) z.polygon.forEach(([x, y]) => { g.beginPath(); g.arc(x * W, y * H, 6, 0, 7); g.fillStyle = "#c8f031"; g.fill(); });
    g.font = "bold 14px Segoe UI, sans-serif";
    g.fillStyle = "#fff";
    g.fillText(z.name, z.polygon[0][0] * W + 6, z.polygon[0][1] * H - 6);
  });
}
function evPos(e) {
  const r = $("#sCanvas").getBoundingClientRect();
  return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
}
$("#sCanvas").addEventListener("mousedown", (e) => {
  if (selZone < 0) return;
  const [x, y] = evPos(e), z = S.zones[selZone];
  const W = e.target.clientWidth, H = e.target.clientHeight;
  const k = z.polygon.findIndex(([px, py]) => Math.hypot((px - x) * W, (py - y) * H) < 10);
  if (k >= 0) drag = k;
  else {
    z.polygon.push([+x.toFixed(4), +y.toFixed(4)]);
    renderZones();
  }
});
$("#sCanvas").addEventListener("mousemove", (e) => {
  if (drag == null) return;
  const [x, y] = evPos(e);
  S.zones[selZone].polygon[drag] = [+Math.min(1, Math.max(0, x)).toFixed(4), +Math.min(1, Math.max(0, y)).toFixed(4)];
  drawEditor();
});
window.addEventListener("mouseup", () => (drag = null));
$("#sCanvas").addEventListener("contextmenu", (e) => {
  e.preventDefault();
  if (selZone < 0) return;
  const [x, y] = evPos(e), z = S.zones[selZone], W = e.target.clientWidth, H = e.target.clientHeight;
  const k = z.polygon.findIndex(([px, py]) => Math.hypot((px - x) * W, (py - y) * H) < 10);
  if (k >= 0) { z.polygon.splice(k, 1); renderZones(); }
});

function renderRules() {
  $("#rList").innerHTML = state.rules
    .map((r) => {
      const cur = S.rules[r.type];
      const on = !!cur && cur.enabled !== false;
      const params = Object.keys(PARAM_LABELS).filter((k) => k in r.defaults);
      return `<div class="rule-item" data-t="${r.type}">
      <div class="rh"><span class="sevdot ${r.severity}"></span><b>${esc(r.title)}</b>
        <label class="switch"><input type="checkbox" ${on ? "checked" : ""} data-on><span></span></label></div>
      <div class="params">${params.map((k) => `<span>${PARAM_LABELS[k]}</span><input type="number" step="0.1" data-p="${k}" value="${cur && cur[k] != null ? cur[k] : r.defaults[k]}">`).join("")}</div></div>`;
    })
    .join("");
}
$("#rList").addEventListener("change", (e) => {
  const it = e.target.closest(".rule-item");
  const t = it.dataset.t;
  if (e.target.hasAttribute("data-on")) {
    S.rules[t] = { ...(S.rules[t] || {}), enabled: e.target.checked };
  } else if (e.target.dataset.p) {
    S.rules[t] = { ...(S.rules[t] || { enabled: true }), [e.target.dataset.p]: parseFloat(e.target.value) };
  }
});
$("#sSave").onclick = async () => {
  const body = { name: $("#sName").value, area: $("#sArea").value, detect_every: +$("#sEvery").value || 2, zones: S.zones.filter((z) => z.polygon.length >= 3), rules: S.rules };
  try {
    const cam = await api(`/api/cameras/${S.id}/config`, { method: "PUT", body: JSON.stringify(body) });
    const i = state.cameras.findIndex((c) => c.id === cam.id);
    state.cameras[i] = { ...state.cameras[i], ...cam };
    $("#sMsg").textContent = "✓ Сохранено, правила применены без перезапуска";
  } catch (err) {
    $("#sMsg").textContent = "Ошибка: " + err.message;
  }
};

$("#tgTest").onclick = async () => {
  try {
    const r = await api("/api/notify/test", { method: "POST" });
    $("#sMsg").textContent = `✓ Тестовое сообщение отправлено в Telegram (чатов: ${r.chats})`;
  } catch (err) {
    $("#sMsg").textContent = "Telegram не настроен: " + err.message;
  }
};

/* ---------------------------------------------------------------- производительность */
function renderPerf() {
  const m = state.metrics;
  if (!m) return;
  const cams = Object.entries(m.cameras);
  const avg = (f) => (cams.length ? cams.reduce((a, [, c]) => a + f(c), 0) / cams.length : 0);
  const name = (id) => (state.cameras.find((c) => c.id === id) || {}).name || id;
  const linq = ["linq", "remote", "rabbit", "relay"].includes(m.backend);
  $("#perfKpis").innerHTML = `
    <div class="kpi ${linq ? "acc" : ""}"><div class="v" style="font-size:20px">${linq ? "LinQ TPU" : "CPU (" + m.backend + ")"}</div><div class="l">вычислитель нейросетей · ${esc(m.host)}</div></div>
    <div class="kpi"><div class="v">${m.total_fps.toFixed(1)}</div><div class="l">кадров/с обрабатывается суммарно</div></div>
    <div class="kpi"><div class="v">${avg((c) => c.latency_ms).toFixed(0)} мс</div><div class="l">средняя задержка кадр → решение</div></div>
    <div class="kpi"><div class="v">${avg((c) => (c.pose_timing.head_ms ?? c.pose_timing.net_ms ?? 0)).toFixed(1)} мс</div><div class="l">расчёт нейросети позы (${linq ? "на LinQ TPU" : "на CPU"})</div></div>`;
  const t = (o, k) => (o && o[k] != null ? o[k].toFixed(1) : "—");
  $("#perfTable").innerHTML = `<tr><th>Камера</th><th>Источник, к/с</th><th>Обработка, к/с</th><th>Задержка, мс</th><th>Поза: пред / сеть / пост</th><th>${linq ? "Расчёт на TPU / сеть до стенда" : "Head / tail"}</th><th>Детектор, мс</th><th>Людей</th><th>Кадров</th></tr>` +
    cams.map(([id, c]) => `<tr><td>${esc(name(id))}</td><td>${(c.src_fps || 0).toFixed(0)}</td><td>${c.fps.toFixed(1)}</td><td>${c.latency_ms.toFixed(0)}</td>
      <td>${t(c.pose_timing, "pre_ms")} / ${t(c.pose_timing, "net_ms")} / ${t(c.pose_timing, "post_ms")}</td>
      <td>${t(c.pose_timing, "head_ms")} / ${c.pose_timing.network_ms != null ? t(c.pose_timing, "network_ms") : t(c.pose_timing, "tail_ms")}</td><td>${c.detect_ms.toFixed(1)}</td><td>${c.persons}</td><td>${c.frames}</td></tr>`).join("");
  // ёмкость узла: время нейросетей на 1 кадр камеры
  const rate = +$("#capFps").value || 5;
  const every = cams.length ? cams.reduce((a, [id]) => a + (+(state.cameras.find((x) => x.id === id) || {}).detect_every || 2), 0) / cams.length : 2;
  const pose = avg((c) => c.pose_timing.head_ms ?? c.pose_timing.net_ms ?? 0);
  const det = avg((c) => c.detect_timing.head_ms ?? c.detect_timing.net_ms ?? 0);
  const perFrame = pose + det / every;
  const camsPerNode = perFrame ? Math.floor(1000 / (perFrame * rate)) : 0;
  if (linq && camsPerNode) lastCamsPerNode = Math.max(1, Math.floor(camsPerNode * 0.7));   // запас 30 %
  $("#capacity").innerHTML = `<div class="kpis">
    <div class="kpi"><div class="v">${perFrame.toFixed(1)} мс</div><div class="l">нейросети на 1 кадр (поза + детектор/${every.toFixed(0)})</div></div>
    <div class="kpi acc"><div class="v">${camsPerNode}</div><div class="l">камер на один ${linq ? "ускоритель LinQ" : "CPU-узел"} при ${rate} к/с</div></div>
    <div class="kpi"><div class="v">${camsPerNode ? Math.ceil(100 / camsPerNode) : "—"}</div><div class="l">узлов на цех со 100 камерами</div></div></div>
    <p class="hint">Пред- и постобработка (letterbox, NMS, трекинг, правила) идут на CPU хоста параллельно с TPU и в расчёт ёмкости ускорителя не входят.</p>`;
}
$("#capFps").addEventListener("input", () => { renderPerf(); renderEcon(); });

/* экономика: сколько узлов LinQ и операторов нужно, окупаемость */
let lastCamsPerNode = 0;
function renderEcon() {
  const v = (id) => +$(id).value || 0;
  const cams = v("#eCams"), perOp = Math.max(1, v("#ePerOp")), salary = v("#eSalary"), node = v("#eNode"), setup = v("#eSetup"), free = v("#eFree") / 100;
  const perNode = lastCamsPerNode || 10;
  const nodes = Math.ceil(cams / perNode);
  const posts = Math.ceil(cams / perOp);
  const opYear = posts * 4.2 * salary * 1.3 * 12;
  const capex = nodes * node + setup;
  const save = opYear * free;
  const payback = save > 0 ? capex / (save / 12) : Infinity;
  const fmt = (x) => (x >= 1e6 ? (x / 1e6).toFixed(1) + " млн ₽" : Math.round(x / 1e3) + " тыс ₽");
  $("#econ").innerHTML = `<div class="kpis">
    <div class="kpi"><div class="v">${nodes}</div><div class="l">узлов LinQ (по ${perNode} камер на узел — из живых замеров)</div></div>
    <div class="kpi"><div class="v">${fmt(capex)}</div><div class="l">вложения: оборудование + внедрение</div></div>
    <div class="kpi"><div class="v">${posts} пост. / ${fmt(opYear)}</div><div class="l">круглосуточных постов операторов и их стоимость в год сейчас</div></div>
    <div class="kpi acc"><div class="v">${fmt(save)}</div><div class="l">экономия в год (операторы работают с событиями, а не со стеной экранов)</div></div>
    <div class="kpi acc"><div class="v">${isFinite(payback) ? payback.toFixed(1) + " мес" : "—"}</div><div class="l">окупаемость</div></div></div>`;
}
["#eCams", "#ePerOp", "#eSalary", "#eNode", "#eSetup", "#eFree"].forEach((id) => $(id).addEventListener("input", renderEcon));

/* ---------------------------------------------------------------- старт */
(async function init() {
  [state.cameras, state.rules, state.events] = await Promise.all([api("/api/cameras"), api("/api/rules"), api("/api/events?limit=300")]);
  // в ленте мониторинга — только живые события; демо-история видна в журнале и аналитике
  state.events = state.events.filter((e) => !(e.info || "").startsWith("[демо]")).slice(0, 100);
  // открытые сейчас события: без end_ts и моложе часа
  const now = Date.now() / 1000;
  state.events.filter((e) => !e.end_ts && now - e.alert_ts < 3600).forEach((e) => state.open.add(e.id));
  renderCams();
  renderFeed();
  $("#fCam").innerHTML += state.cameras.map((c) => `<option value="${c.id}">${esc(c.name)}</option>`).join("");
  $("#fType").innerHTML += state.rules.map((r) => `<option value="${r.type}">${esc(r.title)}</option>`).join("");
  api("/api/zones").then((zs) => { $("#fZone").innerHTML += zs.map((z) => `<option value="${esc(z)}">${esc(z)}</option>`).join(""); }).catch(() => {});
  connectWs();
  api("/api/notify").then((n) => { if (n.telegram) $("#pTg").hidden = false; }).catch(() => {});
  const h = location.hash.slice(1);
  if (h.startsWith("event=")) openEvent(+h.split("=")[1]);
  else if (h) { const b = $(`#tabs [data-tab="${h}"]`); if (b) b.click(); }
})();
