// teleop web client: vanilla JS, no CDN.  Sends HOLD-state input at 50 Hz; draws JPEG frames + HUD from server telemetry.
"use strict";
const q = new URLSearchParams(location.search);
const token = q.get("token") || "";
const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
const banner = document.getElementById("banner"), telEl = document.getElementById("tel");
let ws = null, hello = null, tel = {}, seq = 0, held = new Set(), padPrev = false;
let langState = null, langRecv = 0;
let rtt = null, ackLat = null, pingId = 0, pingSent = {}, sent = {}, decoding = false, nframes = 0, fpsT = performance.now(), fpsShown = 0;
const HANDLED = new Set();

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws?token=${encodeURIComponent(token)}`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { banner.textContent = "connected - press Enter (or pad Start) to ARM"; };
  ws.onclose = () => { banner.className = ""; banner.textContent = "disconnected (the sim stops; commands are zero). Reload to reconnect."; ws = null; };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      const m = JSON.parse(ev.data);
      if (m.type === "hello") { hello = m; Object.keys(m.kbd_axes).concat(Object.keys(m.kbd_events)).forEach(k => HANDLED.add(k)); initLang(); initPick(); }
      else if (m.type === "tel") { tel = m; onTel(); }
      else if (m.type === "lang") { langState = m; onLang(); }
      else if (m.type === "pong") { const t = pingSent[m.id]; if (t) { rtt = performance.now() - t; delete pingSent[m.id]; } }
      else if (m.type === "bye") { banner.textContent = m.why; }
    } else if (!decoding) {
      decoding = true;
      createImageBitmap(new Blob([ev.data.slice(1)], { type: "image/jpeg" })).then(bm => {
        if (cv.width !== bm.width) { cv.width = bm.width; cv.height = bm.height; }
        ctx.drawImage(bm, 0, 0); bm.close(); decoding = false; nframes++;
      }).catch(() => { decoding = false; });
    }
  };
}

// ---- send policy: 50 Hz while any input is active; one explicit ZERO message on active->idle; 10 Hz idle heartbeat (liveness for the watchdog).
const IDLE_PERIOD = 100;
let kbdWasActive = false, padWasActive = false, lastIdleKbd = 0, lastIdlePad = 0;

// ---- keyboard: keydown sets, keyup clears (HOLD-to-move: each key is its own deadman).  blur / hidden clears everything and sends zero.
function sendKbd() {
  if (!ws || ws.readyState !== 1) return;
  const focus = document.hasFocus() && !document.hidden;
  const s = ++seq; sent[s] = performance.now();
  ws.send(JSON.stringify({ type: "input", src: "kbd", seq: s, t: Date.now(), keys: focus ? [...held] : [], focus }));
}
// typing in the language box must never drive the robot: while a text field has focus the global key handlers ignore the event (held keys are cleared on focus)
function typing(e) { const t = e.target; return !!t && (t.tagName === "TEXTAREA" || t.tagName === "INPUT" || t.isContentEditable); }
window.addEventListener("keydown", e => {
  if (typing(e)) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (HANDLED.has(e.code)) e.preventDefault();
  if (e.repeat) return;
  held.add(e.code); sendKbd();
});
window.addEventListener("keyup", e => { if (typing(e)) return; held.delete(e.code); if (HANDLED.has(e.code)) e.preventDefault(); sendKbd(); });
function clearAll() { held.clear(); sendKbd(); }
window.addEventListener("blur", clearAll);
document.addEventListener("visibilitychange", clearAll);

// ---- gamepad (Gamepad API, 'standard' mapping).  Browser axes: +right / +down; the server inverts to REP-103.
function findPad() {
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  for (const p of pads) if (p && p.connected) return p;
  return null;
}
function padActive(p) {
  return p.axes.slice(0, 4).some(a => Math.abs(a) > 0.05) || p.buttons.some(b => b.pressed || b.value > 0.05);
}
function sendPad(force) {
  if (!ws || ws.readyState !== 1) return;
  const p = findPad();
  if (!p) { if (padPrev) { ws.send(JSON.stringify({ type: "pad_gone" })); padPrev = false; padWasActive = false; } return; }
  padPrev = true;
  const focus = document.hasFocus() && !document.hidden;
  const act = focus && padActive(p), now = performance.now();
  if (!act && !padWasActive && !force && now - lastIdlePad < IDLE_PERIOD) return;
  if (!act) lastIdlePad = now;
  padWasActive = act;                                       // act -> idle transition falls through once: that message is the explicit zero
  const s = ++seq; sent[s] = now;
  ws.send(JSON.stringify({ type: "input", src: "pad", seq: s, t: Date.now(), connected: true, mapping: p.mapping, id: p.id,
    axes: focus ? Array.from(p.axes).slice(0, 4) : [0, 0, 0, 0], buttons: focus ? p.buttons.map(b => b.value) : p.buttons.map(() => 0) }));
}
window.addEventListener("gamepaddisconnected", () => { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "pad_gone" })); padPrev = false; padWasActive = false; });

setInterval(() => {
  const now = performance.now();
  const act = held.size > 0;
  if (act || kbdWasActive || now - lastIdleKbd >= IDLE_PERIOD) { sendKbd(); if (!act) lastIdleKbd = now; }
  kbdWasActive = act;
  sendPad(false);
}, 20);
setInterval(() => { if (ws && ws.readyState === 1) { const id = ++pingId; pingSent[id] = performance.now(); ws.send(JSON.stringify({ type: "ping", id })); } }, 500);

// ---- HUD
function row(k, v, cls) { return `<tr><td class="k">${k}</td><td class="${cls || ""}">${v}</td></tr>`; }
function onTel() {
  // input->telemetry latency from the echoed seq
  const a = tel.ack ? Math.max(tel.ack.kbd || 0, 0) : 0;
  if (sent[a]) { ackLat = performance.now() - sent[a]; for (const k in sent) if (+k <= a) delete sent[k]; }
  if (!tel.ready) { banner.className = ""; banner.textContent = tel.error ? "SIM ERROR: " + tel.error : (tel.msg || "loading simulation (first FL-MPC compile can take ~1 min)..."); return; }
  banner.className = tel.enabled ? "on" : "";
  banner.textContent = tel.enabled ? `ARMED  [${tel.src}]  speed ${tel.preset_name} x${tel.scale.toFixed(2)}   (Space = zero, Esc = latched stop, pad Back = disarm; pad needs LB held)`
    : `DISARMED (${tel.stopped || "-"})  press Enter (or pad Start)`;
  const now = performance.now();
  if (now - fpsT > 1000) { fpsShown = nframes * 1000 / (now - fpsT); nframes = 0; fpsT = now; }
  const rtfc = tel.rtf == null ? "" : tel.rtf < 0.8 ? "warn" : "ok";
  const cup = tel.cup_state === "ok" ? "ok" : "bad";
  const pad = tel.pad || {};
  let h = "";
  h += row("input", `${tel.src} ${tel.watchdog ? '<span class="bad">WATCHDOG (stale)</span>' : ""}  pad: ${pad.connected ? "on (" + pad.mode + (pad.deadman ? "" : ", LB up") + ")" : "none"} ${pad.warn ? '<span class="warn">' + pad.warn + "</span>" : ""}`);
  h += row("shield", `${tel.shield ? "ON" : '<span class="bad">OFF</span>'}  interv ${(100 * tel.interv_frac).toFixed(0)}% (now ${tel.interv_now})  ${tel.shield_label || ""}`);
  h += row("real-time", tel.rtf == null ? "-" : `x${tel.rtf}  (${tel.tick_ms} ms/tick, FL-MPC ${tel.impl})`, rtfc);
  h += row("stream", `${fpsShown.toFixed(1)} fps (server ${tel.render_fps ?? "-"}, render ${tel.render_ms ?? "-"} ms), cam ${tel.cam}${tel.inset ? " +inset" : ""}`);
  h += row("latency", `net RTT ${rtt == null ? "-" : rtt.toFixed(0)} ms, input->ack ${ackLat == null ? "-" : ackLat.toFixed(0)} ms`);
  h += row("cup", `${tel.cup_state}  beads out ${tel.beads_out}/${tel.n_beads}  tilt ${tel.cup_tilt} deg`, cup);
  h += row("clearance", `wall ${tel.wall_clr == null ? "-" : (100 * tel.wall_clr).toFixed(0)} cm, person ${tel.person_clr == null || tel.person_clr > 50 ? "-" : (100 * tel.person_clr).toFixed(0)} cm`,
    (tel.wall_clr != null && tel.wall_clr < 0.1) ? "warn" : "");
  h += row("base", `${tel.speed} m/s, wz ${tel.wz} rad/s, lift ${tel.lift_h} m`);
  h += row("cmd", `op [${tel.cmd_twist}]  safe [${tel.safe_twist}]`);
  const g = tel.ghost || {};
  h += row("ghost", !g.on ? "off (O)" : !g.shown ? "on, on the robot (shield passes the command)" :
    `<span class="${g.warn ? "warn" : ""}">${g.gap}</span><br>requested ${g.requested}<br>permitted ${g.permitted}`, g.shown && g.warn ? "warn" : "");
  h += row("footprint", `${tel.fp_mode}  ${tel.fp_w == null ? "-" : (100 * tel.fp_w).toFixed(0)} cm wide x ${tel.fp_l == null ? "-" : (100 * tel.fp_l).toFixed(0)} cm long  (${tel.fp_ms} ms/tick)   tray ${tel.tuck || "-"}`,
    tel.tuck === "tucked" ? "ok" : "");
  h += row("people", `${tel.people}`);
  h += row("tray", `${tel.grasped ? "GRASPED" : "free"}  can_grasp ${tel.can_grasp ? "yes" : "no"}  bumps ${tel.bumps}  ${tel.infeasible ? '<span class="warn">QP infeasible</span>' : ""}`);
  h += row("sim", `t=${tel.t}s  ${tel.scene}/${tel.task}  pose [${tel.pose}]`);
  telEl.innerHTML = h;
  const ts = document.getElementById("tuck_st");
  if (ts) ts.textContent = tel.tuck ? "tray: " + tel.tuck : "";
  updateAssist(tel);
  updateEpisode(tel);
  updatePick(tel);
}
// ---- assisted friction grasp (only when the server runs --grasp-mode friction: telemetry carries `assist`; otherwise the panel stays hidden).
// The buttons tap the same keys as G / H (one 120 ms press through the normal keyboard path: arming, watchdog and the deadman logic are unchanged).
function tapKey(code) { held.add(code); sendKbd(); setTimeout(() => { held.delete(code); sendKbd(); }, 120); }
function updateAssist(tel) {
  const box = document.getElementById("assist");
  if (!box) return;
  if (!tel.assist) { box.style.display = "none"; return; }
  box.style.display = "block";
  const a = tel.assist;
  const busy = ["PRE", "APPROACH", "CLOSE", "VERIFY", "PLACE_LOWER", "PLACE_OPEN", "PLACE_RETREAT", "PLACE_SETTLE"].includes(a.state);
  document.getElementById("assist_btn").disabled = busy || a.state === "HOLD" || !tel.enabled;
  document.getElementById("place_btn").disabled = !(a.state === "HOLD" || a.state.startsWith("PLACE"));
  document.getElementById("assist_st").innerHTML = `state <b>${a.state}</b> attempt ${a.attempt} (${a.t_state}s)${a.fail ? ' <span class="warn">' + esc(a.fail) + "</span>" : ""}` +
    `  ALIGN ${tel.can_grasp ? "within tolerance" : "not aligned"}${a.place === true ? '  <span class="ok">placed</span>' : a.place === false ? '  <span class="bad">place failed</span>' : ""}`;
}
document.addEventListener("DOMContentLoaded", () => {
  const ab = document.getElementById("assist_btn"), pb = document.getElementById("place_btn");
  const tb = document.getElementById("tuck_btn"), cb = document.getElementById("carry_btn");
  if (tb) tb.onclick = () => { tapKey("KeyN"); tb.blur(); };
  if (cb) cb.onclick = () => { tapKey("KeyM"); cb.blur(); };
  if (ab) ab.onclick = () => tapKey("KeyG");
  if (pb) pb.onclick = () => tapKey("KeyH");
});
// ---- language layer panel (only when the server was started with --lang: the hello message carries `lang`; otherwise the panel stays hidden)
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function initLang() {
  const box = document.getElementById("lang");
  if (!box || !hello || !hello.lang) return;
  box.style.display = "block";
  const ta = document.getElementById("lang_in"), btn = document.getElementById("lang_go"), clr = document.getElementById("lang_clr");
  ta.addEventListener("focus", () => { clearAll(); });                    // stop any held key before typing
  const go = () => { if (ws && ws.readyState === 1 && ta.value.trim()) ws.send(JSON.stringify({ type: "lang", instruction: ta.value.trim() })); };
  btn.onclick = go;
  clr.onclick = () => { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "lang_clear" })); };
  ta.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); go(); } else if (e.key === "Escape") ta.blur(); });
}
function onLang() {
  const out = document.getElementById("lang_out");
  if (!out || !langState) return;
  langRecv = performance.now();
  const L = langState, h = L.hud;
  let t = "";
  if (L.status === "running") t += `<div class="warn">asking the VLM ... (${esc(L.instruction || "")})</div>`;
  else if (L.status === "error") t += `<div class="bad">VLM layer error: ${esc(L.error || "")}</div>`;
  else if (L.status === "idle") t += `<div>defaults: hand-written tray spec, mu [0.18, 0.35]</div>`;
  if (h) {
    const g = Object.entries(h.grounded || {}).map(([k, v]) => `${esc(k)}${v.n > 1 ? " x" + v.n : ""} (${v.conf})`).join(", ") || "-";
    const c = h.checks || {};
    const ok = x => x ? '<span class="ok">ok</span>' : '<span class="bad">FAIL</span>';
    const st = h.spec_status, stc = st === "vlm" ? "ok" : (st === "repaired" ? "warn" : "bad");
    t += `<div><b>instruction</b> ${esc(h.instruction)}</div>`;
    t += `<div><b>grounded</b> (${h.n_marks} marks, ${h.n_unknown} unknown; ${esc(h.source || "")}): ${g}</div>`;
    t += `<div><b>spec</b> <span class="${stc}">${esc(st)}</span>  compile ${ok(c.compile)} eval ${ok(c.eval)} sem ${ok(c.sem)}  backend ${esc(h.backend)}</div>`;
    t += `<pre>${esc(h.spec_pretty || "")}</pre>`;
    if (h.unresolved && h.unresolved.length) t += `<div class="warn">not found in the scene: ${esc(h.unresolved.join(", "))}</div>`;
    if (h.clearances && Object.keys(h.clearances).length) t += `<div><b>clearances</b> ${esc(JSON.stringify(h.clearances))}</div>`;
    if (h.spec_status === "fallback") t += `<div class="bad">${esc((h.spec_reasons || []).join(" | "))}</div>`;
    const m = h.mu || {};
    t += `<div><b>mu set</b> [${m.lo}, ${m.hi}] ${m.center != null ? "(" + m.center + " +- " + m.sigma + ", " + m.n + " samples)" : "(default)"} <span class="${m.source === "vlm" ? "ok" : "warn"}">${esc(m.source)}</span>  (hard-coded [0.18, 0.35], sim truth 0.25)</div>`;
    const lt = h.latency || {};
    t += `<div><b>VLM latency</b> ground ${lt.ground_s}s, author ${lt.author_s}s, mu ${lt.mu_s}s (model); wall ${h.total_wall_s}s; ${lt.n_vlm_calls} calls</div>`;
  }
  if (L.marks_jpg) t += `<img src="data:image/jpeg;base64,${L.marks_jpg}" style="max-width:100%;margin-top:4px;border:1px solid #333">`;
  out.innerHTML = t;
}
// ---- SafeTeleop Beta: episode card (tel.episode; absent => hidden so old servers work unchanged)
function fmtM(k, v) {
  if (v == null) return "-";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return esc(String(v));
}
function updateEpisode(tel) {
  const box = document.getElementById("epi");
  if (!box) return;
  const e = tel.episode;
  if (!e) { box.style.display = "none"; return; }
  box.style.display = "block";
  let h = "";
  if (e.state === "IDLE") {
    box.className = "idle";
    h = "<b>EPISODE: ready</b> - Press Enter / Start to begin";
  } else if (e.state === "ACTIVE") {
    box.className = "active";
    const goal = e.goal == null ? "" : typeof e.goal === "object" ? esc(JSON.stringify(e.goal)) : esc(e.goal);
    h = `<b>EPISODE ACTIVE</b> ${(+e.t_active || 0).toFixed(1)} s` + (goal ? `<br>goal: ${goal}` : "") + `<br><span style="color:#aaa">K = end episode</span>`;
  } else if (e.state === "OUTCOME" && e.outcome) {
    const o = e.outcome, r = String(o.result);
    box.className = r === "success" ? "success" : r === "fail" ? "fail" : "ended";
    h = `<b>EPISODE ${esc(r.toUpperCase())}</b>` + (o.reason ? ` (${esc(o.reason)})` : "") + "<table>";
    for (const k of Object.keys(o.metrics || {})) h += `<tr><td class="k">${esc(k)}</td><td>${fmtM(k, o.metrics[k])}</td></tr>`;
    h += "</table><b>Backspace = new episode</b>";
  } else { box.className = "idle"; h = `<b>EPISODE</b> ${esc(e.state)}`; }
  box.innerHTML = h;
}

// ---- pure box geometry (no DOM; the offline harness loads this block)
// GEOM-BEGIN
function scaleBox(box, frame, dw, dh) {          // frame = [W,H] of the camera image the box is in; returns [x0,y0,x1,y1] in displayed px
  const sx = dw / frame[0], sy = dh / frame[1];
  return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy];
}
function hitBox(proposals, frame, dw, dh, x, y) {  // id of the smallest non-infeasible box under (x,y) in displayed px, else null
  let best = null, bestA = Infinity;
  for (const p of proposals || []) {
    if (!p.feasible || !p.box2d) continue;
    const b = scaleBox(p.box2d, frame, dw, dh);
    if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) {
      const a = (b[2] - b[0]) * (b[3] - b[1]);
      if (a < bestA) { bestA = a; best = p.id; }
    }
  }
  return best;
}
// GEOM-END

// ---- pick panel + box overlay (only when hello.pick is non-null)
let ovCv = null, ovCtx = null;
function sendJ(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }
function initPick() {
  const box = document.getElementById("pick");
  if (!box || !hello || !hello.pick) return;
  box.style.display = "block";
  const sb = document.getElementById("pick_scan"), cb = document.getElementById("pick_cancel");
  // mousedown preventDefault keeps focus on the page (no blur => no stop); blur after click so Space/Enter never re-press a button
  for (const [b, msg] of [[sb, { type: "pick_propose" }], [cb, { type: "pick_cancel" }]]) {
    b.addEventListener("mousedown", e => e.preventDefault());
    b.onclick = () => { sendJ(msg); b.blur(); };
  }
  ovCv = document.getElementById("ov"); ovCtx = ovCv.getContext("2d");
  ovCv.style.display = "block";
  ovCv.addEventListener("mousedown", e => e.preventDefault());
  ovCv.addEventListener("click", e => {
    const P = tel.pick;
    if (!P || P.stale || tel.cam !== "ego") return;
    const r = ovCv.getBoundingClientRect();
    const id = hitBox(P.proposals, P.frame || [640, 360], r.width, r.height, e.clientX - r.left, e.clientY - r.top);
    if (id != null) sendJ({ type: "pick_select", id });
  });
  window.addEventListener("resize", () => updatePick(tel));
}
function syncOverlay() {   // place the overlay exactly over the displayed canvas content (CSS-scaled size, inside the border)
  const w = cv.clientWidth, h = cv.clientHeight;
  ovCv.style.left = (cv.offsetLeft + cv.clientLeft) + "px"; ovCv.style.top = (cv.offsetTop + cv.clientTop) + "px";
  ovCv.style.width = w + "px"; ovCv.style.height = h + "px";
  if (ovCv.width !== w || ovCv.height !== h) { ovCv.width = w; ovCv.height = h; }
  return [w, h];
}
function updatePick(tel) {
  const box = document.getElementById("pick");
  if (!box || !hello || !hello.pick || !ovCv) return;
  const P = tel.pick;
  const [dw, dh] = syncOverlay();
  ovCtx.clearRect(0, 0, dw, dh);
  if (!P) { document.getElementById("pick_st").textContent = "no pick telemetry"; return; }
  const sk = P.skill ? ` skill ${esc(P.skill.state || "")}${P.skill.phase ? " / " + esc(P.skill.phase) : ""}` : "";
  const ready = P.ready_to_pick ? '<span class="ok"><b>READY TO PICK</b></span>' : '<span style="color:#888">not ready</span>';
  let t = `<div>status <b>${esc(P.status)}</b>${sk}  ${ready}${P.stale ? ' <span class="warn">stale (view moved: Scan again)</span>' : ""}</div>`;
  if (P.error) t += `<div class="bad">error: ${esc(P.error)}</div>`;
  if (P.last_reject) t += `<div class="warn">rejected: ${esc(P.last_reject)}</div>`;
  t += `<div style="color:#888">planner ${esc(P.planner || "-")} backend ${esc(P.backend || "-")}</div>`;
  t += "<table>";
  for (const p of P.proposals || []) {
    const sel = P.selected === p.id;
    t += `<tr><td>${sel ? "&gt;" : ""}${esc(p.id)}</td><td>${esc(p.label)}</td><td>${esc(p.mode || "")} ${esc(p.arm || "")}</td>` +
      `<td class="${p.feasible ? "ok" : "bad"}">${p.feasible ? "feasible" : "no: " + esc(p.why || "")}</td></tr>`;
  }
  t += "</table>";
  document.getElementById("pick_st").innerHTML = t;
  if (tel.cam !== "ego") return;
  const fr = P.frame || [640, 360];
  ovCtx.font = "12px ui-monospace,monospace"; ovCtx.textBaseline = "top";
  for (const p of P.proposals || []) {
    if (!p.box2d) continue;
    const b = scaleBox(p.box2d, fr, dw, dh);
    const col = P.stale ? "#999" : p.feasible ? "#3d3" : "#e44";
    ovCtx.strokeStyle = col; ovCtx.fillStyle = col; ovCtx.lineWidth = P.selected === p.id ? 4 : 2;
    ovCtx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
    const lab = `${p.id}:${p.label}` + (!P.stale && !p.feasible && p.why ? ` (${String(p.why).slice(0, 24)})` : "");
    const ty = b[1] >= 14 ? b[1] - 14 : b[1] + 2;
    ovCtx.fillStyle = "rgba(0,0,0,.6)"; ovCtx.fillRect(b[0], ty, ovCtx.measureText(lab).width + 4, 13);
    ovCtx.fillStyle = col; ovCtx.fillText(lab, b[0] + 2, ty + 1);
  }
}
connect();
