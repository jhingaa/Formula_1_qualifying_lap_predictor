// ============================================================================
// F1 Lap Predictor — frontend controller
// ============================================================================
const STATE = { data: null, events: [], maps: {}, current: {}, filter: 'all', view: 'dashboard' };
const WX_ICONS = { air: 'thermometer', track: 'gauge', humidity: 'droplets', wind: 'wind' };

async function getJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(url + ' -> ' + r.status);
  return r.json();
}

// ---- Boot -------------------------------------------------------------------
async function init() {
  wireNav();
  wireControls();
  // One clock drives every countdown on the page. Renderers just stamp a target
  // onto an element, so switching views never leaves a stale timer running.
  setInterval(tickClocks, 1000);
  try {
    STATE.maps = await getJSON('data/track_maps.json').catch(() => ({}));
    STATE.events = await getJSON('/api/events').catch(() => []);
  } catch (e) { /* static mode */ }
  buildSelector();
  // Apply the hash view first so the nav highlight is correct immediately
  // (loadDashboard populates it once data arrives — no Dashboard flash).
  const hash = location.hash.replace('#', '');
  if (hash && document.getElementById('view-' + hash)) switchView(hash);
  await loadDashboard();
  window.addEventListener('hashchange', () => {
    const h = location.hash.replace('#', '');
    if (h && document.getElementById('view-' + h)) switchView(h);
  });
}

async function loadDashboard() {
  const { year, event } = STATE.current;
  const qs = (year && event) ? `?year=${year}&event=${encodeURIComponent(event)}` : '';
  showBusy('Predicting qualifying pace…',
           'Running the practice → qualifying model on this weekend’s practice data.',
           'Still working — the model refits and re-validates leave-one-round-out '
           + 'whenever new practice data has landed.');
  let d;
  try {
    try { d = await getJSON('/api/dashboard' + qs); }
    catch (e) {
      try { d = await getJSON('data/dashboard.json'); }
      catch (e2) { showFatal(e2.message); return; }
    }
    STATE.data = d;
    STATE.current = { year: d.event.year, event: d.event.eventName };  // sync for selector/highlight
    renderAll(d);
    renderActiveView();
    icons();
  } finally { hideBusy(); }
}

// Refresh asks the server to pull anything that finished since its last check,
// then reloads. Falls back to a plain reload in static mode (no API).
async function refreshNow() {
  const btn = document.getElementById('btn-refresh');
  btn.disabled = true;
  const label = btn.innerHTML;
  btn.innerHTML = '<i data-lucide="refresh-cw"></i>Checking…';
  icons();
  showBusy('Checking for new session data…',
           'Asking FastF1 for any session that has finished since the last check.',
           'Downloading timing data — a session that has just finished can take '
           + 'a minute or so to pull.');
  try { await getJSON('/api/refresh'); } catch (e) { /* static mode */ }
  finally { hideBusy(); }
  await loadDashboard();
  btn.disabled = false;
  btn.innerHTML = label;
  icons();
}

// ---- Busy overlay -----------------------------------------------------------
// A warm prediction is near-instant, so the overlay usually just flashes. It
// earns its keep when the server is pulling a session off FastF1 or refitting
// after new practice data — then the page has to say what it is doing rather
// than sit on stale numbers.
const BUSY = { t0: 0, timer: null };
const BUSY_SLOW_S = 4;   // when to swap the detail line for the "why" line

function showBusy(title, detail, slowNote) {
  const el = document.getElementById('busy');
  if (!el) return;
  document.getElementById('busy-title').textContent = title;
  document.getElementById('busy-detail').textContent = detail || '';
  document.getElementById('busy-elapsed').textContent = '';
  el.hidden = false;
  BUSY.t0 = Date.now();
  clearInterval(BUSY.timer);
  BUSY.timer = setInterval(() => {
    const s = Math.round((Date.now() - BUSY.t0) / 1000);
    document.getElementById('busy-elapsed').textContent = `${s}s elapsed`;
    if (s >= BUSY_SLOW_S && slowNote)
      document.getElementById('busy-detail').textContent = slowNote;
  }, 1000);
}

function hideBusy() {
  clearInterval(BUSY.timer);
  const el = document.getElementById('busy');
  if (el) el.hidden = true;
}

function icons() { if (window.lucide) lucide.createIcons(); }

// ---- Countdowns -------------------------------------------------------------
// Any element with data-countdown="<ISO UTC>" is refreshed every second.
// data-cd-style picks the format: "blocks" for the big DD/HH/MM/SS tiles,
// anything else for a compact inline string.
function tickClocks() {
  document.querySelectorAll('[data-countdown]').forEach(el => {
    const ms = new Date(el.dataset.countdown).getTime() - Date.now();
    el.innerHTML = el.dataset.cdStyle === 'blocks' ? cdBlocks(ms) : cdShort(ms);
  });
}
function cdUnits(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  return [Math.floor(s / 86400), Math.floor(s / 3600) % 24,
          Math.floor(s / 60) % 60, s % 60];
}
const pad2 = n => String(n).padStart(2, '0');
function cdBlocks(ms) {
  if (ms <= 0) return `<div class="cd-now">Under way now</div>`;
  const labels = ['DAYS', 'HOURS', 'MINS', 'SECS'];
  return cdUnits(ms).map((v, i) =>
    `<div class="cd-cell"><b>${pad2(v)}</b><span>${labels[i]}</span></div>`).join('');
}
// "FP2 and FP3" / "FP1, FP2 and FP3" — reads better than a bare join in prose.
function list(items) {
  if (!items || !items.length) return '';
  if (items.length === 1) return items[0];
  return items.slice(0, -1).join(', ') + ' and ' + items[items.length - 1];
}

function cdShort(ms) {
  if (ms <= 0) return 'now';
  const [d, h, m, s] = cdUnits(ms);
  return (d ? `${d}d ` : '') + `${pad2(h)}:${pad2(m)}:${pad2(s)}`;
}
function showFatal(msg) {
  document.getElementById('view-dashboard').innerHTML =
    `<div class="card" style="color:#ff6b6b">Could not load data: ${msg}. Run <code>python src/webapp.py</code>.</div>`;
}

// ---- Navigation & controls --------------------------------------------------
function wireNav() {
  document.querySelectorAll('.nav-item').forEach(el =>
    el.addEventListener('click', () => switchView(el.dataset.view)));
  document.body.addEventListener('click', e => {
    const g = e.target.closest('[data-goto]');
    if (g) switchView(g.dataset.goto);
  });
}
function switchView(view) {
  STATE.view = view;
  // Standings aren't tied to a live session, so the Drivers view drops the
  // session/weather/refresh controls and keeps the topbar as a race-week picker.
  document.querySelector('.topbar').classList.toggle('picker-only', view === 'drivers');
  document.querySelectorAll('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.view === view));
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById('view-' + view)?.classList.add('active');
  if (location.hash !== '#' + view) history.replaceState(null, '', '#' + view);
  renderActiveView();
  window.scrollTo(0, 0);
}
function activeView() { return STATE.view || 'dashboard'; }
function renderActiveView() {
  const v = activeView();
  const R = { predictions: renderPredictions, drivers: renderDrivers, circuits: renderCircuits,
              circuit: renderCircuitDetail, data: renderSessionData, model: renderModelInsights,
              live: renderLive, settings: renderSettings, about: renderAbout };
  if (R[v]) R[v]();
  icons();
}

function wireControls() {
  document.getElementById('btn-refresh').addEventListener('click', refreshNow);
  // Circuit selector open/close
  const sel = document.getElementById('selector');
  document.getElementById('sel-btn').addEventListener('click', e => { e.stopPropagation(); sel.classList.toggle('open'); });
  // Driver filter dropdown
  const filt = document.getElementById('drv-filter');
  filt.addEventListener('click', e => { e.stopPropagation(); filt.classList.toggle('open'); });
  document.getElementById('filter-menu').addEventListener('click', e => {
    const a = e.target.closest('[data-filter]'); if (!a) return;
    STATE.filter = a.dataset.filter;
    filt.childNodes[0].nodeValue = 'Show: ' + a.textContent + ' ';
    renderDriverTable(STATE.data.drivers);
  });
  document.addEventListener('click', () => { sel.classList.remove('open'); filt.classList.remove('open'); });
}

function buildSelector() {
  const menu = document.getElementById('sel-menu');
  if (!STATE.events.length) { menu.innerHTML = '<div class="sel-empty">Run the server for circuit list</div>'; return; }
  let html = '', year = null;
  STATE.events.forEach(ev => {
    if (ev.year !== year) { year = ev.year; html += `<div class="sel-year">${year} Season</div>`; }
    const soon = ev.hasData === false;
    html += `<a class="sel-opt${soon ? ' soon' : ''}" data-year="${ev.year}" data-event="${ev.event}">
      <span class="fi fi-${ev.flag || 'un'}"></span>R${ev.round} · ${ev.label}
      <span class="sel-when">${soon ? (ev.date || 'upcoming') : ''}</span></a>`;
  });
  menu.innerHTML = html;
  menu.querySelectorAll('.sel-opt').forEach(a => a.addEventListener('click', () => {
    STATE.current = { year: +a.dataset.year, event: a.dataset.event };
    document.getElementById('selector').classList.remove('open');
    // Stay where the user is: picking a weekend from the Drivers view should
    // reload the standings for it, not bounce them back to the dashboard.
    loadDashboard();
  }));
}

// ============================================================================
// DASHBOARD renderers
// ============================================================================
function renderAll(d) {
  renderHeader(d);
  // A weekend that hasn't run has nothing to predict, so the dashboard panels
  // are swapped out for the schedule and countdown rather than left blank.
  document.getElementById('dash-upcoming').hidden = !d.upcoming;
  document.getElementById('dash-live').hidden = !!d.upcoming;
  if (d.upcoming) { renderUpcoming(d); renderModel(d.model); return; }

  renderPreQuali(d);
  renderWeather(d.weather); renderStepper(d.progress);
  renderTrack(d.track); renderGauge(d.confidence); renderDriverTable(d.drivers);
  renderPoleBattle(d.poleBattle);
  renderSectors(d.sectors, d.idealLapFmt); renderInsights(d.insights);
  renderAccuracy(d.accuracy); renderModel(d.model);
}

function renderHeader(d) {
  document.getElementById('event-name').textContent = d.event.name;
  document.getElementById('event-meta').textContent = `${d.event.date} | ${d.event.circuit}`;
  document.getElementById('event-flag').className = `fi fi-${d.event.flag} event-flag`;
  document.getElementById('sel-label').textContent = d.event.name.replace(/ Grand Prix.*/, ' GP');
  document.getElementById('sp-tag').textContent = d.session.current;
  document.getElementById('weather').hidden = !d.weather;

  const s = d.session, pill = document.querySelector('.session-pill');
  const liveEl = document.querySelector('.sp-live'), endsEl = document.querySelector('.sp-ends');
  // Countdowns are stamped as targets, not baked strings, so the pill ticks.
  const clock = s.targetUtc ? `<b data-countdown="${s.targetUtc}"></b>` : `<b>${s.endsIn}</b>`;
  pill.classList.toggle('is-live', !!s.live);
  if (s.live) {
    document.getElementById('sp-live').textContent = 'Live';
    liveEl.style.display = '';
    endsEl.style.display = '';
    endsEl.innerHTML = `Ends in ${clock}`;
  } else if (s.note === 'Weekend Complete') {
    liveEl.style.display = 'none';
    endsEl.innerHTML = '<b>Weekend Complete</b>';
  } else { // upcoming
    liveEl.style.display = 'none';
    endsEl.innerHTML = `Starts in ${clock}`;
  }
  tickClocks();
}

// A banner only while the prediction is genuinely ahead of the result: practice
// done, qualifying not yet run. Once quali has happened it disappears, because
// then the page is showing a held-out prediction of something already known.
function renderPreQuali(d) {
  const el = document.getElementById('prequali');
  el.hidden = !d.preQuali;
  if (!d.preQuali) return;
  const p = d.practice || { used: [], remaining: [], complete: false };
  const q = (d.progress || []).find(s => s.name === 'QUALI');
  const ahead = q && new Date(q.startUtc).getTime() > Date.now();
  const used = p.used.length ? p.used.join(', ') : 'practice';
  // Laps only, not driver count: practice includes FP1-only rookies who are
  // filtered out of the predicted order, so the two numbers wouldn't agree.
  const evidence = p.laps ? ` (${p.laps} clean laps logged)` : '';
  // Partial practice is still worth predicting from, but it must not read as
  // settled — name what is still to come so the number is taken for what it is.
  const more = p.complete
    ? `all ${p.used.length} practice session${p.used.length === 1 ? '' : 's'}`
    : `${used} so far — ${list(p.remaining)} still to run`;
  el.classList.toggle('partial', !p.complete);
  el.innerHTML = `<i data-lucide="${p.complete ? 'zap' : 'hourglass'}"></i>
    <div><b>Predicted before qualifying.</b> Built from ${more}${evidence}.
    ${p.complete ? '' : 'It will sharpen as those sessions run. '}${ahead
      ? `Qualifying starts in <span data-countdown="${q.startUtc}"></span> (${q.localTime}).`
      : ''}</div>`;
  tickClocks();
}

function renderWeather(w) {
  if (!w) return;
  const items = [['Air', w.air + '°C', 'air'], ['Track', w.track + '°C', 'track'],
                 ['Humidity', w.humidity + '%', 'humidity'], ['Wind', w.wind + 'km/h', 'wind']];
  document.getElementById('weather').innerHTML = items.map(([l, v, k]) =>
    `<div class="wx"><i data-lucide="${WX_ICONS[k]}"></i><div><div class="wx-l">${l}</div><div class="wx-v">${v}</div></div></div>`).join('');
}

// ---- Upcoming weekend -------------------------------------------------------
function renderUpcoming(d) {
  const t = d.track, ev = d.event;
  const round = ev.round ? `Round ${ev.round}${ev.totalRounds ? ' of ' + ev.totalRounds : ''}` : '';
  const rows = (d.schedule || []).map(s => `
    <div class="up-sess${s.name === 'RACE' ? ' race' : ''}">
      <span class="up-abbr">${s.name}</span>
      <span class="up-full">${s.full}</span>
      <span class="up-when">${s.localTime || s.date}</span>
      <span class="up-in" data-countdown="${s.startUtc}"></span>
    </div>`).join('');
  const st = d.standings;
  const leader = st && st.drivers && st.drivers.length ? st.drivers[0] : null;

  document.getElementById('dash-upcoming').innerHTML = `
    <div class="card up-hero">
      <div class="up-tag"><i data-lucide="calendar-clock"></i> Upcoming race weekend</div>
      <h2 class="up-title"><span class="fi fi-${ev.flag}"></span>${ev.name}</h2>
      <p class="up-sub">${round} · ${t.name}${t.location && t.location !== '-' ? ' · ' + t.location : ''}</p>
      <div class="up-starts">Track action starts
        <b>${d.startsLocal || ev.date}</b>${d.startsSession ? ` — ${d.startsSession}` : ''}
        <span class="up-tz">(local circuit time)</span></div>
      <div class="up-clock" data-countdown="${d.startsAtUtc}" data-cd-style="blocks"></div>
      ${d.raceAtUtc ? `<div class="up-race">Lights out <span data-countdown="${d.raceAtUtc}"></span> from now</div>` : ''}
    </div>
    <section class="up-grid">
      <div class="card">
        <h3 class="card-title">SESSION SCHEDULE</h3>
        <div class="up-sessions">${rows || '<div class="muted">Schedule unavailable</div>'}</div>
      </div>
      <div class="card">
        <h3 class="card-title">CIRCUIT</h3>
        ${trackMapSvg(t.map, 'up-map')}
        ${[['Location', t.location], ['Length', t.length], ['Turns', t.turns],
           ['Laps', t.laps], ['Type', t.type], ['First Grand Prix', t.firstGP]]
          .filter(([, v]) => v != null && v !== '' && v !== '-')
          .map(([k, v]) => `<div class="tf-row"><span>${k}</span><b>${v}</b></div>`).join('')}
        <button class="link-btn" data-goto="circuit" style="margin-top:12px">Circuit details</button>
      </div>
      <div class="card">
        <h3 class="card-title">GOING INTO THIS ROUND</h3>
        ${leader ? `<div class="up-leader">
            <div class="up-leader-l">Championship leader</div>
            <div class="up-leader-n">${leader.name}</div>
            <div class="up-leader-t" style="color:${leader.teamColor}">${leader.team} · ${leader.points} pts</div>
          </div>` : '<p class="muted">Standings unavailable</p>'}
        <p class="muted" style="margin-top:10px">Predictions appear here once FP1 has run —
          the model needs practice pace before it can rank the field.</p>
        <button class="link-btn" data-goto="drivers" style="margin-top:12px">View full standings</button>
      </div>
    </section>`;
  tickClocks();
}

function renderStepper(steps) {
  const el = document.getElementById('stepper');
  if (!steps || !steps.length) { el.innerHTML = '<div class="muted">Schedule unavailable</div>'; return; }
  el.innerHTML = steps.map(s => {
    const mark = s.state === 'done' ? '✓' : (s.state === 'live' ? '' : '›');
    return `<div class="step ${s.state}"><div class="dot">${mark}</div><div class="nm">${s.name}</div><div class="dt">${s.date}</div></div>`;
  }).join('');
}

// Maps are derived from real position telemetry, so a circuit we've never
// recorded has none — say so rather than drawing a made-up outline.
function trackSvg(map) {
  return (map && map.path) ? `<path d="${map.path}"/>` : '';
}
function trackMapSvg(map, cls) {
  if (!map || !map.path)
    return `<div class="map-none ${cls}">Track map unavailable — no recorded telemetry yet</div>`;
  return `<svg viewBox="${map.viewBox || '0 0 200 120'}" class="${cls}">${trackSvg(map)}</svg>`;
}
function renderTrack(t) {
  const svg = document.getElementById('track-map');
  svg.setAttribute('viewBox', (t.map && t.map.viewBox) || '0 0 200 120');
  svg.innerHTML = trackSvg(t.map);
  const rows = [['Location', t.location], ['Length', t.length], ['Turns', t.turns],
                ['Type', t.type], ['First Grand Prix', t.firstGP]];
  document.getElementById('track-facts').innerHTML = rows.map(([k, v]) =>
    `<div class="tf-row"><span>${k}</span><b>${v}</b></div>`).join('');
}

// Labels for the four measured components behind the headline number.
const CONF_PARTS = {
  accuracy:   ['Circuit accuracy', 'Share of this circuit’s pace spread the model explains, out-of-fold'],
  resolution: ['Order resolution', 'How cleanly the predicted order separates given the error at this circuit'],
  data:       ['Practice data',    'Sessions run, laps completed and whether soft-tyre runs are available'],
  conditions: ['Conditions',       'Dry, representative practice — wet sessions say little about dry pace'],
};

function renderGauge(c) {
  const has = c && c.value != null;
  const pct = has ? Math.max(0, Math.min(100, Number(c.value))) : 0;
  const R = 95, cx = 120, cy = 118;
  // Track and fill share one path so they can never end up on different circles.
  // large-arc-flag stays 0: a half-circle gauge never sweeps more than 180deg.
  const d = `M${cx - R},${cy} A${R},${R} 0 0 1 ${cx + R},${cy}`;
  const len = Math.PI * R;               // exact arc length of the semicircle
  const col = pct >= 80 ? '#2fd968' : (pct >= 60 ? '#f5a623' : '#e10600');
  const fill = pct > 0
    ? `<path d="${d}" fill="none" stroke="${col}" stroke-width="15" stroke-linecap="round"
         stroke-dasharray="${(len * pct / 100).toFixed(2)} ${len.toFixed(2)}"/>`
    : '';
  document.getElementById('gauge').innerHTML = `
    <path d="${d}" fill="none" stroke="#1c2432" stroke-width="15" stroke-linecap="round"/>
    ${fill}
    <text x="${cx}" y="100" text-anchor="middle" font-size="42" font-weight="800" fill="#e6edf3">${has ? Math.round(pct) + '%' : '—'}</text>
    <text x="${cx}" y="122" text-anchor="middle" font-size="13" font-weight="700" fill="${col}">${c ? c.label : ''}</text>
    <text x="${cx - R}" y="138" text-anchor="middle" font-size="11" fill="#5b6472">0%</text>
    <text x="${cx + R}" y="138" text-anchor="middle" font-size="11" fill="#5b6472">100%</text>`;

  const parts = (c && c.parts) || {};
  const weights = (c && c.weights) || {};
  document.getElementById('conf-parts').innerHTML = Object.keys(CONF_PARTS)
    .filter(k => parts[k] != null)
    .map(k => {
      const v = parts[k], [label, why] = CONF_PARTS[k];
      const pc = v >= 80 ? '#2fd968' : (v >= 55 ? '#f5a623' : '#e10600');
      const w = weights[k] != null ? ` (${Math.round(weights[k] * 100)}% weight)` : '';
      return `<div class="cp-row" title="${why}${w}">
        <span class="cp-l">${label}</span>
        <div class="cp-track"><div class="cp-fill" style="width:${v}%;background:${pc}"></div></div>
        <span class="cp-v">${Math.round(v)}</span></div>`;
    }).join('');

  document.getElementById('gauge-note').innerHTML = has
    ? `Measured from ${c.basis}. The model explains <b>${Math.round(parts.accuracy)}%</b> of the
       pace spread here, with a typical error of <b>${c.maePct}%</b> of pole
       (&sigma;<sub>order</sub> ${c.sigmaOrderPct}%).`
    : 'Confidence unavailable — build the practice dataset with '
      + '<code>python dataset.py --kind practice --years 2026</code>.';
}

function renderPoleBattle(pb) {
  const sec = document.getElementById('pole-section');
  if (!pb || !pb.drivers || !pb.drivers.length) { sec.style.display = 'none'; return; }
  sec.style.display = '';

  document.getElementById('pole-teams').innerHTML = pb.teams.map(t =>
    `<span class="pt" title="Season-to-date median qualifying gap to pole">
       <i style="background:${t.color}"></i>${t.team}
       <b>+${t.paceGapPct.toFixed(2)}%</b></span>`).join('');

  const f = pb.favourite;
  document.getElementById('pole-sub').innerHTML = f
    ? `<b style="color:#e6edf3">${f.name}</b> is the pole favourite at
       ${f.polePct.toFixed(0)}% &mdash; ${pb.favouriteWhy || ''}`
    : 'No comparable practice laps';

  // Which practice sessions actually ran (a sprint weekend has FP1 only), so
  // the column count is data-driven rather than fixed at three.
  const cols = pb.sessionsUsed || [];
  sec.style.setProperty('--pb-cols',
    `1.7fr 1.3fr ${cols.map(() => '1fr').join(' ')} 1fr 1.2fr`);
  document.getElementById('pole-head').innerHTML =
    `<span>DRIVER</span><span>TEAM</span>` +
    cols.map(s => `<span class="ta-r">${s}</span>`).join('') +
    `<span class="ta-r">GAP IN ${pb.drivers[0].refSession || 'REF'}</span><span>POLE CHANCE</span>`;

  document.getElementById('pole-body').innerHTML = pb.drivers.map(d => {
    const cells = cols.map(s => {
      const v = d.sessions[s];
      if (!v) return `<span class="ta-r muted2">—</span>`;
      const soft = v.compound === 'SOFT';
      return `<span class="ta-r pb-t" title="${v.compound}, ${v.laps} clean laps">
        ${fmtLap(v.time)}<i class="cmp ${soft ? 'soft' : 'hard'}">${(v.compound || '?')[0]}</i></span>`;
    }).join('');
    const gap = d.refGapPct == null ? '—'
      : (d.refGapPct === 0 ? 'benchmark' : `+${d.refGapPct.toFixed(3)}%`);
    const p = d.polePct == null ? 0 : d.polePct;
    return `<div class="pbrow" style="border-left-color:${d.teamColor}">
      <span class="pb-d"><span class="fi fi-${d.flag}"></span>${d.name}</span>
      <span class="pb-team"><span class="team-bar" style="background:${d.teamColor}"></span>${d.team}</span>
      ${cells}
      <span class="ta-r pb-gap ${d.refGapPct === 0 ? 'best' : ''}">${gap}</span>
      <span class="pb-p"><div class="conf-track"><div class="conf-fill" style="width:${p}%"></div></div>
        <b>${p.toFixed(0)}%</b></span></div>`;
  }).join('');

  document.getElementById('pole-note').innerHTML =
    `Drivers are compared inside a single reference session (<b>${pb.drivers[0].refSession}</b>) because a
     lap is only meaningful against the others set in the same track and fuel conditions. Pole chance comes
     from simulating the full field 20,000 times at this circuit's measured error.`
    + (pb.cliffPct ? ` The top 4 teams are ${pb.cliffPct.toFixed(2)}% clear of ${pb.nextTeam}.` : '');
}

function fmtLap(s) {
  if (s == null) return '—';
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(3).padStart(6, '0')}`;
}

function filteredDrivers(drivers) {
  if (STATE.filter === 'top10') return drivers.slice(0, 10);
  if (STATE.filter === 'top5') return drivers.slice(0, 5);
  return drivers;
}
function renderDriverTable(drivers) {
  document.getElementById('qbody').innerHTML = filteredDrivers(drivers).map(driverRow).join('');
}
function driverRow(d) {
  const outside = d.marker > 0.9;
  const delta = d.delta == null ? '-' : `+${d.delta.toFixed(3)}`;
  const conf = d.confidence == null ? 0 : d.confidence;
  const confTip = d.polePct == null ? ''
    : `Chance of finishing within one place of P${d.pos}. Pole chance: ${d.polePct}%`;
  return `<div class="qrow" style="border-left-color:${d.teamColor}">
    <div class="q-pos">${d.pos}</div>
    <div class="q-driver"><span class="fi fi-${d.flag}"></span><span class="q-name">${d.name}</span></div>
    <div class="q-team"><span class="team-bar" style="background:${d.teamColor}"></span>${d.team}</div>
    <div class="q-range"><span class="lo">${d.rangeLowFmt}</span>
      <div class="range-bar"><span class="mk" style="left:${Math.min(97, Math.max(3, d.marker * 100))}%;background:${outside ? '#e10600' : '#fff'}"></span></div>
      <span class="hi">${d.rangeHighFmt}</span></div>
    <div class="q-pred">${d.predBestFmt}</div>
    <div class="q-conf" title="${confTip}"><span class="conf-pct">${d.confidence == null ? '—' : conf + '%'}</span>
      <div class="conf-track"><div class="conf-fill" style="width:${conf}%"></div></div>
      <span class="q-delta">${delta}</span></div></div>`;
}

function renderSectors(sectors, idealFmt) {
  document.getElementById('sectors').innerHTML = sectors.map(s =>
    `<div class="sector s${s.sector}"><span class="s-name">Sector ${s.sector}</span>
      <span class="s-time">${s.time.toFixed(3)}s</span><span class="s-drv">${s.driver}</span></div>`).join('');
  document.getElementById('ideal-lap').textContent = idealFmt + 's';
}

function renderInsights(insights) {
  const icon = { up: 'trending-up', target: 'target' };
  document.getElementById('insights').innerHTML = insights.map(i =>
    `<div class="insight ${i.icon === 'target' ? 'target' : ''}">
      <div class="ic"><i data-lucide="${icon[i.icon] || 'trending-up'}"></i></div>
      <div><div class="it">${i.title}</div><div class="is">${i.sub}</div></div></div>`).join('');
}

function renderModel(m) {
  document.getElementById('ms-version').textContent = m.version;
  document.getElementById('ms-trained').textContent = m.trainedOn;
  document.getElementById('ms-updated').textContent = m.lastUpdated;
  document.getElementById('ms-next').textContent = m.nextUpdate;
  document.getElementById('ms-status').textContent = m.status;
}

function renderAccuracy(a) {
  const fmt = (v) => v == null ? '—' : v.toFixed(v < 1 ? 3 : 2) + 's';
  document.getElementById('acc-metrics').innerHTML = `
    <div class="acc-metric"><div class="v green">${fmt(a.mae)}</div><div class="l">Mean Absolute Error</div><div class="d">MAE</div></div>
    <div class="acc-metric"><div class="v">${a.withinRange == null ? '—' : a.withinRange + '%'}</div><div class="l">Predictions in Range</div><div class="d">Within ±0.5s</div></div>
    <div class="acc-metric"><div class="v">${fmt(a.rmse)}</div><div class="l">Root Mean Squared Error</div><div class="d">RMSE</div></div>`;
  drawAccChart(document.getElementById('acc-chart'), a.history || []);
}

function drawAccChart(svg, hist) {
  if (!hist.length) { svg.innerHTML = `<text x="260" y="90" text-anchor="middle" fill="#5b6472">No history yet</text>`; return; }
  const W = 520, H = 180, pad = 34;
  const xs = hist.map((_, i) => pad + i * (W - 2 * pad) / Math.max(1, hist.length - 1));
  const maxV = Math.max(...hist.map(h => h.rmse), 0.6);
  const y = v => H - pad - (v / maxV) * (H - 2 * pad);
  const line = (key, color, dash = '') => {
    const pts = hist.map((h, i) => `${xs[i]},${y(h[key])}`).join(' ');
    const dots = hist.map((h, i) => `<circle cx="${xs[i]}" cy="${y(h[key])}" r="3" fill="${color}"/>`).join('');
    return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" ${dash}/>${dots}`;
  };
  const grid = [0, .25, .5, .75, 1].map(f => {
    const yy = H - pad - f * (H - 2 * pad);
    return `<line x1="${pad}" x2="${W - pad}" y1="${yy}" y2="${yy}" stroke="#1c2432"/>
      <text x="${pad - 8}" y="${yy + 3}" text-anchor="end" fill="#5b6472" font-size="10">${(maxV * f).toFixed(1)}</text>`;
  }).join('');
  const labels = hist.map((h, i) => `<text x="${xs[i]}" y="${H - 10}" text-anchor="middle" fill="#5b6472" font-size="10">${h.track}</text>`).join('');
  svg.innerHTML = grid + line('rmse', '#8b95a5', 'stroke-dasharray="5 4"') + line('mae', '#e10600') + labels;
}

// ============================================================================
// OTHER VIEWS
// ============================================================================
function viewHead(title, sub) {
  return `<div class="view-head"><h2>${title}</h2><p>${sub}</p></div>`;
}

// Views that need a predicted order have nothing to draw before FP1.
function notYetCard(d, what) {
  return `<div class="card not-yet">
    <h3 class="card-title">${what} not available yet</h3>
    <p class="muted">${d.event.name} hasn't run. Track action starts
      <b>${d.startsLocal || d.event.date}</b>${d.startsSession ? ` (${d.startsSession})` : ''} —
      <span data-countdown="${d.startsAtUtc}"></span> from now.</p>
    <button class="link-btn" data-goto="dashboard" style="margin-top:12px;width:auto;padding:8px 14px">
      View the countdown</button></div>`;
}

function renderPredictions() {
  const d = STATE.data; if (!d) return;
  if (d.upcoming) {
    document.getElementById('view-predictions').innerHTML =
      viewHead('Predictions', `Predicted qualifying order — ${d.event.name}`) +
      notYetCard(d, 'Predictions');
    tickClocks(); return;
  }
  document.getElementById('view-predictions').innerHTML =
    viewHead('Predictions', `Full predicted qualifying order — ${d.event.name}`) +
    `<div class="card"><div class="qtable"><div class="qhead">
      <span>POS</span><span>DRIVER</span><span>TEAM</span><span>BEST LAP RANGE (S)</span>
      <span class="ta-r">PREDICTED BEST (S)</span><span>CONFIDENCE</span></div>
      ${d.drivers.map(driverRow).join('')}</div></div>`;
}

// The Drivers view is the championship table as it stood at the selected race
// week — not today's — so scrolling back through the season shows the standings
// a fan would have seen at the time.
function renderDrivers() {
  const d = STATE.data; if (!d) return;
  const el = document.getElementById('view-drivers'), st = d.standings;
  if (!st || !st.drivers || !st.drivers.length) {
    el.innerHTML = viewHead('Drivers', `Drivers' Championship — ${d.event.year} season`) +
      `<div class="card">No results cached for ${d.event.year}. Build them with
       <code>python src/standings.py ${d.event.year}</code>.</div>`;
    return;
  }
  const of = st.totalRounds ? ` of ${st.totalRounds}` : '';
  const sub = st.includesThisRound
    ? `Drivers' Championship after Round ${st.round}${of} — ${st.roundName}`
    : `Drivers' Championship going into ${d.event.eventName} — standings after
       Round ${st.round}${of}, ${st.roundName}`;
  el.innerHTML = viewHead('Drivers', sub) +
    `<div class="std-grid">${st.drivers.map(standingCard).join('')}</div>`;
}

function standingCard(x) {
  const move = !x.move ? ''
    : `<span class="std-move ${x.move > 0 ? 'up' : 'down'}"
         title="${Math.abs(x.move)} place${Math.abs(x.move) > 1 ? 's' : ''} ${x.move > 0 ? 'gained' : 'lost'} last round"
       >${x.move > 0 ? '▲' : '▼'}${Math.abs(x.move)}</span>`;
  // The portrait is a transparent cut-out; if it is missing the avatar falls
  // back to the driver's three-letter code rather than an empty circle.
  const img = x.photo
    ? `<img class="std-face" src="${x.photo}" alt="${x.name}" loading="lazy"
         onerror="this.parentNode.classList.add('no-img');this.remove()">` : '';
  const gap = x.pos === 1 ? 'Leader' : `−${x.gapToLeader}`;
  const logo = x.logo
    ? `<img class="std-logo" src="${x.logo}" alt="" onerror="this.remove()">` : '';
  return `<div class="std-card${x.pos <= 3 ? ' p' + x.pos : ''}" style="--tc:${x.teamColor}">
    <div class="std-top"><span class="std-pos">${x.pos}</span>${move}
      <span class="fi fi-${x.flag}" title="${x.name}"></span></div>
    <div class="std-id">
      <div class="std-avatar${x.photo ? '' : ' no-img'}" data-code="${x.code}">${img}</div>
      <div class="std-pts" title="${x.starts} race${x.starts === 1 ? '' : 's'} started">
        <b>${x.points}</b><span>PTS</span></div>
    </div>
    <div class="std-name" title="${x.name}">${x.name}</div>
    <div class="std-team" style="color:${x.teamColor}" title="${x.team}">${logo}<span>${x.team}</span></div>
    <div class="std-bar" title="${x.share}% of the leader's points">
      <i style="width:${x.share}%"></i></div>
    <div class="std-foot">
      <span>${x.wins}<b>W</b></span><span>${x.podiums}<b>POD</b></span>
      <span class="std-gap" title="${x.pos === 1 ? 'Championship leader' : x.gapToLeader + ' points behind the leader'}">${gap}</span>
    </div>
  </div>`;
}

function renderCircuits() {
  const evs = STATE.events.length ? STATE.events : [];
  const season = evs.length ? evs[0].year : '';
  const done = evs.filter(e => e.hasData !== false).length;
  const cards = evs.map(ev => {
    const m = STATE.maps[ev.circuit];
    const map = m ? `<svg viewBox="${m.viewBox}" class="circ-map"><path d="${m.path}"/></svg>`
                  : `<div class="circ-map circ-nomap">no map yet</div>`;
    const active = (ev.event === STATE.current.event) ? ' active' : '';
    // Rounds still to come open the countdown instead of a prediction.
    const soon = ev.hasData === false;
    const meta = soon && ev.startsAtUtc
      ? `<span data-countdown="${ev.startsAtUtc}"></span> away`
      : `Round ${ev.round}`;
    return `<div class="circ-card${active}${soon ? ' soon' : ''}"
        data-year="${ev.year}" data-event="${ev.event}">
      ${soon ? '<span class="circ-chip">UPCOMING</span>' : ''}
      ${map}
      <div class="circ-name"><span class="fi fi-${ev.flag}"></span>${ev.label}</div>
      <div class="circ-year">${ev.date ? `R${ev.round} · ${ev.date}` : `Round ${ev.round}`}</div>
      <div class="circ-in">${meta}</div></div>`;
  }).join('');
  const el = document.getElementById('view-circuits');
  el.innerHTML = viewHead('Circuits',
      `${season} season — ${done} of ${evs.length} rounds run. Select a circuit for its
       predictions, or an upcoming one for the countdown.`) +
    `<div class="circ-grid">${cards || '<div class="card">Run the server to list circuits.</div>'}</div>`;
  el.querySelectorAll('.circ-card').forEach(c => c.addEventListener('click', () => {
    STATE.current = { year: +c.dataset.year, event: c.dataset.event };
    switchView('dashboard'); loadDashboard();
  }));
  tickClocks();
}

function renderCircuitDetail() {
  const d = STATE.data; if (!d) return;
  const t = d.track, m = t.map || {};
  const facts = [['Location', t.location], ['Length', t.length], ['Turns', t.turns],
    ['Laps', t.laps], ['Race Distance', t.raceDistance], ['DRS Zones', t.drsZones],
    ['Lap Record', t.lapRecord ? `${t.lapRecord} — ${t.recordHolder || ''}` : null],
    ['Type', t.type], ['First Grand Prix', t.firstGP]];
  document.getElementById('view-circuit').innerHTML =
    viewHead(t.name, `${d.event.name} · ${t.location}`) +
    `<div class="cd-grid">
      <div class="card cd-map">${trackMapSvg(m, 'cd-svg')}</div>
      <div class="card">
        <h3 class="card-title">CIRCUIT INFORMATION</h3>
        ${facts.filter(([, v]) => v != null && v !== '').map(([k, v]) =>
          `<div class="tf-row"><span>${k}</span><b>${v}</b></div>`).join('')}
        <button class="link-btn" data-goto="dashboard" style="margin-top:14px">← Back to predictions</button>
      </div>
    </div>
    ${t.about ? `<div class="card"><h3 class="card-title">ABOUT THE CIRCUIT</h3><p class="cd-about">${t.about}</p></div>` : ''}`;
}

function renderSessionData() {
  const d = STATE.data; if (!d) return;
  if (d.upcoming) {
    document.getElementById('view-data').innerHTML =
      viewHead('Session Data', `Raw predicted values — ${d.event.name}`) +
      notYetCard(d, 'Session data');
    tickClocks(); return;
  }
  const rows = d.drivers.map(x => `<tr>
    <td>${x.pos}</td><td>${x.code}</td><td>${x.name}</td><td>${x.team}</td>
    <td>${x.compound || '-'}</td><td class="num">${x.predBest.toFixed(3)}</td>
    <td class="num">${x.rangeLow.toFixed(3)}</td><td class="num">${x.rangeHigh.toFixed(3)}</td>
    <td class="num">${x.confidence == null ? '—' : x.confidence + '%'}</td></tr>`).join('');
  document.getElementById('view-data').innerHTML =
    viewHead('Session Data', `Raw predicted values — ${d.event.name}`) +
    `<div class="card"><table class="dtable"><thead><tr>
      <th>POS</th><th>CODE</th><th>DRIVER</th><th>TEAM</th><th>TYRE</th>
      <th>PRED (s)</th><th>LOW</th><th>HIGH</th><th>CONF</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderModelInsights() {
  const d = STATE.data; if (!d) return;
  const a = d.accuracy;
  document.getElementById('view-model').innerHTML =
    viewHead('Model Insights', 'How the qualifying model performs and what drives it') +
    `<section class="grid-3">
      <div class="card kpi"><div class="v green">${a.mae ?? '—'}s</div><div class="l">Mean Absolute Error</div></div>
      <div class="card kpi"><div class="v">${a.rmse ?? '—'}s</div><div class="l">Root Mean Squared Error</div></div>
      <div class="card kpi"><div class="v">${a.withinRange ?? '—'}%</div><div class="l">Within ±0.5s</div></div>
    </section>
    <section class="grid-accuracy"><div class="card">
      <h3 class="card-title">PER-EVENT ERROR (RECENT)</h3>
      <div class="acc-legend"><span><i class="lg red"></i> MAE</span><span><i class="lg grey"></i> RMSE</span></div>
      <svg id="model-chart" viewBox="0 0 520 180"></svg></div>
      <div class="card"><h3 class="card-title">MODEL DETAILS</h3>
        <div class="tf-row"><span>Algorithm</span><b>${d.model.algorithm || '—'}</b></div>
        <div class="tf-row"><span>Encoder</span><b>${d.model.encoder || '—'}</b></div>
        <div class="tf-row"><span>Predicts</span><b>${d.model.target || '—'}</b></div>
        <div class="tf-row"><span>Version</span><b>${d.model.version}</b></div>
        <div class="tf-row"><span>Trained on</span><b>${d.model.trainedOn}</b></div>
        <div class="tf-row"><span>Validation</span><b>${d.model.validation || '—'}</b></div>
        <h3 class="card-title" style="margin-top:16px">FEATURES</h3>
        <div class="chips">${(d.model.features || []).map(f=>`<span class="chip">${f}</span>`).join('')}</div>
      </div></section>`;
  drawAccChart(document.getElementById('model-chart'), a.history || []);
}

function renderLive() {
  const d = STATE.data;
  document.getElementById('view-live').innerHTML =
    viewHead('Live Weekend', 'Track a live session and compare real pace to the model') +
    `<div class="card">
      <div class="live-status"><span class="sp-live"><i class="dot"></i>${d && d.session.live ? d.session.current + ' — Live (demo)' : 'No live session'}</span>
      <span class="muted">Next race weekend: Dutch Grand Prix, 23 Aug 2026</span></div>
      <p class="muted" style="margin:12px 0">During a live session, record the timing stream and watch predicted-vs-actual lap times update in real time:</p>
      <pre class="cmd"># Terminal 1 — record the live stream
python src/live.py record --out data/live/session.txt

# Terminal 2 — live predictions every 30s
python src/live.py monitor --file data/live/session.txt \\
    --year 2026 --round 12 --session R</pre>
      <p class="muted" style="margin-top:12px">Test the same logic now against a finished race:</p>
      <pre class="cmd">python src/live.py replay --year 2024 --round 1 --session R</pre>
    </div>`;
}

function renderSettings() {
  document.getElementById('view-settings').innerHTML =
    viewHead('Settings', 'Display preferences') +
    `<div class="card">
      <div class="set-row"><div><b>Default event</b><p class="muted">${STATE.current.event} ${STATE.current.year}</p></div>
        <button class="link-btn" data-goto="circuits" style="width:auto;padding:8px 14px">Change</button></div>
      <div class="set-row"><div><b>Data source</b><p class="muted">FastF1 (live API) with Ergast metadata</p></div><span class="chip">Connected</span></div>
      <div class="set-row"><div><b>Prediction scenario</b><p class="muted">Representative dry qualifying conditions</p></div><span class="chip">Dry</span></div>
    </div>`;
}

function renderAbout() {
  document.getElementById('view-about').innerHTML =
    viewHead('About', 'F1 Lap Predictor') +
    `<div class="card">
      <p>An ML system that predicts Formula 1 lap times from tyre compound, tyre age, weather, driver/team, circuit and a fuel-load proxy — then tracks live race weekends and compares real pace to the model.</p>
      <div class="tf-row" style="margin-top:12px"><span>Qualifying model</span><b>${(STATE.data && STATE.data.model.algorithm) || '—'} · practice pace + season form</b></div>
      <div class="tf-row"><span>Race model</span><b>Per-lap stint pace (tyre + fuel + weather)</b></div>
      <div class="tf-row"><span>Data</span><b>FastF1 — 2022–2025</b></div>
      <div class="tf-row"><span>Track maps</span><b>Derived from real position telemetry</b></div>
    </div>`;
}

init();
