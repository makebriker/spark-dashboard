/* SparkView Dashboard — multi-node board, built from /api/board.
 *
 * Every node gets its own copy of each configured card. Cards are found by
 * data-f attributes scoped to their node, never by global id, so the same
 * markup can appear up to four times on one page. A node's DOM is rebuilt only
 * when its card list or the config changes; otherwise each refresh just
 * updates values in place.
 */

const SPARK_POINTS = 60;
let refreshMs = 2000;
let configKey = '';
const nodeState = new Map();   // node key -> {key, el, cards, history}
let inference = null;          // board-wide vLLM band when vllm.placement = "top"

/* ---------- helpers ---------- */

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function barClass(pct) {
  if (pct >= 85) return 'red';
  if (pct >= 70) return 'orange';
  if (pct >= 50) return 'yellow';
  return 'green';
}

function fmt(bytes) {
  if (bytes == null) return 'N/A';
  return (bytes / (1024 ** 3)).toFixed(1) + ' GiB';
}

function fmtRate(bps) {
  if (bps >= 1e9) return (bps / 1e9).toFixed(1) + ' GB/s';
  if (bps >= 1e6) return (bps / 1e6).toFixed(1) + ' MB/s';
  if (bps >= 1e3) return (bps / 1e3).toFixed(1) + ' KB/s';
  return (bps || 0).toFixed(0) + ' B/s';
}

function throttleClass(status) {
  return { PASS: 'green', IDLE: 'dim', LOCKED: 'orange', THROTTLED: 'red' }[status] || 'dim';
}

function psiClass(level) {
  return { LOW: 'green', MOD: 'yellow', HIGH: 'orange', CRITICAL: 'red' }[level] || 'dim';
}

function umaClass(level) {
  if (level === 'HIGH' || level === 'CRITICAL') return 'red';
  if (level === 'MOD') return 'yellow';
  return 'green';
}

// "enP7s7", "enp*", "wl*" -- shell-style globs for include/exclude lists.
function globMatch(patterns, name) {
  return patterns.some(p => new RegExp('^' + p.split('*').map(
    s => s.replace(/[.+?^${}()|[\]\\]/g, '\\$&')).join('.*') + '$').test(name));
}

function filtered(items, key, opts) {
  return items.filter(it => {
    const name = it[key];
    if (opts.include && opts.include.length && !globMatch(opts.include, name)) return false;
    if (opts.exclude && opts.exclude.length && globMatch(opts.exclude, name)) return false;
    return true;
  });
}

function bar(f) {
  return '<div class="bar-track"><div data-f="' + f + '" class="bar-fill"></div></div>';
}

function metricRow(label, f, extra) {
  return '<div class="metric-row"' + (extra || '') + '><span class="label">' + label + '</span>' +
    bar(f + '-bar') + '<span data-f="' + f + '-text" class="value"></span></div>';
}

/* Scoped field access within one card. */
function fields(root) {
  const cache = {};
  const get = f => cache[f] || (cache[f] = root.querySelector('[data-f="' + f + '"]'));
  return {
    el: get,
    text(f, v) { const e = get(f); if (e) e.textContent = v; },
    html(f, v) { const e = get(f); if (e) e.innerHTML = v; },
    bar(f, pct, cls) {
      const e = get(f);
      if (!e) return;
      pct = Math.max(0, Math.min(100, pct || 0));
      e.style.width = pct.toFixed(1) + '%';
      e.className = 'bar-fill ' + (cls || barClass(pct));
    },
    badge(f, text, cls) {
      const e = get(f);
      if (!e) return;
      e.textContent = text;
      e.className = 'stat badge' + (cls ? ' badge-' + cls : '');
    },
    show(f, on) { const e = get(f); if (e) e.classList.toggle('hidden', !on); },
  };
}

/* ---------- cards ----------
 * Each card: title, build(opts) -> inner HTML, update(F, data, opts, ctx).
 * update() returns false to hide the card (no data for it on this node).
 */

const CARDS = {
  vllm: {
    title: 'INFERENCE · vLLM',
    build: () => '<div data-f="servers"></div>',
    update: updateVLLM,
  },

  gpu: {
    title: 'GPU',
    build: () =>
      metricRow('Utilization', 'util') + metricRow('Memory', 'mem') +
      '<div class="stats-row"><span data-f="temp" class="stat"></span>' +
      '<span data-f="power" class="stat"></span><span data-f="uma" class="stat badge"></span></div>' +
      '<div data-f="procs" class="proc-table"></div>',
    update(F, data, opts) {
      const g = data.gpu && data.gpu[0];
      if (!g) return false;
      const util = g.utilization || 0;
      F.bar('util-bar', util);
      F.text('util-text', util + '%');
      const memPct = g.mem_total ? (g.mem_used || 0) / g.mem_total * 100 : 0;
      F.bar('mem-bar', memPct);
      F.text('mem-text', fmt(g.mem_used) + ' / ' + fmt(g.mem_total));
      F.text('temp', g.temperature != null ? g.temperature.toFixed(0) + '°C' : '—');
      F.text('power', g.power != null ? (g.power / 1000).toFixed(1) + 'W' : '—');
      if (g.is_uma) {
        const psi = data.pressure && data.pressure.mem ? data.pressure.mem.level : 'LOW';
        F.badge('uma', '⚡ UMA', umaClass(psi));
      } else {
        F.badge('uma', '', '');
      }
      const procs = (g.processes || []).slice(0, opts.processes);
      let html = '';
      if (procs.length) {
        html = '<div class="proc-row header"><span class="proc-pid">PID</span><span class="proc-user">USER</span>' +
          '<span class="proc-mem">MEM</span><span class="proc-cpu">CPU%</span><span class="proc-cmd">CMD</span></div>';
        for (const p of procs) {
          html += '<div class="proc-row"><span class="proc-pid">' + esc(p.pid) + '</span>' +
            '<span class="proc-user">' + esc((p.user || '').substring(0, 8)) + '</span>' +
            '<span class="proc-mem">' + fmt(p.gpu_mem) + '</span>' +
            '<span class="proc-cpu">' + (p.cpu_pct || 0).toFixed(0) + '%</span>' +
            '<span class="proc-cmd" title="' + esc(p.cmd) + '">' + esc(p.cmd) + '</span></div>';
        }
      }
      F.html('procs', html);
    },
  },

  memory: {
    title: 'MEMORY',
    build: opts =>
      metricRow('RAM', 'ram') +
      (opts.swap ? metricRow('Swap', 'swap', ' data-f="swap-row"') : '') +
      (opts.psi ? '<div data-f="psi">' + metricRow('PSI some', 'some') + metricRow('PSI full', 'full') +
        '<div class="stats-row"><span data-f="level" class="stat badge"></span></div></div>' : ''),
    update(F, data) {
      const m = data.memory;
      if (!m || m.percent == null) return false;
      F.bar('ram-bar', m.percent);
      F.text('ram-text', fmt(m.used) + ' / ' + fmt(m.total) + '  (' + m.percent.toFixed(1) + '%)');
      F.show('swap-row', m.swap_total > 0);
      if (m.swap_total > 0) {
        F.bar('swap-bar', m.swap_percent, m.swap_percent > 20 ? 'yellow' : 'green');
        F.text('swap-text', fmt(m.swap_used) + ' / ' + fmt(m.swap_total));
      }
      const psi = data.pressure && data.pressure.mem;
      F.show('psi', psi && psi.available);
      if (psi && psi.available) {
        F.bar('some-bar', psi.some_avg10 * 100 / 0.30, psiClass(psi.level));
        F.text('some-text', psi.some_avg10.toFixed(2));
        F.bar('full-bar', psi.full_avg10 * 100 / 0.30, psiClass(psi.level));
        F.text('full-text', psi.full_avg10.toFixed(2));
        F.badge('level', psi.level, psiClass(psi.level));
      }
    },
  },

  cpu: {
    title: 'CPU',
    build: opts =>
      metricRow('Overall', 'all') +
      (opts.per_core ? '<div data-f="cores" class="core-grid"></div>' : '') +
      '<div class="stats-row"><span data-f="temp" class="stat"></span><span data-f="freq" class="stat"></span></div>',
    update(F, data) {
      const c = data.cpu;
      if (!c || c.percent == null) return false;
      F.bar('all-bar', c.percent);
      F.text('all-text', c.percent.toFixed(1) + '%');
      const coresEl = F.el('cores');
      if (coresEl) {
        const colors = { red: 'var(--red)', orange: 'var(--orange)' };
        coresEl.innerHTML = (c.per_core || []).map(pct =>
          '<div class="core-bar"><div class="core-bar-inner" style="width:' + pct.toFixed(1) +
          '%;background:' + (colors[barClass(pct)] || 'var(--accent)') + '"></div></div>').join('');
      }
      F.text('temp', c.temperature != null ? c.temperature.toFixed(0) + '°C' : '');
      F.text('freq', c.freq ? (c.freq.current || 0).toFixed(0) + ' MHz' : '');
    },
  },

  clock: {
    title: 'CLOCK',
    build: () =>
      metricRow('Speed', 'clk') +
      '<div class="stats-row"><span data-f="status" class="stat badge"></span><span data-f="pstate" class="stat"></span></div>' +
      '<div data-f="reasons" class="reasons"></div>',
    update(F, data) {
      const t = data.throttle && data.throttle[0];
      if (!t || !t.available) return false;
      const status = t.status;
      const pct = t.clk_max_mhz > 0 ? t.clk_mhz / t.clk_max_mhz * 100 : 0;
      F.bar('clk-bar', status === 'IDLE' ? 0 : pct, status === 'IDLE' ? 'dim' : throttleClass(status));
      F.text('clk-text', t.clk_mhz != null ? t.clk_mhz.toFixed(0) + ' / ' + t.clk_max_mhz.toFixed(0) + ' MHz' : '—');
      F.badge('status', status, throttleClass(status));
      F.text('pstate', t.pstate || '');
      const reasons = t.throttle_reasons || [];
      F.text('reasons', reasons.length ? '⚠ ' + reasons.join(', ') : '');
    },
  },

  network: {
    title: 'NETWORK',
    build: () => '<div data-f="list"></div>',
    update(F, data, opts) {
      let nets = filtered(data.network || [], 'iface', opts);
      if (opts.hide_down) nets = nets.filter(n => n.state === 'UP');
      if (!nets.length) return false;
      let html = '';
      for (const n of nets) {
        const up = n.state === 'UP';
        const speed = n.speed_mbps >= 1000 ? (n.speed_mbps / 1000) + 'G' : (n.speed_mbps || '?') + 'M';
        html += '<div class="net-iface"><div class="net-header"><span class="net-name">' + esc(n.iface) +
          '</span><span class="net-state ' + (up ? 'up' : 'down') + '">' + esc(n.state) + '</span>' +
          '<span class="net-speed">' + speed + (n.transport === 'rdma' ? ' · RDMA' : '') + '</span></div>';
        if (n.ipv4 && n.ipv4.address) {
          html += '<div class="net-stats"><span>' + esc(n.ipv4.address) + (n.ipv4.prefix ? '/' + n.ipv4.prefix : '') + '</span></div>';
        }
        if (up) {
          html += '<div class="net-stats"><span>TX ' + fmtRate(n.tx_rate) + '</span><span>RX ' + fmtRate(n.rx_rate) + '</span>';
          if (n.rx_errors > 0 || n.tx_errors > 0) {
            html += '<span class="err">ERR rx:' + n.rx_errors + ' tx:' + n.tx_errors + '</span>';
          }
          html += '</div>';
        }
        html += '</div>';
      }
      F.html('list', html);
    },
  },

  disk: {
    title: 'DISK',
    build: () => '<div data-f="list"></div>',
    update(F, data, opts) {
      const disks = filtered(data.disk || [], 'mountpoint', opts);
      if (!disks.length) return false;
      F.html('list', disks.map(d =>
        '<div class="disk-mount"><div class="disk-label">' + esc(d.mountpoint) +
        ' <span class="disk-dev">(' + esc(d.device) + ' ' + esc(d.fstype) + ')</span></div>' +
        '<div class="bar-track"><div class="bar-fill ' + barClass(d.percent) + '" style="width:' + d.percent.toFixed(1) + '%"></div></div>' +
        '<div class="disk-detail">' + fmt(d.used) + ' / ' + fmt(d.total) + '  (' + d.percent.toFixed(1) + '%)</div></div>'
      ).join(''));
    },
  },
};

/* ---------- vLLM ---------- */

const num = (v, d) => v == null ? '—' : v.toFixed(d);
const ms = v => v == null ? '—' : v < 1 ? (v * 1000).toFixed(0) + ' ms' : v.toFixed(2) + ' s';

// key -> [label, formatter, unit-or-null]. Order here is irrelevant: the
// config's vllm.metrics list decides which appear and in what order.
const VLLM_TILES = {
  decode_tps: ['Decode', v => num(v, 1), 'tok/s'],
  stream_tps: ['Per stream', v => num(v, 1), 'tok/s'],
  prefill_tps: ['Prefill', v => num(v, 0), 'tok/s'],
  running: ['Running', v => num(v, 0), null],
  waiting: ['Waiting', v => num(v, 0), null],
  ttft: ['TTFT', ms, null],
  itl: ['TPOT', ms, null],
  e2e: ['E2E latency', ms, null],
  req_rate: ['Requests', v => num(v, 2), 'req/s'],
  prefix_hit: ['Prefix hit', v => num(v, 1), '%'],
  spec_accept: ['Spec accept', v => num(v, 1), '%'],
  spec_len: ['Accept len', v => num(v, 2), 'tok/step'],
  preemptions: ['Preemptions', v => num(v, 0), null],
  max_model_len: ['Context', v => v == null ? '—' : v >= 1024 ? Math.round(v / 1024) + 'k' : String(v), null],
};

function sparkline(points) {
  if (points.length < 2) return '';
  const max = Math.max(...points, 1);
  const w = 200, h = 36;
  const xy = points.map((v, i) =>
    (i / (SPARK_POINTS - 1) * w).toFixed(1) + ',' + (h - 2 - (v / max) * (h - 4)).toFixed(1));
  const last = xy[xy.length - 1].split(',')[0];
  return '<svg class="spark" viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none">' +
    '<polygon class="spark-fill" points="0,' + h + ' ' + xy.join(' ') + ' ' + last + ',' + h + '"/>' +
    '<polyline class="spark-line" points="' + xy.join(' ') + '"/></svg>' +
    '<span class="spark-max">peak ' + max.toFixed(0) + ' tok/s</span>';
}

function updateVLLM(F, data, _opts, ctx) {
  const v = data.vllm;
  const want = ctx.vllm;
  if (want.enabled === false) return false;
  const servers = v ? v.servers : [];
  if (!servers.length) {
    if (want.enabled !== true) return false;
    F.html('servers', '<div class="muted">' + (v ? 'No vLLM server detected' : 'Node does not report vLLM (older dashboard version)') + '</div>');
    return;
  }

  let html = '';
  servers.forEach((s, i) => {
    html += '<div class="vllm-server">';
    if (!s.ok) {
      html += '<div class="vllm-head"><span class="net-state down">DOWN</span><span class="vllm-ep">' +
        esc(s.endpoint) + '</span></div><div class="muted">' + esc(s.error) + '</div></div>';
      return;
    }
    const m = s.metrics;
    const active = (m.running || 0) > 0;
    html += '<div class="vllm-main"><div class="vllm-head"><span class="net-state ' + (active ? 'up' : 'idle') + '">' +
      (active ? 'SERVING' : 'IDLE') + '</span><span class="vllm-model" title="' + esc(s.endpoint) + '">' +
      esc(s.model || s.endpoint) + '</span>' + (s.node ? '<span class="vllm-node">on ' + esc(s.node) + '</span>' : '') + '</div>';

    const hkey = 'vllm' + (s.node || '') + i;
    const hist = ctx.history[hkey] || (ctx.history[hkey] = []);
    hist.push(m.decode_tps || 0);
    if (hist.length > SPARK_POINTS) hist.shift();
    if (want.sparkline && want.metrics.includes('decode_tps')) {
      html += '<div class="spark-wrap">' + sparkline(hist) + '</div>';
    }

    if (want.metrics.includes('kv_cache')) {
      const kv = m.kv_cache;
      html += '<div class="metric-row"><span class="label">KV cache</span><div class="bar-track">' +
        '<div class="bar-fill ' + barClass(kv || 0) + '" style="width:' + (kv || 0).toFixed(1) + '%"></div></div>' +
        '<span class="value">' + num(kv, 1) + '%</span></div>';
    }
    html += '</div><div class="vllm-side">';

    html += '<div class="tiles">';
    for (const key of want.metrics) {
      const t = VLLM_TILES[key];
      if (!t) continue;
      const [label, f, unit] = t;
      const val = f(m[key]);
      html += '<div class="tile' + (key === 'decode_tps' ? ' tile-hero' : '') + '"><span class="tile-label">' + label +
        '</span><span class="tile-value">' + val + (unit && val !== '—' ? '<small> ' + unit + '</small>' : '') + '</span></div>';
    }
    html += '</div><div class="vllm-foot">averaged over ' + num(s.window_s, 0) + ' s</div></div></div>';
  });
  F.html('servers', html);
}

/* ---------- nodes ---------- */

function nodeHeader(info, data) {
  const gpu = data.gpu && data.gpu[0];
  return [
    info.gpu_name || (gpu ? gpu.name : ''),
    info.driver ? 'Driver ' + info.driver : '',
    info.cuda ? 'CUDA ' + info.cuda : '',
    info.uptime ? 'Up ' + info.uptime : '',
  ].filter(Boolean).join(' · ');
}

function buildNode(node, cfg, cards) {
  const el = document.createElement('section');
  el.className = 'node';
  let html = '<div class="node-head"><span class="node-dot"></span><span class="node-name"></span>' +
    '<span class="node-meta"></span><span class="node-err"></span></div><div class="node-grid">';
  for (const name of cards) {
    html += '<section class="card" data-card="' + name + '"><h2 class="card-title">' + CARDS[name].title +
      '</h2>' + CARDS[name].build(cfg.cards[name] || {}) + '</section>';
  }
  el.innerHTML = html + '</div>';
  return el;
}

function renderNode(state, node, cfg) {
  const el = state.el;
  const dot = el.querySelector('.node-dot');
  el.querySelector('.node-err').textContent = node.ok ? '' : node.error;
  dot.className = 'node-dot ' + (node.ok ? 'ok' : 'bad');
  if (node.ok) state.seen = true;
  // Down after having answered: keep its last values, greyed. Never answered:
  // there is nothing to grey out, so show only the header and its error.
  el.classList.toggle('stale', !node.ok && !!state.seen);
  el.classList.toggle('never', !node.ok && !state.seen);
  if (!node.ok) {
    if (!el.querySelector('.node-name').textContent) el.querySelector('.node-name').textContent = node.name || node.url;
    return;
  }

  const data = node.data;
  const info = data.info || {};
  el.querySelector('.node-name').textContent = node.name || data.hostname || node.url;
  el.querySelector('.node-meta').textContent = nodeHeader(info, data);
  el.dataset.kernel = info.kernel || '';

  const ctx = { vllm: cfg.vllm, history: state.history };
  for (const card of el.querySelectorAll('.card')) {
    const name = card.dataset.card;
    try {
      const shown = CARDS[name].update(fields(card), data, cfg.cards[name] || {}, ctx);
      card.classList.toggle('hidden', shown === false);
    } catch (err) {
      // One card failing -- a missing key from an older node -- stays local.
      console.error('SparkView: ' + (node.name || node.url) + '/' + name + ' failed to render', err);
    }
  }
}

function renderBoard(payload) {
  const cfg = payload.config;
  const board = document.getElementById('board');
  const key = JSON.stringify(cfg) + '|' + payload.nodes.map(n => n.url + n.name + n.cards).join(',');

  const top = cfg.vllm.placement === 'top' && cfg.board.cards.includes('vllm');

  if (key !== configKey) {
    configKey = key;
    board.innerHTML = '';
    if (top) {
      // One band across every node column, gathering each node's servers.
      inference = { el: document.createElement('section'), history: inference ? inference.history : {} };
      inference.el.className = 'card inference';
      inference.el.innerHTML = '<h2 class="card-title">' + CARDS.vllm.title + '</h2>' + CARDS.vllm.build();
      board.appendChild(inference.el);
    } else {
      inference = null;
    }
    const keep = new Map();
    payload.nodes.forEach((node, i) => {
      const id = i + '|' + node.url;
      const cards = (node.cards || cfg.board.cards).filter(c => CARDS[c] && !(top && c === 'vllm'));
      const prev = nodeState.get(id);
      const state = { el: buildNode(node, cfg, cards), history: prev ? prev.history : {}, seen: false };
      keep.set(id, state);
      board.appendChild(state.el);
    });
    nodeState.clear();
    keep.forEach((v, k) => nodeState.set(k, v));

    board.className = 'board ' + cfg.board.layout + (payload.nodes.length > 1 ? ' multi' : ' single');
    board.style.setProperty('--n', payload.nodes.length);
    refreshMs = cfg.board.refresh_ms;
    setTitle(cfg.board.title);
  }

  payload.nodes.forEach((node, i) => renderNode(nodeState.get(i + '|' + node.url), node, cfg));
  if (inference) renderInference(payload, cfg);

  const ok = payload.nodes.filter(n => n.ok).length;
  document.getElementById('node-count').textContent =
    payload.nodes.length > 1 ? ok + ' / ' + payload.nodes.length + ' nodes up' : '';
  const first = payload.nodes.find(n => n.ok);
  document.getElementById('footer-note').textContent =
    first && first.data.info && first.data.info.kernel ? 'Kernel ' + first.data.info.kernel : '';
  document.getElementById('last-refresh').textContent = 'Last update: ' + new Date().toLocaleTimeString();
}

function renderInference(payload, cfg) {
  const servers = [];
  let reporting = false;
  for (const node of payload.nodes) {
    if (!node.ok || !node.data.vllm) continue;
    reporting = true;
    const name = node.name || node.data.hostname || node.url;
    for (const s of node.data.vllm.servers) servers.push(Object.assign({ node: name }, s));
  }
  const data = { vllm: reporting ? { detected: servers.some(s => s.ok), servers } : undefined };
  try {
    const shown = updateVLLM(fields(inference.el), data, {}, { vllm: cfg.vllm, history: inference.history });
    inference.el.classList.toggle('hidden', shown === false);
  } catch (err) {
    console.error('SparkView: inference band failed to render', err);
  }
}

function setTitle(title) {
  const t = (title || 'SPARKVIEW').trim();
  const logo = document.getElementById('logo');
  // Accent the second word, or the trailing "VIEW" of the default one-word title.
  const sp = t.indexOf(' ');
  const cut = sp > 0 ? sp : t.toUpperCase().endsWith('VIEW') && t.length > 4 ? t.length - 4 : t.length;
  logo.innerHTML = esc(t.slice(0, cut)) + '<span class="logo-accent">' + esc(t.slice(cut)) + '</span>';
  document.title = t;
}

async function tick() {
  const dot = document.getElementById('conn-status');
  try {
    const resp = await fetch('/api/board');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    renderBoard(await resp.json());
    dot.className = 'conn-dot connected';
    dot.title = 'Connected';
  } catch (e) {
    dot.className = 'conn-dot disconnected';
    dot.title = 'Connection lost';
    console.error('Fetch error:', e);
  }
  setTimeout(tick, refreshMs);
}

tick();
