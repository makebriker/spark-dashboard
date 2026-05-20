/* SparkView Dashboard — auto-refresh and DOM updates */

const REFRESH_MS = 2000;
const BAR_COLORS = {
  green: s => s < 50,
  yellow: s => s >= 50 && s < 70,
  orange: s => s >= 70 && s < 85,
  red: s => s >= 85,
};

function barClass(pct) {
  if (pct >= 85) return 'red';
  if (pct >= 70) return 'orange';
  if (pct >= 50) return 'yellow';
  return 'green';
}

function setBar(id, pct, extraClass) {
  const bar = document.getElementById(id);
  if (!bar) return;
  const cls = extraClass || barClass(pct);
  bar.style.width = pct.toFixed(1) + '%';
  bar.className = 'bar-fill ' + cls;
}

function fmt(bytes) {
  if (bytes == null) return 'N/A';
  const g = bytes / (1024 ** 3);
  return g.toFixed(1) + ' GiB';
}

function fmtRate(bps) {
  if (bps >= 1e9) return (bps / 1e9).toFixed(1) + ' GB/s';
  if (bps >= 1e6) return (bps / 1e6).toFixed(1) + ' MB/s';
  if (bps >= 1e3) return (bps / 1e3).toFixed(1) + ' KB/s';
  return bps.toFixed(0) + ' B/s';
}

function badge(text, cls) {
  const span = document.createElement('span');
  span.className = 'stat badge badge-' + cls;
  span.textContent = text;
  return span;
}

function throttleClass(status) {
  if (status === 'PASS') return 'green';
  if (status === 'IDLE') return 'dim';
  if (status === 'LOCKED') return 'orange';
  if (status === 'THROTTLED') return 'red';
  return 'dim';
}

function psiClass(level) {
  if (level === 'LOW') return 'green';
  if (level === 'MOD') return 'yellow';
  if (level === 'HIGH') return 'orange';
  if (level === 'CRITICAL') return 'red';
  return 'dim';
}

function umaClass(level) {
  if (level === 'HIGH' || level === 'CRITICAL') return 'red';
  if (level === 'MOD') return 'yellow';
  return 'green';
}

function updateGPU(data) {
  const g = data.gpu && data.gpu[0];
  if (!g) {
    document.getElementById('card-gpu').classList.add('hidden');
    return;
  }
  document.getElementById('card-gpu').classList.remove('hidden');

  const util = g.utilization || 0;
  setBar('gpu-util-bar', util);

  document.getElementById('gpu-util-text').textContent = util + '%';

  const memPct = g.mem_total ? ((g.mem_used || 0) / g.mem_total * 100) : 0;
  setBar('gpu-mem-bar', memPct);
  document.getElementById('gpu-mem-text').textContent = fmt(g.mem_used) + ' / ' + fmt(g.mem_total);

  const temp = g.temperature != null ? g.temperature.toFixed(0) + '°C' : '—';
  const power = g.power != null ? (g.power / 1000).toFixed(1) + 'W' : '—';
  document.getElementById('gpu-temp').textContent = temp;
  document.getElementById('gpu-power').textContent = power;

  const umaEl = document.getElementById('gpu-uma');
  if (g.is_uma) {
    const psiLevel = data.pressure && data.pressure.mem ? data.pressure.mem.level : 'LOW';
    umaEl.textContent = '⚡ UMA';
    umaEl.className = 'stat badge badge-' + umaClass(psiLevel);
  } else {
    umaEl.textContent = '';
    umaEl.className = 'stat badge';
  }

  const procEl = document.getElementById('gpu-procs');
  const procs = (g.processes || []).slice(0, 5);
  if (procs.length) {
    let html = '<div class="proc-row header"><span class="proc-pid">PID</span><span class="proc-user">USER</span><span class="proc-mem">MEM</span><span class="proc-cpu">CPU%</span><span class="proc-cmd">CMD</span></div>';
    for (const p of procs) {
      html += `<div class="proc-row"><span class="proc-pid">${p.pid}</span><span class="proc-user">${(p.user||'').substring(0,8)}</span><span class="proc-mem">${fmt(p.gpu_mem)}</span><span class="proc-cpu">${(p.cpu_pct||0).toFixed(0)}%</span><span class="proc-cmd">${(p.cmd||'').substring(0,20)}</span></div>`;
    }
    procEl.innerHTML = html;
  } else {
    procEl.innerHTML = '';
  }
}

function updateMemory(data) {
  const m = data.memory;
  if (!m) return;

  setBar('mem-bar', m.percent);
  document.getElementById('mem-text').textContent = fmt(m.used) + ' / ' + fmt(m.total) + '  (' + m.percent.toFixed(1) + '%)';

  const swapRow = document.getElementById('swap-row');
  if (m.swap_total > 0) {
    swapRow.classList.remove('hidden');
    setBar('swap-bar', m.swap_percent, m.swap_percent > 20 ? 'yellow' : 'green');
    document.getElementById('swap-text').textContent = fmt(m.swap_used) + ' / ' + fmt(m.swap_total);
  } else {
    swapRow.classList.add('hidden');
  }
}

function updateCPU(data) {
  const c = data.cpu;
  if (!c) return;

  setBar('cpu-bar', c.percent);
  document.getElementById('cpu-text').textContent = c.percent.toFixed(1) + '%';

  const coresEl = document.getElementById('cpu-cores');
  const cores = c.per_core || [];
  coresEl.innerHTML = '';
  for (const pct of cores) {
    const outer = document.createElement('div');
    outer.className = 'core-bar';
    const inner = document.createElement('div');
    inner.className = 'core-bar-inner';
    inner.style.width = pct.toFixed(1) + '%';
    inner.style.background = barClass(pct) === 'red' ? 'var(--red)' : barClass(pct) === 'orange' ? 'var(--orange)' : 'var(--accent)';
    outer.appendChild(inner);
    coresEl.appendChild(outer);
  }

  if (c.temperature != null) {
    document.getElementById('cpu-temp').textContent = c.temperature.toFixed(0) + '°C';
  }
  if (c.freq) {
    document.getElementById('cpu-freq').textContent = (c.freq.current || 0).toFixed(0) + ' MHz';
  }
}

function updateThrottle(data) {
  const t = data.throttle && data.throttle[0];
  if (!t || !t.available) {
    document.getElementById('card-throttle').classList.add('hidden');
    return;
  }
  document.getElementById('card-throttle').classList.remove('hidden');

  const clkPct = t.clk_max_mhz > 0 ? (t.clk_mhz / t.clk_max_mhz * 100) : 0;
  const status = t.status;
  const barColor = status === 'IDLE' ? 'dim' : throttleClass(status);
  setBar('clk-bar', status === 'IDLE' ? 0 : clkPct, barColor);

  const clkText = t.clk_mhz != null ? t.clk_mhz.toFixed(0) + ' / ' + t.clk_max_mhz.toFixed(0) + ' MHz' : '—';
  document.getElementById('clk-text').textContent = clkText;

  const statusEl = document.getElementById('clk-status');
  statusEl.textContent = status;
  statusEl.className = 'stat badge badge-' + throttleClass(status);

  document.getElementById('clk-pstate').textContent = t.pstate || '';

  const reasonsEl = document.getElementById('clk-reasons');
  const reasons = t.throttle_reasons || [];
  reasonsEl.textContent = reasons.length ? '⚠ ' + reasons.join(', ') : '';
}

function updatePower(data) {
  const card = document.getElementById('card-power');
  const rails = data.power_rails;

  if (!rails) {
    const pw = data.power;
    if (!pw || !pw.available) { card.classList.add('hidden'); return; }
    card.classList.remove('hidden');
    document.getElementById('power-content').innerHTML =
      `<div class="power-row"><span class="power-label">Power Draw</span><span class="power-value">${pw.power_w.toFixed(1)} W</span></div>`;
    return;
  }

  card.classList.remove('hidden');
  let html = '';

  if (rails.gpu_w != null) {
    html += `<div class="power-row"><span class="power-label">GPU</span><span class="power-value">${rails.gpu_w.toFixed(1)} W</span></div>`;
  }
  if (rails.peak_gpu_w > 0) {
    html += `<div class="power-row"><span class="power-label">Peak GPU</span><span class="power-value">${rails.peak_gpu_w.toFixed(1)} W</span></div>`;
  }
  if (rails.dc_w != null) {
    html += `<div class="power-row"><span class="power-label">DC Input</span><span class="power-value">${rails.dc_w.toFixed(1)} W</span></div>`;
  }
  if (rails.syspl1_cap_w != null) {
    const exceeded = rails.cap_exceeded;
    const style = exceeded ? 'color: var(--red); font-weight: bold;' : '';
    html += `<div class="power-row"><span class="power-label">SysPL1 Cap</span><span class="power-value" style="${style}">${rails.syspl1_cap_w.toFixed(0)} W${exceeded ? ' ⚠ EXCEEDED' : ''}</span></div>`;
  }
  if (rails.prochot != null) {
    const prochotColor = rails.prochot ? 'var(--red)' : 'var(--green)';
    html += `<div class="power-row"><span class="power-label">PROCHOT</span><span class="power-value" style="color:${prochotColor}">${rails.prochot ? '● ACTIVE' : '○ Clear'}</span></div>`;
  }
  if (rails.pl_level != null && rails.pl_level > 0) {
    html += `<div class="power-row"><span class="power-label">PL Level</span><span class="power-value">${rails.pl_level}</span></div>`;
  }
  if (rails.tj_rise_c != null && rails.tj_rise_c > 0) {
    const tjColor = rails.tj_rise_c > 15 ? 'var(--red)' : rails.tj_rise_c > 8 ? 'var(--orange)' : 'var(--text)';
    html += `<div class="power-row"><span class="power-label">Tj Rise</span><span class="power-value" style="color:${tjColor}">+${rails.tj_rise_c.toFixed(1)}°C</span></div>`;
  }

  document.getElementById('power-content').innerHTML = html;
}

function updateNetwork(data) {
  const nets = data.network || [];
  const card = document.getElementById('card-network');
  if (!nets.length) { card.classList.add('hidden'); return; }
  card.classList.remove('hidden');

  let html = '';
  for (const n of nets) {
    const state = n.state === 'UP' ? 'up' : 'down';
    const speed = n.speed_mbps >= 1000 ? (n.speed_mbps / 1000) + 'G' : (n.speed_mbps || '?') + 'M';
    html += `<div class="net-iface"><div class="net-header"><span class="net-name">${n.iface}</span><span class="net-state ${state}">${n.state}</span><span class="net-speed">${speed}</span></div>`;
    if (n.state === 'UP') {
      html += `<div class="net-stats"><span>TX ${fmtRate(n.tx_rate)}</span><span>RX ${fmtRate(n.rx_rate)}</span>`;
      if (n.rx_errors > 0 || n.tx_errors > 0) {
        html += `<span style="color:var(--red)">ERR rx:${n.rx_errors} tx:${n.tx_errors}</span>`;
      }
      html += `</div>`;
    }
    html += `</div>`;
  }
  document.getElementById('net-content').innerHTML = html;
}

function updatePressure(data) {
  const psi = data.pressure && data.pressure.mem;
  const card = document.getElementById('card-pressure');
  if (!psi || !psi.available) { card.classList.add('hidden'); return; }
  card.classList.remove('hidden');

  const somePct = Math.min(psi.some_avg10 * 100 / 0.30, 100);
  const fullPct = Math.min(psi.full_avg10 * 100 / 0.30, 100);

  setBar('psi-some-bar', somePct, psiClass(psi.level));
  document.getElementById('psi-some-text').textContent = psi.some_avg10.toFixed(2);

  setBar('psi-full-bar', fullPct, psiClass(psi.level));
  document.getElementById('psi-full-text').textContent = psi.full_avg10.toFixed(2);

  const levelEl = document.getElementById('psi-level');
  levelEl.textContent = psi.level;
  levelEl.className = 'stat badge badge-' + psiClass(psi.level);
}

function updateDisk(data) {
  const disks = data.disk || [];
  const card = document.getElementById('card-disk');
  if (!disks.length) { card.classList.add('hidden'); return; }
  card.classList.remove('hidden');

  let html = '';
  for (const d of disks) {
    const pct = d.percent;
    const cls = barClass(pct);
    const used = fmt(d.used);
    const total = fmt(d.total);
    html += `<div class="disk-mount">
      <div class="disk-label">${d.mountpoint} <span style="color:var(--text-dim);font-size:11px">(${d.device} ${d.fstype})</span></div>
      <div class="bar-track"><div class="bar-fill ${cls}" style="width:${pct.toFixed(1)}%"></div></div>
      <div class="disk-detail">${used} / ${total}  (${pct.toFixed(1)}%)</div>
    </div>`;
  }
  document.getElementById('disk-content').innerHTML = html;
}

function updateHeader(data) {
  const info = data.info || {};
  document.getElementById('hostname').textContent = data.hostname || '';

  const gpu = data.gpu && data.gpu[0];
  document.getElementById('gpu-name').textContent = info.gpu_name || (gpu ? gpu.name : '');

  const driver = info.driver ? 'Driver ' + info.driver : '';
  const cuda = info.cuda ? 'CUDA ' + info.cuda : '';
  document.getElementById('driver-info').textContent = [driver, cuda].filter(Boolean).join(' | ');

  document.getElementById('uptime-info').textContent = info.uptime ? 'Up ' + info.uptime : '';
  document.getElementById('kernel-info').textContent = info.kernel ? 'Kernel ' + info.kernel : '';
}

function updateAll(data) {
  updateHeader(data);
  updateGPU(data);
  updateMemory(data);
  updateCPU(data);
  updateThrottle(data);
  updatePower(data);
  updateNetwork(data);
  updatePressure(data);
  updateDisk(data);

  const t = new Date().toLocaleTimeString();
  document.getElementById('last-refresh').textContent = 'Last update: ' + t;
}

async function fetchMetrics() {
  const dot = document.getElementById('conn-status');
  try {
    const resp = await fetch('/api/metrics');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    updateAll(data);
    dot.className = 'conn-dot connected';
    dot.title = 'Connected';
  } catch (e) {
    dot.className = 'conn-dot disconnected';
    dot.title = 'Connection lost';
    console.error('Fetch error:', e);
  }
}

fetchMetrics();
setInterval(fetchMetrics, REFRESH_MS);