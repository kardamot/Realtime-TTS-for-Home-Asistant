const espCommands = [
  "test_speaker", "test_mic", "wake_on", "wake_off", "servo_left", "servo_center",
  "servo_right", "amp_mute_on", "amp_mute_off", "reconnect", "reboot"
];
const serverCommands = ["restart_stt", "restart_tts", "reload_prompt", "clear_logs", "safe_mode_on", "safe_mode_off"];
let token = localStorage.getItem("alice_panel_token") || "";
let currentConfig = {};
let currentPrompt = {};
let logs = [];
let paused = false;
let configDirty = false;
let logSocket = null;
let logSocketSeq = 0;
let eventSocket = null;
let eventSocketSeq = 0;
let statusTimer = null;
let statusRefreshTimer = null;

const $ = (id) => document.getElementById(id);
const text = (id, value) => { const el = $(id); if (el) el.textContent = value ?? "-"; };

function notice(value) {
  const el = $("notice");
  if (!value) { el.classList.add("hidden"); return; }
  el.textContent = value;
  el.classList.remove("hidden");
}

function rememberToken(value) {
  token = value || "";
  if (token) {
    localStorage.setItem("alice_panel_token", token);
    document.cookie = `alice_panel_token=${encodeURIComponent(token)}; path=/; SameSite=Lax`;
  } else {
    localStorage.removeItem("alice_panel_token");
    document.cookie = "alice_panel_token=; Max-Age=0; path=/; SameSite=Lax";
  }
}

async function guard(label, fn) {
  try {
    return await fn();
  } catch (err) {
    notice(`${label}: ${err.message}`);
    return null;
  }
}

function fmtSeconds(value) {
  const total = Number(value || 0);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = Math.floor(total % 60);
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}

function tone(value) {
  const key = String(value || "").toLowerCase();
  if (key.includes("online") || key.includes("idle") || key.includes("ok")) return "good";
  if (key.includes("error") || key.includes("offline")) return "bad";
  if (key.includes("mock") || key.includes("warn")) return "warn";
  return "info";
}

function setPill(id, value, forcedTone) {
  const el = $(id);
  if (!el) return;
  el.textContent = value;
  el.className = `pill ${forcedTone || tone(value)}`;
}

async function api(path, options = {}, auth = token) {
  const headers = new Headers(options.headers || {});
  if (auth) headers.set("X-Alice-Token", auth);
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const resp = await fetch(path, { ...options, headers });
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}: ${await resp.text()}`);
  const contentType = resp.headers.get("content-type") || "";
  return contentType.includes("application/json") ? resp.json() : resp.text();
}

function wsPath(path) {
  const url = new URL(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`);
  if (token) url.searchParams.set("token", token);
  return url.toString();
}

function getDeep(obj, path) {
  return path.split(".").reduce((acc, key) => acc && acc[key], obj);
}

function setDeep(obj, path, value) {
  const keys = path.split(".");
  let cursor = obj;
  keys.slice(0, -1).forEach((key) => {
    if (!cursor[key] || typeof cursor[key] !== "object") cursor[key] = {};
    cursor = cursor[key];
  });
  cursor[keys[keys.length - 1]] = value;
}

function stripMasked(value) {
  if (Array.isArray(value)) return value.map(stripMasked);
  if (value && typeof value === "object") {
    const out = {};
    Object.entries(value).forEach(([key, item]) => {
      if (item !== "********") out[key] = stripMasked(item);
    });
    return out;
  }
  return value;
}

async function boot() {
  renderButtons();
  $("refresh-btn").onclick = () => guard("Refresh failed", loadStatus);
  $("unlock-btn").onclick = () => guard("Unlock failed", unlock);
  $("pipeline-send").onclick = () => guard("Pipeline failed", runPipeline);
  $("config-save").onclick = () => guard("Config save failed", saveConfig);
  $("config-export").onclick = () => guard("Config export failed", exportConfig);
  $("config-import").onclick = () => $("config-import-file").click();
  $("config-import-file").onchange = () => guard("Config import failed", importConfig);
  $("prompt-new").onclick = () => guard("Prompt create failed", createPrompt);
  $("prompt-copy").onclick = () => guard("Prompt copy failed", copyPrompt);
  $("prompt-delete").onclick = () => guard("Prompt delete failed", deletePrompt);
  $("prompt-save").onclick = () => guard("Prompt save failed", savePrompt);
  $("prompt-activate").onclick = () => guard("Prompt activate failed", activatePrompt);
  $("logs-download").onclick = () => guard("Log download failed", downloadLogs);
  $("logs-clear").onclick = () => guard("Clear logs failed", () => sendCommand("clear_logs"));
  $("logs-pause").onclick = () => {
    paused = !paused;
    $("logs-pause").textContent = paused ? "Resume" : "Pause";
    if (!paused) {
      loadLogSnapshot().catch(() => undefined);
      if (!logSocket || logSocket.readyState === WebSocket.CLOSED) connectLogs();
    }
  };
  $("log-search").oninput = renderLogs;
  $("log-level").onchange = renderLogs;
  $("log-category").onchange = renderLogs;

  try {
    const auth = await api("/api/auth/check", {}, "");
    if (auth.auth_required && !token) {
      $("login").classList.remove("hidden");
      return;
    }
    await loadAll();
    connectLogs();
    connectEvents();
    startStatusPolling();
  } catch (err) {
    notice(err.message);
  }
}

async function unlock() {
  const draft = $("token-input").value;
  try {
    await api("/api/status", {}, draft);
    rememberToken(draft);
    $("login").classList.add("hidden");
    await loadAll();
    connectLogs();
    connectEvents();
    startStatusPolling();
  } catch (err) {
    $("login-error").textContent = err.message;
  }
}

async function loadAll() {
  await loadStatus();
  await loadPrompts();
}

function startStatusPolling() {
  if (statusTimer) window.clearInterval(statusTimer);
  statusTimer = window.setInterval(() => loadStatus().catch(() => undefined), 5000);
}

function scheduleStatusRefresh(delay = 250) {
  if (statusRefreshTimer) window.clearTimeout(statusRefreshTimer);
  statusRefreshTimer = window.setTimeout(() => loadStatus().catch(() => undefined), delay);
}

async function loadStatus() {
  const data = await api("/api/status");
  const esp = data.esp || {};
  const pipe = data.pipeline || {};
  const health = (data.health || {}).system || {};
  const backend = data.health || {};
  if (!configDirty) currentConfig = data.config || {};

  $("summary").textContent = esp.online ? "Robot linked" : esp.mock_mode ? "ESP offline, mock mode active" : "Waiting for robot status";
  text("backend-version", `${backend.service || "alice_control_panel"} ${backend.version || ""} - FastAPI backend online`);
  setPill("state-pill", pipe.state || "IDLE");
  setPill("esp-pill", esp.online ? "ONLINE" : esp.mock_mode ? "MOCK" : "OFFLINE");
  setPill("stream-pill", pipe.stream_active ? "STREAM ON" : "STREAM OFF", pipe.stream_active ? "good" : "info");
  text("robot-status", esp.online ? "ONLINE" : esp.mock_mode ? "MOCK" : "OFFLINE");
  text("robot-ip", esp.ip || "no ESP base URL");
  text("wifi-status", esp.wifi?.connected ? "connected" : "unknown");
  text("wifi-rssi", esp.wifi?.rssi ? `${esp.wifi.rssi} dBm` : "RSSI n/a");
  text("cpu-status", health.cpu_percent == null ? "n/a" : `${health.cpu_percent}%`);
  text("ram-status", `RAM ${health.ram_used_mb || "n/a"} MB`);
  text("heap-status", esp.heap_free || "n/a");
  text("heap-min", esp.heap_min ? `min ${esp.heap_min}` : "offline");
  text("server-uptime", fmtSeconds(health.uptime_sec));
  text("esp-uptime", `ESP ${fmtSeconds(esp.uptime_sec)}`);
  text("conn-esp", esp.online ? "online" : "offline");
  text("conn-stt", data.stt?.provider || "faster_whisper");
  text("conn-llm", `${data.llm?.provider || "openai"} / ${data.llm?.model || "n/a"}`);
  text("conn-tts", `${data.tts?.provider || "openai"} / ${data.tts?.pcm_sample_rate || "n/a"}`);
  text("conn-reconnects", esp.reconnects || 0);
  text("last-error", esp.last_error || "");
  text("hw-mic", esp.hardware?.mic || "unknown");
  text("hw-speaker", esp.hardware?.speaker || "unknown");
  text("hw-servo", esp.hardware?.servo_position || "center");
  text("hw-amp", esp.hardware?.amp_muted == null ? "unknown" : esp.hardware.amp_muted ? "muted" : "active");
  text("hw-wake", esp.hardware?.wake_enabled == null ? "unknown" : esp.hardware.wake_enabled ? "on" : "off");
  text("hw-state", esp.state || "OFFLINE");
  text("stt-text", pipe.stt_result || pipe.last_user_text || "No utterance yet");
  text("llm-text", pipe.llm_response || "FastAPI backend ready. Send a text test or configure providers.");
  renderTimeline(pipe.timeline || []);
  if (!configDirty) fillConfig();
}

function fillConfig() {
  document.querySelectorAll("[data-path]").forEach((el) => {
    const value = getDeep(currentConfig, el.dataset.path);
    if (el.type === "checkbox") el.checked = Boolean(value);
    else el.value = value ?? "";
    el.oninput = () => {
      configDirty = true;
      const next = el.type === "checkbox" ? el.checked : el.type === "number" ? Number(el.value) : el.value;
      setDeep(currentConfig, el.dataset.path, next);
    };
  });
}

function renderTimeline(items) {
  const box = $("timeline");
  const list = items.slice(-6);
  box.innerHTML = list.length ? "" : "<div><b>IDLE</b><span>Waiting for audio/text</span></div>";
  list.forEach((item) => {
    const row = document.createElement("div");
    row.innerHTML = `<b>${item.category || "STEP"}</b><span>${item.message || ""}</span>`;
    box.appendChild(row);
  });
}

function renderButtons() {
  $("esp-commands").innerHTML = "";
  espCommands.forEach((cmd) => {
    const btn = document.createElement("button");
    btn.textContent = cmd.replaceAll("_", " ");
    btn.onclick = () => guard("Command failed", () => sendCommand(cmd));
    $("esp-commands").appendChild(btn);
  });
  $("server-commands").innerHTML = "";
  serverCommands.forEach((cmd) => {
    const btn = document.createElement("button");
    btn.textContent = cmd.replaceAll("_", " ");
    btn.onclick = () => guard("Command failed", () => sendCommand(cmd));
    $("server-commands").appendChild(btn);
  });
}

async function sendCommand(command) {
  const result = await api("/api/command", { method: "POST", body: JSON.stringify({ command, payload: {} }) });
  if (command === "clear_logs") logs = [];
  notice(result.message || `${command} sent`);
  renderLogs();
  await loadStatus();
}

async function saveConfig() {
  await api("/api/config", { method: "POST", body: JSON.stringify(stripMasked(currentConfig)) });
  const nextToken = getDeep(currentConfig, "panel.token") || getDeep(currentConfig, "panel.password");
  if (nextToken && nextToken !== "********") rememberToken(nextToken);
  configDirty = false;
  notice("Config saved");
  await loadStatus();
}

async function exportConfig() {
  const includeSecrets = $("config-export-secrets").checked ? "true" : "false";
  const data = await api(`/api/config/export?include_secrets=${includeSecrets}`);
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = includeSecrets === "true" ? "alice_config_with_secrets.json" : "alice_config.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 500);
}

async function importConfig() {
  const input = $("config-import-file");
  const file = input.files && input.files[0];
  if (!file) return;
  const doc = JSON.parse(await file.text());
  await api("/api/config/import", { method: "POST", body: JSON.stringify(doc) });
  input.value = "";
  configDirty = false;
  notice("Config imported");
  await loadStatus();
}

async function downloadLogs() {
  const body = await api("/api/logs/download");
  const blob = new Blob([body], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "alice_logs.txt";
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 500);
}

async function loadPrompts(preferredSlug) {
  const data = await api("/api/prompts");
  const select = $("prompt-select");
  const profiles = data.profiles || [];
  select.innerHTML = "";
  profiles.forEach((profile) => {
    const opt = document.createElement("option");
    opt.value = profile.slug;
    opt.textContent = profile.name;
    select.appendChild(opt);
  });
  const desired = [preferredSlug, data.active_profile, profiles[0]?.slug, "alice"].find((slug) =>
    slug && profiles.some((profile) => profile.slug === slug)
  );
  select.value = desired || "";
  select.onchange = () => loadPrompt(select.value);
  if (select.value) await loadPrompt(select.value);
}

async function loadPrompt(slug) {
  currentPrompt = await api(`/api/prompts/${slug}`);
  $("prompt-name").value = currentPrompt.name || "";
  $("prompt-description").value = currentPrompt.description || "";
  $("prompt-text").value = currentPrompt.prompt || "";
}

async function createPrompt() {
  const name = window.prompt("New prompt profile name", "Alice Copy");
  if (!name) return;
  const result = await api("/api/prompts", {
    method: "POST",
    body: JSON.stringify({
      name,
      description: "Custom prompt profile",
      prompt: $("prompt-text").value || currentPrompt.prompt || "",
    }),
  });
  notice("Prompt created");
  await loadPrompts(result.prompt.slug);
}

async function copyPrompt() {
  if (!currentPrompt.slug) return;
  const name = window.prompt("Copied prompt profile name", `${currentPrompt.name || currentPrompt.slug} Copy`);
  if (!name) return;
  const result = await api(`/api/prompts/${currentPrompt.slug}/copy`, {
    method: "POST",
    body: JSON.stringify({ name }),
  });
  notice("Prompt copied");
  await loadPrompts(result.prompt.slug);
}

async function deletePrompt() {
  if (!currentPrompt.slug) return;
  if (!window.confirm(`Delete prompt profile "${currentPrompt.name || currentPrompt.slug}"?`)) return;
  await api(`/api/prompts/${currentPrompt.slug}`, { method: "DELETE" });
  notice("Prompt deleted");
  await loadPrompts();
}

async function savePrompt() {
  currentPrompt.name = $("prompt-name").value;
  currentPrompt.description = $("prompt-description").value;
  currentPrompt.prompt = $("prompt-text").value;
  await api(`/api/prompts/${currentPrompt.slug}`, { method: "POST", body: JSON.stringify(currentPrompt) });
  notice("Prompt saved");
  await loadPrompts(currentPrompt.slug);
}

async function activatePrompt() {
  await api(`/api/prompts/${currentPrompt.slug}/activate`, { method: "POST" });
  notice("Prompt activated");
  await loadPrompts();
}

async function runPipeline() {
  const input = $("pipeline-input");
  if (!input.value.trim()) return;
  await api("/api/pipeline/text", { method: "POST", body: JSON.stringify({ text: input.value }) });
  input.value = "";
  await loadStatus();
}

function connectLogs() {
  const seq = ++logSocketSeq;
  if (logSocket) {
    logSocket.onclose = null;
    logSocket.close();
  }
  loadLogSnapshot().catch(() => undefined);
  const socket = new WebSocket(wsPath("/api/ws/logs"));
  logSocket = socket;
  socket.onopen = () => notice("");
  socket.onmessage = (event) => {
    if (paused) return;
    const doc = JSON.parse(event.data);
    const incoming = doc.entries || [];
    if (!incoming.length) return;
    mergeLogs(incoming);
    renderLogCategories();
    renderLogs();
  };
  socket.onerror = () => {
    notice("Log WebSocket baglanamadi; HTTP log snapshot kullaniliyor.");
    loadLogSnapshot().catch(() => undefined);
  };
  socket.onclose = () => {
    window.setTimeout(() => {
      if (logSocketSeq === seq && !paused) connectLogs();
    }, 3000);
  };
}

async function loadLogSnapshot() {
  const data = await api("/api/logs?limit=250");
  mergeLogs(data.entries || []);
  renderLogCategories();
  renderLogs();
}

function mergeLogs(entries) {
  const map = new Map(logs.map((entry) => [entry.id, entry]));
  entries.forEach((entry) => {
    if (entry && entry.id) map.set(entry.id, entry);
  });
  logs = Array.from(map.values()).sort((a, b) => (a.ts || 0) - (b.ts || 0)).slice(-1000);
}

function connectEvents() {
  const seq = ++eventSocketSeq;
  if (eventSocket) {
    eventSocket.onclose = null;
    eventSocket.close();
  }
  const socket = new WebSocket(wsPath("/api/ws/events"));
  eventSocket = socket;
  socket.onmessage = (event) => {
    const doc = JSON.parse(event.data);
    if (doc.type === "snapshot" || doc.type === "esp_status" || doc.type === "pipeline_status" || doc.type === "config_updated") {
      scheduleStatusRefresh();
    }
  };
  socket.onclose = () => {
    window.setTimeout(() => {
      if (eventSocketSeq === seq) connectEvents();
    }, 4000);
  };
}

function renderLogCategories() {
  const select = $("log-category");
  const old = select.value;
  const cats = ["ALL", ...Array.from(new Set(logs.map((entry) => entry.category))).sort()];
  select.innerHTML = cats.map((cat) => `<option>${cat}</option>`).join("");
  select.value = cats.includes(old) ? old : "ALL";
}

function renderLogs() {
  const q = $("log-search").value.toLowerCase().trim();
  const level = $("log-level").value;
  const cat = $("log-category").value;
  const rows = logs.filter((entry) => {
    if (level !== "ALL" && entry.level !== level) return false;
    if (cat !== "ALL" && entry.category !== cat) return false;
    if (!q) return true;
    return `${entry.level} ${entry.category} ${entry.message} ${JSON.stringify(entry.details || {})}`.toLowerCase().includes(q);
  }).slice(-220);
  $("log-list").innerHTML = "";
  rows.forEach((entry) => {
    const row = document.createElement("div");
    row.className = `log-row ${String(entry.level || "").toLowerCase()}`;
    row.innerHTML = `<time>${new Date(entry.ts * 1000).toLocaleTimeString()}</time><b>${entry.level}</b><span>${entry.category}</span><p></p>`;
    row.querySelector("p").textContent = entry.message || "";
    $("log-list").appendChild(row);
  });
}

window.addEventListener("load", boot);
