import {
  Activity,
  Bot,
  Brain,
  CheckCircle2,
  Cpu,
  Download,
  FileText,
  Gauge,
  KeyRound,
  Loader2,
  Mic,
  Pause,
  Play,
  Power,
  Radio,
  RefreshCw,
  RotateCcw,
  Save,
  Search,
  Send,
  Server,
  Settings,
  Shield,
  SlidersHorizontal,
  Terminal,
  Trash2,
  Upload,
  Volume2,
  Wifi,
  Wrench
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { apiFetch, downloadJson, getStoredToken, setDeep, setStoredToken, wsUrl } from "./api";
import type { AnyRecord, LogEntry, PromptProfile, StatusResponse } from "./types";

const espCommands = [
  ["test_speaker", "Speaker", Volume2],
  ["test_mic", "Mic", Mic],
  ["wake_on", "Wake On", Radio],
  ["wake_off", "Wake Off", Radio],
  ["servo_left", "Servo L", RotateCcw],
  ["servo_center", "Servo C", Gauge],
  ["servo_right", "Servo R", RefreshCw],
  ["amp_mute_on", "Mute", Volume2],
  ["amp_mute_off", "Unmute", Volume2],
  ["reconnect", "Reconnect", RefreshCw],
  ["reboot", "Reboot", Power]
] as const;

const serverCommands = [
  ["restart_stt", "Restart STT", Mic],
  ["restart_tts", "Restart TTS", Volume2],
  ["reload_prompt", "Reload Prompt", FileText],
  ["clear_logs", "Clear Logs", Trash2],
  ["safe_mode_on", "Safe On", Shield],
  ["safe_mode_off", "Safe Off", Shield]
] as const;

function stripMaskedSecrets(value: any): any {
  if (Array.isArray(value)) return value.map(stripMaskedSecrets);
  if (value && typeof value === "object") {
    const next: AnyRecord = {};
    Object.entries(value).forEach(([key, item]) => {
      if (item === "********") return;
      next[key] = stripMaskedSecrets(item);
    });
    return next;
  }
  return value;
}

function fmtSeconds(value?: number) {
  const total = Number(value || 0);
  const hours = Math.floor(total / 3600);
  const mins = Math.floor((total % 3600) / 60);
  const secs = Math.floor(total % 60);
  if (hours) return `${hours}h ${mins}m`;
  if (mins) return `${mins}m ${secs}s`;
  return `${secs}s`;
}

function stateTone(value: string) {
  const key = value.toLowerCase();
  if (key.includes("online") || key.includes("idle") || key.includes("ok")) return "good";
  if (key.includes("error") || key.includes("offline")) return "bad";
  if (key.includes("mock") || key.includes("warn")) return "warn";
  return "info";
}

function StatusPill({ value, tone }: { value: string; tone?: string }) {
  return <span className={`pill ${tone || stateTone(value)}`}>{value}</span>;
}

function Panel({
  title,
  icon: Icon,
  actions,
  children,
  className = ""
}: {
  title: string;
  icon: any;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <header className="panel-head">
        <div className="panel-title">
          <Icon size={18} />
          <h2>{title}</h2>
        </div>
        {actions}
      </header>
      {children}
    </section>
  );
}

function Metric({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {sub ? <small>{sub}</small> : null}
    </div>
  );
}

function Field({
  label,
  children,
  wide = false
}: {
  label: string;
  children: ReactNode;
  wide?: boolean;
}) {
  return (
    <label className={`field ${wide ? "wide" : ""}`}>
      <span>{label}</span>
      {children}
    </label>
  );
}

function App() {
  const [token, setToken] = useState(getStoredToken());
  const [tokenDraft, setTokenDraft] = useState("");
  const [authRequired, setAuthRequired] = useState(false);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [configDraft, setConfigDraft] = useState<AnyRecord>({});
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [logsPaused, setLogsPaused] = useState(false);
  const [logSearch, setLogSearch] = useState("");
  const [logLevel, setLogLevel] = useState("ALL");
  const [logCategory, setLogCategory] = useState("ALL");
  const [prompts, setPrompts] = useState<PromptProfile[]>([]);
  const [activePrompt, setActivePrompt] = useState("alice");
  const [promptDoc, setPromptDoc] = useState<AnyRecord>({});
  const [pipelineText, setPipelineText] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const loadStatus = useCallback(async () => {
    const data = await apiFetch<StatusResponse>("/api/status", {}, token);
    setStatus(data);
    setConfigDraft(data.config || {});
  }, [token]);

  const loadPrompts = useCallback(async () => {
    const data = await apiFetch<{ active_profile: string; profiles: PromptProfile[] }>("/api/prompts", {}, token);
    setPrompts(data.profiles || []);
    setActivePrompt(data.active_profile || "alice");
    const selected = data.active_profile || data.profiles?.[0]?.slug || "alice";
    const doc = await apiFetch<AnyRecord>(`/api/prompts/${selected}`, {}, token);
    setPromptDoc(doc);
  }, [token]);

  const boot = useCallback(async () => {
    const auth = await apiFetch<{ auth_required: boolean }>("/api/auth/check", {});
    setAuthRequired(auth.auth_required);
    if (!auth.auth_required || token) {
      await loadStatus();
      await loadPrompts();
    }
  }, [loadPrompts, loadStatus, token]);

  useEffect(() => {
    boot().catch((err) => setNotice(err.message));
  }, [boot]);

  useEffect(() => {
    if (authRequired && !token) return;
    const id = window.setInterval(() => {
      loadStatus().catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(id);
  }, [authRequired, loadStatus, token]);

  useEffect(() => {
    if (authRequired && !token) return;
    let closed = false;
    const socket = new WebSocket(wsUrl("/api/ws/logs", token));
    socket.onmessage = (event) => {
      if (closed || logsPaused) return;
      const doc = JSON.parse(event.data);
      const entries: LogEntry[] = doc.entries || [];
      if (entries.length) {
        setLogs((prev) => [...prev, ...entries].slice(-1000));
      }
    };
    return () => {
      closed = true;
      socket.close();
    };
  }, [authRequired, logsPaused, token]);

  const filteredLogs = useMemo(() => {
    const query = logSearch.toLowerCase().trim();
    return logs.filter((entry) => {
      if (logLevel !== "ALL" && entry.level !== logLevel) return false;
      if (logCategory !== "ALL" && entry.category !== logCategory) return false;
      if (!query) return true;
      return `${entry.category} ${entry.level} ${entry.message} ${JSON.stringify(entry.details || {})}`.toLowerCase().includes(query);
    });
  }, [logCategory, logLevel, logSearch, logs]);

  const categories = useMemo(() => ["ALL", ...Array.from(new Set(logs.map((entry) => entry.category))).sort()], [logs]);

  async function unlock() {
    setBusy(true);
    try {
      setStoredToken(tokenDraft);
      setToken(tokenDraft);
      await apiFetch("/api/status", {}, tokenDraft);
      setNotice("");
    } catch (err: any) {
      setStoredToken("");
      setToken("");
      setNotice(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function sendCommand(command: string) {
    setBusy(true);
    try {
      const result = await apiFetch<AnyRecord>(
        "/api/command",
        { method: "POST", body: JSON.stringify({ command, payload: {} }) },
        token
      );
      if (command === "clear_logs") setLogs([]);
      setNotice(result.message || `${command} sent`);
      await loadStatus();
    } catch (err: any) {
      setNotice(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function saveConfig() {
    setBusy(true);
    try {
      await apiFetch("/api/config", { method: "POST", body: JSON.stringify(stripMaskedSecrets(configDraft)) }, token);
      setNotice("Config saved");
      await loadStatus();
    } catch (err: any) {
      setNotice(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function exportConfig(includeSecrets: boolean) {
    const data = await apiFetch<AnyRecord>(`/api/config/export?include_secrets=${includeSecrets ? "true" : "false"}`, {}, token);
    downloadJson(includeSecrets ? "alice_config_with_secrets.json" : "alice_config.json", data);
  }

  async function importConfig(file?: File) {
    if (!file) return;
    const text = await file.text();
    const parsed = JSON.parse(text);
    await apiFetch("/api/config/import", { method: "POST", body: JSON.stringify(parsed) }, token);
    await loadStatus();
    setNotice("Config imported");
  }

  async function selectPrompt(slug: string) {
    setActivePrompt(slug);
    setPromptDoc(await apiFetch<AnyRecord>(`/api/prompts/${slug}`, {}, token));
  }

  async function savePrompt() {
    await apiFetch(`/api/prompts/${promptDoc.slug || activePrompt}`, { method: "POST", body: JSON.stringify(promptDoc) }, token);
    await loadPrompts();
    setNotice("Prompt saved");
  }

  async function activatePrompt() {
    await apiFetch(`/api/prompts/${promptDoc.slug || activePrompt}/activate`, { method: "POST" }, token);
    await loadPrompts();
    setNotice("Prompt activated");
  }

  async function runPipelineText() {
    if (!pipelineText.trim()) return;
    setBusy(true);
    try {
      await apiFetch("/api/pipeline/text", { method: "POST", body: JSON.stringify({ text: pipelineText }) }, token);
      setPipelineText("");
      await loadStatus();
    } catch (err: any) {
      setNotice(err.message);
    } finally {
      setBusy(false);
    }
  }

  if (authRequired && !token) {
    return (
      <main className="login-shell">
        <section className="login-panel">
          <div className="brand-mark"><KeyRound size={22} /></div>
          <h1>Alice Control Panel</h1>
          <p>Panel token/password gerekli.</p>
          <input
            type="password"
            value={tokenDraft}
            onChange={(event) => setTokenDraft(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && unlock()}
            placeholder="Token veya password"
          />
          <button className="primary" onClick={unlock} disabled={busy || !tokenDraft}>
            {busy ? <Loader2 className="spin" size={16} /> : <Shield size={16} />} Unlock
          </button>
          {notice ? <small className="error-text">{notice}</small> : null}
        </section>
      </main>
    );
  }

  const esp = status?.esp || {};
  const pipe = status?.pipeline || {};
  const health = status?.health?.system || {};
  const cfg = configDraft || {};

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="app-brand">
          <div className="brand-mark"><Bot size={22} /></div>
          <div>
            <strong>Alice</strong>
            <span>Control Panel</span>
          </div>
        </div>
        <nav>
          <a href="#status"><Activity size={17} />Status</a>
          <a href="#pipeline"><Brain size={17} />Pipeline</a>
          <a href="#hardware"><Wrench size={17} />Hardware</a>
          <a href="#config"><Settings size={17} />Config</a>
          <a href="#prompts"><FileText size={17} />Prompts</a>
          <a href="#logs"><Terminal size={17} />Logs</a>
        </nav>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>Alice Control Panel</h1>
            <p>{esp.online ? "Robot linked" : esp.mock_mode ? "ESP offline, mock mode active" : "Waiting for robot status"}</p>
          </div>
          <div className="top-actions">
            <StatusPill value={pipe.state || "IDLE"} />
            <button className="icon-button" title="Refresh" onClick={loadStatus}><RefreshCw size={17} /></button>
            {token ? (
              <button className="icon-button" title="Lock panel" onClick={() => { setStoredToken(""); setToken(""); }}>
                <Shield size={17} />
              </button>
            ) : null}
          </div>
        </header>

        {notice ? <div className="notice">{notice}</div> : null}

        <section id="status" className="status-strip">
          <Metric label="Robot" value={<StatusPill value={esp.online ? "ONLINE" : esp.mock_mode ? "MOCK" : "OFFLINE"} />} sub={esp.ip || "no ESP base URL"} />
          <Metric label="Wi-Fi" value={esp.wifi?.connected ? "connected" : "unknown"} sub={esp.wifi?.rssi ? `${esp.wifi.rssi} dBm` : "RSSI n/a"} />
          <Metric label="Server CPU" value={health.cpu_percent == null ? "n/a" : `${health.cpu_percent}%`} sub={`RAM ${health.ram_used_mb || "n/a"} MB`} />
          <Metric label="ESP Heap" value={esp.heap_free || "n/a"} sub={esp.heap_min ? `min ${esp.heap_min}` : "offline"} />
          <Metric label="Uptime" value={fmtSeconds(health.uptime_sec)} sub={`ESP ${fmtSeconds(esp.uptime_sec)}`} />
        </section>

        <section className="grid">
          <Panel title="Connections" icon={Server} className="span-4">
            <div className="connection-list">
              <div><Wifi size={16} /><span>ESP</span><StatusPill value={esp.online ? "online" : "offline"} /></div>
              <div><Mic size={16} /><span>STT</span><StatusPill value={status?.stt?.provider || "faster_whisper"} tone="info" /></div>
              <div><Brain size={16} /><span>LLM</span><b>{status?.llm?.provider || "openai"} / {status?.llm?.model || "n/a"}</b></div>
              <div><Volume2 size={16} /><span>TTS</span><b>{status?.tts?.provider || "openai"} / {cfg.tts?.[status?.tts?.provider || "openai"]?.model || cfg.tts?.cartesia?.model_id || "n/a"}</b></div>
              <div><RefreshCw size={16} /><span>Reconnects</span><b>{esp.reconnects || 0}</b></div>
            </div>
            {esp.last_error ? <div className="inline-error">{esp.last_error}</div> : null}
          </Panel>

          <Panel title="Voice Pipeline" icon={Brain} className="span-8">
            <div className="pipeline-layout">
              <div className="pipeline-copy">
                <label>Text test</label>
                <div className="send-row">
                  <input value={pipelineText} onChange={(event) => setPipelineText(event.target.value)} placeholder="Pipeline'a test metni gönder" />
                  <button className="primary icon-text" onClick={runPipelineText} disabled={busy}>
                    <Send size={16} />Send
                  </button>
                </div>
                <div className="transcript-box">
                  <span>User/STT</span>
                  <p>{pipe.stt_result || pipe.last_user_text || "No utterance yet"}</p>
                  <span>LLM</span>
                  <p>{pipe.llm_response || "No response yet"}</p>
                </div>
              </div>
              <div className="timeline">
                {(pipe.timeline || []).slice(-5).map((item: AnyRecord, idx: number) => (
                  <div key={`${item.ts}-${idx}`}>
                    <i />
                    <strong>{item.category}</strong>
                    <span>{item.message}</span>
                  </div>
                ))}
                {!(pipe.timeline || []).length ? <div><i /><strong>IDLE</strong><span>Waiting for audio/text</span></div> : null}
              </div>
            </div>
          </Panel>

          <Panel title="Hardware" icon={Wrench} className="span-4" actions={<StatusPill value={esp.hardware?.errors?.length ? "errors" : "nominal"} />}>
            <div className="hardware-grid">
              <Metric label="Mic" value={esp.hardware?.mic || "unknown"} />
              <Metric label="Speaker" value={esp.hardware?.speaker || "unknown"} />
              <Metric label="Servo" value={esp.hardware?.servo_position || "center"} />
              <Metric label="Amp" value={esp.hardware?.amp_muted == null ? "unknown" : esp.hardware?.amp_muted ? "muted" : "active"} />
              <Metric label="Wake" value={esp.hardware?.wake_enabled == null ? "unknown" : esp.hardware?.wake_enabled ? "on" : "off"} />
              <Metric label="State" value={esp.state || "OFFLINE"} />
            </div>
          </Panel>

          <Panel title="Command Panel" icon={SlidersHorizontal} className="span-8">
            <div className="command-group">
              {espCommands.map(([command, label, Icon]) => (
                <button key={command} className="tool-button" title={command} onClick={() => sendCommand(command)}>
                  <Icon size={17} />{label}
                </button>
              ))}
            </div>
            <div className="command-group secondary">
              {serverCommands.map(([command, label, Icon]) => (
                <button key={command} className="tool-button" title={command} onClick={() => sendCommand(command)}>
                  <Icon size={17} />{label}
                </button>
              ))}
            </div>
          </Panel>

          <Panel
            title="Config"
            icon={Settings}
            className="span-12"
            actions={
              <div className="panel-actions">
                <button className="icon-button" title="Import" onClick={() => fileInput.current?.click()}><Upload size={16} /></button>
                <button className="icon-button" title="Export masked" onClick={() => exportConfig(false)}><Download size={16} /></button>
                <button className="primary icon-text" onClick={saveConfig} disabled={busy}><Save size={16} />Save</button>
              </div>
            }
          >
            <input ref={fileInput} type="file" accept="application/json" hidden onChange={(event) => importConfig(event.target.files?.[0])} />
            <div className="config-grid" id="config">
              <Field label="Panel port"><input type="number" value={cfg.panel?.port || 8099} onChange={(e) => setConfigDraft(setDeep(cfg, "panel.port", Number(e.target.value)))} /></Field>
              <Field label="Panel token"><input type="password" value={cfg.panel?.token || ""} onChange={(e) => setConfigDraft(setDeep(cfg, "panel.token", e.target.value))} /></Field>
              <Field label="Panel password"><input type="password" value={cfg.panel?.password || ""} onChange={(e) => setConfigDraft(setDeep(cfg, "panel.password", e.target.value))} /></Field>
              <Field label="ESP base URL"><input value={cfg.esp?.base_url || ""} onChange={(e) => setConfigDraft(setDeep(cfg, "esp.base_url", e.target.value))} placeholder="http://192.168.1.50" /></Field>
              <Field label="Poll sec"><input type="number" value={cfg.esp?.poll_interval_sec || 3} onChange={(e) => setConfigDraft(setDeep(cfg, "esp.poll_interval_sec", Number(e.target.value)))} /></Field>
              <Field label="STT model"><input value={cfg.stt?.model || "small"} onChange={(e) => setConfigDraft(setDeep(cfg, "stt.model", e.target.value))} /></Field>
              <Field label="Compute"><select value={cfg.stt?.compute_type || "int8"} onChange={(e) => setConfigDraft(setDeep(cfg, "stt.compute_type", e.target.value))}><option>int8</option><option>float16</option><option>float32</option></select></Field>
              <Field label="LLM provider"><select value={cfg.llm?.provider || "openai"} onChange={(e) => setConfigDraft(setDeep(cfg, "llm.provider", e.target.value))}><option>openai</option><option>openrouter</option><option>mock</option><option>none</option></select></Field>
              <Field label="LLM model"><input value={cfg.llm?.model || ""} onChange={(e) => setConfigDraft(setDeep(cfg, "llm.model", e.target.value))} /></Field>
              <Field label="LLM key"><input type="password" value={cfg.llm?.api_key || ""} onChange={(e) => setConfigDraft(setDeep(cfg, "llm.api_key", e.target.value))} /></Field>
              <Field label="LLM base URL"><input value={cfg.llm?.base_url || ""} onChange={(e) => setConfigDraft(setDeep(cfg, "llm.base_url", e.target.value))} /></Field>
              <Field label="TTS provider"><select value={cfg.tts?.provider || "openai"} onChange={(e) => setConfigDraft(setDeep(cfg, "tts.provider", e.target.value))}><option>openai</option><option>cartesia</option><option>elevenlabs</option><option>google_ai</option><option>google_cloud</option></select></Field>
              <Field label="PCM rate"><input type="number" value={cfg.tts?.pcm_sample_rate || 44100} onChange={(e) => setConfigDraft(setDeep(cfg, "tts.pcm_sample_rate", Number(e.target.value)))} /></Field>
              <Field label="OpenAI TTS key"><input type="password" value={cfg.tts?.openai?.api_key || ""} onChange={(e) => setConfigDraft(setDeep(cfg, "tts.openai.api_key", e.target.value))} /></Field>
              <Field label="OpenAI voice"><input value={cfg.tts?.openai?.voice || ""} onChange={(e) => setConfigDraft(setDeep(cfg, "tts.openai.voice", e.target.value))} /></Field>
              <Field label="Cartesia key"><input type="password" value={cfg.tts?.cartesia?.api_key || ""} onChange={(e) => setConfigDraft(setDeep(cfg, "tts.cartesia.api_key", e.target.value))} /></Field>
              <Field label="Cartesia voice"><input value={cfg.tts?.cartesia?.voice_id || ""} onChange={(e) => setConfigDraft(setDeep(cfg, "tts.cartesia.voice_id", e.target.value))} /></Field>
              <Field label="Safe mode"><input type="checkbox" checked={Boolean(cfg.safe_mode)} onChange={(e) => setConfigDraft(setDeep(cfg, "safe_mode", e.target.checked))} /></Field>
              <Field label="Debug logs"><input type="checkbox" checked={Boolean(cfg.debug_logs)} onChange={(e) => setConfigDraft(setDeep(cfg, "debug_logs", e.target.checked))} /></Field>
              <Field label="LLM system prompt" wide><textarea value={cfg.llm?.system_prompt || ""} onChange={(e) => setConfigDraft(setDeep(cfg, "llm.system_prompt", e.target.value))} /></Field>
            </div>
            <button className="subtle" onClick={() => exportConfig(true)}><KeyRound size={15} />Export with secrets</button>
          </Panel>

          <Panel
            title="Prompt Editor"
            icon={FileText}
            className="span-6"
            actions={
              <div className="panel-actions">
                <button className="icon-button" title="Activate" onClick={activatePrompt}><CheckCircle2 size={16} /></button>
                <button className="primary icon-text" onClick={savePrompt}><Save size={16} />Save</button>
              </div>
            }
          >
            <div id="prompts" className="prompt-shell">
              <select value={activePrompt} onChange={(event) => selectPrompt(event.target.value)}>
                {prompts.map((profile) => <option key={profile.slug} value={profile.slug}>{profile.name}</option>)}
              </select>
              <input value={promptDoc.name || ""} onChange={(event) => setPromptDoc({ ...promptDoc, name: event.target.value })} />
              <textarea className="prompt-editor" value={promptDoc.prompt || ""} onChange={(event) => setPromptDoc({ ...promptDoc, prompt: event.target.value })} />
            </div>
          </Panel>

          <Panel
            title="Logs"
            icon={Terminal}
            className="span-6"
            actions={
              <div className="panel-actions">
                <button className="icon-button" title={logsPaused ? "Resume" : "Pause"} onClick={() => setLogsPaused(!logsPaused)}>{logsPaused ? <Play size={16} /> : <Pause size={16} />}</button>
                <button className="icon-button" title="Clear" onClick={() => sendCommand("clear_logs")}><Trash2 size={16} /></button>
              </div>
            }
          >
            <div id="logs" className="log-tools">
              <div className="search-box"><Search size={15} /><input value={logSearch} onChange={(e) => setLogSearch(e.target.value)} placeholder="Search logs" /></div>
              <select value={logLevel} onChange={(e) => setLogLevel(e.target.value)}><option>ALL</option><option>DEBUG</option><option>INFO</option><option>WARN</option><option>ERROR</option></select>
              <select value={logCategory} onChange={(e) => setLogCategory(e.target.value)}>{categories.map((cat) => <option key={cat}>{cat}</option>)}</select>
            </div>
            <div className="log-list">
              {filteredLogs.slice(-220).map((entry) => (
                <div key={entry.id} className={`log-row ${entry.level.toLowerCase()}`}>
                  <time>{new Date(entry.ts * 1000).toLocaleTimeString()}</time>
                  <b>{entry.level}</b>
                  <span>{entry.category}</span>
                  <p>{entry.message}</p>
                </div>
              ))}
            </div>
          </Panel>
        </section>
      </section>
    </main>
  );
}

export default App;
