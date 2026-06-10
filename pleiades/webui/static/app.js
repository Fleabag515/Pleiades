/* ═══════════════════════════════════════════════════════════
   Pleiades — frontend v2
   Single-page app over the FastAPI backend. No build step, no deps.
   Views: Overview · Chat · Agent · Characters · Models · Hardware · Settings
   ═══════════════════════════════════════════════════════════ */
"use strict";

/* ───────────── tiny API client ───────────── */
const api = {
  async req(method, path, body) {
    const opt = { method, headers: {} };
    if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
    const r = await fetch(path, opt);
    let data = null;
    try { data = await r.json(); } catch (_) {}
    if (!r.ok) throw new Error((data && (data.detail || data.error)) || `${r.status} ${r.statusText}`);
    return data;
  },
  get:  (p)    => api.req("GET", p),
  post: (p, b) => api.req("POST", p, b ?? {}),
  put:  (p, b) => api.req("PUT", p, b ?? {}),
  del:  (p)    => api.req("DELETE", p),
};

/* ───────────── helpers ───────────── */
const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false)
    n.append(kid.nodeType ? kid : document.createTextNode(kid));
  return n;
};
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;" }[c]));
const initials = (name) => (name || "?").slice(0, 2).toUpperCase();
const GiB = (b) => (b / 1073741824).toFixed(1);
const ago = (ts) => {
  if (!ts) return "—";
  const d = Math.floor(Date.now() / 1000 - ts);
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d/60)}m ago`;
  if (d < 86400) return `${Math.floor(d/3600)}h ago`;
  return `${Math.floor(d/86400)}d ago`;
};

function toast(title, msg = "", kind = "") {
  const t = el("div", { class: `toast ${kind}` },
    el("div", { class: "t-title" }, title),
    msg ? el("div", { class: "t-msg" }, msg) : null);
  $("#toast-host").append(t);
  setTimeout(() => { t.style.opacity = "0"; t.style.transition = "opacity .3s"; setTimeout(() => t.remove(), 300); }, 3400);
}
const ok  = (t, m) => toast(t, m, "good");
const err = (e)    => toast("Something went wrong", e.message || String(e), "bad");

/* ───────────── modal ───────────── */
const modal = {
  open({ title, body, footer }) {
    $("#modal-title").textContent = title;
    const mb = $("#modal-body"); mb.innerHTML = ""; mb.append(body);
    const mf = $("#modal-foot"); mf.innerHTML = ""; if (footer) mf.append(...footer);
    $("#modal-backdrop").hidden = false;
  },
  close() { $("#modal-backdrop").hidden = true; },
};
$("#modal-close").addEventListener("click", () => modal.close());
$("#modal-backdrop").addEventListener("click", e => { if (e.target.id === "modal-backdrop") modal.close(); });
document.addEventListener("keydown", e => { if (e.key === "Escape") modal.close(); });

/* form field builders */
function field(label, input, hint) {
  return el("div", { class: "field" },
    el("label", {}, label),
    input,
    hint ? el("div", { class: "hint" }, hint) : null);
}
function textInput(value = "", attrs = {}) {
  return el("input", { class: "input", value, ...attrs });
}
function switchToggle(checked, labelText) {
  const inp = el("input", { type: "checkbox" });
  inp.checked = !!checked;
  return el("label", { class: "switch" }, inp, el("span", { class: "track" }), labelText ? el("span", {}, labelText) : null);
}

/* ───────────── global state + router ───────────── */
const state = {
  view: "dashboard", detail: null, cache: {},
  chats: {},            // character -> [{role, text}]
  chatWith: null,       // active chat character
  workJob: null,        // active job id being watched
};

const VIEWS = {
  dashboard:  { title: "Overview",   sub: "Services, characters, and models at a glance.", render: renderDashboard },
  chat:       { title: "Chat",       sub: "Talk to a character — memory and tools wired in.", render: renderChat, full: true },
  work:       { title: "Agent",      sub: "Give a character real machine work: files, git, shell, web, and 60+ tools.", render: renderWork },
  characters: { title: "Characters", sub: "Each character binds memory, email, vault, browser, and a model to one identity.", render: renderCharacters },
  models:     { title: "Models",     sub: "Register GGUF models and run a local OpenAI-compatible server for each.", render: renderModels },
  hardware:   { title: "Hardware",   sub: "Detected GPU / RAM and the planned GPU offload for each model.", render: renderHardware },
  settings:   { title: "Settings",   sub: "Engine, services, and agent-harness configuration.", render: renderSettings },
};

function go(view, detail = null) {
  state.view = view; state.detail = detail;
  $$(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.view === view));
  const v = VIEWS[view];
  $("#view-title").textContent = v.title;
  $("#view-sub").textContent = v.sub;
  $("#content").classList.toggle("full", !!v.full);
  v.render();
}
$$(".nav-item").forEach(b => b.addEventListener("click", () => go(b.dataset.view)));
$("#refresh-btn").addEventListener("click", () => { loadStatusMini(); go(state.view, state.detail); });

function loading() { $("#content").innerHTML = `<div class="skeleton" style="padding:36px"><span class="spinner"></span> Loading…</div>`; }

/* ═══════════════ OVERVIEW ═══════════════ */
async function renderDashboard() {
  loading();
  let s;
  try { s = await api.get("/api/status"); } catch (e) { return err(e); }
  const c = $("#content"); c.innerHTML = "";

  /* first-run onboarding */
  const needsModel = !s.counts.models && s.services.inference.state === "no_model";
  const needsChar  = !s.counts.profiles;
  if (needsModel || needsChar || !s.services.anamnesis.up) {
    const step = (n, done, title, sub, action) => el("div", { class: `step ${done ? "done" : ""}` },
      el("span", { class: "step-num" }, done ? "✓" : String(n)),
      el("div", { class: "step-body" }, el("div", { class: "step-title" }, title), el("div", { class: "step-sub" }, sub)),
      !done && action ? el("button", { class: "btn sm primary", onclick: action.fn }, action.label) : null);
    c.append(el("div", { class: "card pad-lg onboard", style: "margin-bottom:24px" },
      el("div", { class: "card-head" }, el("h3", {}, "Welcome to Pleiades"), el("span", { class: "pill run" }, "setup")),
      el("div", { class: "muted", style: "margin-bottom:8px" }, "Three steps and your first character is alive — local model, persistent memory, and a full tool belt."),
      el("div", { class: "steps" },
        step(1, s.services.anamnesis.up, "Start the memory daemon", "Run `anamnesis status` in a terminal — it powers every character's persistent memory.", null),
        step(2, !needsModel, "Get a model", "Fetch a GGUF from Hugging Face — Pleiades picks the best quantization for this machine.",
          { label: "Fetch a model", fn: () => go("models") }),
        step(3, !needsChar, "Create a character", "One name binds memory, email, vault, browser, and the model.",
          { label: "New character", fn: () => go("characters") }))));
  }

  /* services */
  const svc = (name, info, extra, actions) => {
    const up = info.up;
    let dot = up ? "up" : "down", pill = up ? "up" : "down", label = up ? "online" : "offline";
    if (!up && info.state === "on_demand") { dot = "idle"; pill = "warn"; label = "idle — starts on first chat"; }
    if (!up && info.state === "no_model") { dot = "idle"; pill = "warn"; label = "no model yet"; }
    return el("div", { class: "svc" },
      el("div", { class: "svc-top" },
        el("div", { class: "inline-actions" }, el("span", { class: `dot ${dot}` }), el("span", { class: "svc-name" }, name)),
        el("span", { class: `pill ${pill}` }, label)),
      el("div", { class: "svc-url" }, info.url || "—"),
      extra ? el("div", { class: "svc-note", title: extra }, extra) : null,
      actions ? el("div", { class: "svc-actions" }, ...actions) : null);
  };

  const searxBtn = el("button", { class: "btn sm", onclick: async (e) => {
    const dir = s.services.searxng.up ? "down" : "up";
    e.target.disabled = true; e.target.innerHTML = `<span class="spinner"></span>`;
    try {
      await api.post(`/api/search/${dir}`);
      const timer = setInterval(async () => {
        const st = await api.get("/api/search/status").catch(() => null);
        if (!st || st.status === "working") return;
        clearInterval(timer);
        if (st.status === "error") { err(new Error(st.error)); renderDashboard(); }
        else { ok(dir === "up" ? "Search service starting" : "Search service stopped"); setTimeout(renderDashboard, 1500); }
      }, 1200);
    } catch (er) { e.target.disabled = false; err(er); }
  } }, s.services.searxng.up ? "Stop" : "Start");

  c.append(el("div", { class: "section-label" }, "Services"));
  c.append(el("div", { class: "svc-grid" },
    svc("Memory · Anamnesis", s.services.anamnesis, `${s.services.anamnesis.characters} character(s) registered`),
    svc("Inference engine", s.services.inference,
      s.services.inference.model_path ? `default model: ${s.services.inference.model_path.split(/[\\/]/).pop()}`
        : s.services.inference.state === "no_model" ? "fetch a model under Models" : "serves registered models on demand"),
    svc("Web search · SearXNG", s.services.searxng, "private local search the characters use", [searxBtn])));

  /* stats */
  c.append(el("div", { class: "section-label" }, "At a glance"));
  const stat = (val, label, sub, onclick) => el("div", { class: "card stat hoverable", style: onclick ? "cursor:pointer" : "", onclick },
    el("div", { class: "stat-row" }, el("span", { class: "stat-val" }, String(val)), sub ? el("span", { class: "stat-sub" }, sub) : null),
    el("div", { class: "stat-label" }, label));
  c.append(el("div", { class: "grid cols-3" },
    stat(s.counts.profiles, "Characters", s.counts.orphans ? `${s.counts.orphans} adoptable` : "", () => go("characters")),
    stat(s.counts.models, "Models registered", `${s.counts.models_running} running`, () => go("models")),
    stat(s.counts.models_running, "Live model servers", "", () => go("models"))));

  /* running models */
  if (s.running_models.length) {
    c.append(el("div", { class: "section-label" }, "Running models"));
    const list = el("div", { class: "list" });
    s.running_models.forEach(m => list.append(el("div", { class: "list-row" },
      el("span", { class: "pill run" }, "● live"),
      el("span", { class: "list-key" }, m.name),
      el("span", { class: "list-spacer" }),
      el("span", { class: "badge-soft" }, `port ${m.port}`),
      el("span", { class: "badge-soft" }, m.n_gpu_layers === -1 ? "GPU: all" : `GPU layers: ${m.n_gpu_layers}`))));
    c.append(el("div", { class: "card" }, list));
  }

  /* shortcuts */
  c.append(el("div", { class: "section-label" }, "Jump in"));
  c.append(el("div", { class: "grid cols-2" },
    el("div", { class: "card hoverable", style: "cursor:pointer", onclick: () => go("chat") },
      el("div", { class: "card-head" }, el("h3", {}, "💬 Chat with a character"), el("span", { class: "pill run" }, "live")),
      el("div", { class: "muted" }, "Streamed conversation with memory injection and the full per-character tool belt — search, email, browser, vault.")),
    el("div", { class: "card hoverable", style: "cursor:pointer", onclick: () => go("work") },
      el("div", { class: "card-head" }, el("h3", {}, "⚡ Run agent work"), el("span", { class: "pill violet" }, "harness")),
      el("div", { class: "muted" }, "Give a task to the workspace harness — files, git, shell, subagents — with a live tool-call feed and approval gate."))));
}

/* ═══════════════ CHAT ═══════════════ */
async function renderChat() {
  const c = $("#content"); c.innerHTML = "";
  let profiles = [];
  try { profiles = (await api.get("/api/profiles")).profiles; } catch (e) { return err(e); }

  if (!profiles.length) {
    c.classList.remove("full");
    c.append(el("div", { class: "empty" },
      el("div", { class: "empty-ico" }, "💬"),
      el("h3", {}, "No characters to chat with"),
      el("div", {}, "Create a character first — it gets its own memory, tools, and model."),
      el("div", { style: "margin-top:18px" }, el("button", { class: "btn primary", onclick: () => go("characters") }, "Create a character"))));
    return;
  }

  if (!state.chatWith || !profiles.find(p => p.name === state.chatWith)) state.chatWith = profiles[0].name;
  const active = profiles.find(p => p.name === state.chatWith);

  /* layout */
  const side = el("div", { class: "chat-side" }, el("div", { class: "chat-side-label" }, "Characters"));
  profiles.forEach(p => side.append(el("button", {
    class: `chat-char ${p.name === state.chatWith ? "active" : ""}`,
    onclick: () => { state.chatWith = p.name; renderChat(); },
  }, el("span", { class: "avatar" }, initials(p.name)), el("span", {}, p.name))));

  const scroll = el("div", { class: "chat-scroll" });
  const history = state.chats[state.chatWith] || [];
  const addMsg = (role, text) => {
    const m = el("div", { class: `msg ${role}` },
      el("span", { class: "avatar" }, role === "user" ? "You" : initials(state.chatWith)),
      el("div", { class: "bubble" }, text));
    scroll.append(m);
    scroll.scrollTop = scroll.scrollHeight;
    return m.querySelector(".bubble");
  };

  if (!history.length) {
    scroll.append(el("div", { class: "empty", style: "margin:auto" },
      el("div", { class: "empty-ico" }, "✶"),
      el("h3", {}, `Say hello to ${state.chatWith}`),
      el("div", {}, "Every turn is stored in persistent memory. The character can search the web, read its inbox, drive its browser, and use its vault.")));
  } else {
    history.forEach(m => addMsg(m.role, m.text));
  }

  /* input */
  const ta = el("textarea", { rows: "1", placeholder: `Message ${state.chatWith}…` });
  ta.addEventListener("input", () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 160) + "px"; });
  const sendBtn = el("button", { class: "send-btn", title: "Send (Enter)" },
    el("span", { html: `<svg viewBox="0 0 24 24"><path d="M2 21 23 12 2 3v7l15 2-15 2v7z"/></svg>` }));

  let busy = false;
  async function send() {
    const text = ta.value.trim();
    if (!text || busy) return;
    busy = true; sendBtn.disabled = true; ta.value = ""; ta.style.height = "auto";
    if (!state.chats[state.chatWith]) state.chats[state.chatWith] = [];
    const hist = state.chats[state.chatWith];
    if (!hist.length) scroll.innerHTML = "";                     // clear empty state
    hist.push({ role: "user", text });
    addMsg("user", text);

    const bubble = addMsg("assistant", "");
    bubble.append(el("span", { class: "typing" }, el("span"), el("span"), el("span")));
    let acc = "";

    try {
      const r = await fetch(`/api/profiles/${encodeURIComponent(state.chatWith)}/chat`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      if (!r.ok || !r.body) throw new Error(`${r.status} ${r.statusText}`);
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n"); buf = lines.pop();
        for (const line of lines) {
          if (!line.trim()) continue;
          let evt; try { evt = JSON.parse(line); } catch { continue; }
          if (evt.type === "chunk") {
            acc += evt.text;
            bubble.textContent = acc;
            scroll.scrollTop = scroll.scrollHeight;
          } else if (evt.type === "error") {
            throw new Error(evt.error);
          }
        }
      }
      if (!acc) { acc = "(no reply)"; bubble.textContent = acc; }
      hist.push({ role: "assistant", text: acc });
    } catch (e) {
      bubble.closest(".msg").classList.add("error");
      bubble.textContent = `Couldn't reach ${state.chatWith}: ${e.message}. Check that the model and the Anamnesis daemon are running (see Overview).`;
      hist.push({ role: "error", text: bubble.textContent });
    } finally {
      busy = false; sendBtn.disabled = false; ta.focus();
    }
  }
  sendBtn.addEventListener("click", send);
  ta.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });

  const main = el("div", { class: "chat-main" },
    el("div", { class: "chat-head" },
      el("span", { class: "avatar" }, initials(state.chatWith)),
      el("div", {},
        el("div", { class: "chat-head-name" }, state.chatWith),
        el("div", { class: "chat-head-sub" }, `${active.model ? "model: " + active.model : "default engine"} · memory on · tools on`)),
      el("span", { class: "list-spacer" }),
      el("button", { class: "btn sm ghost", onclick: () => { state.chats[state.chatWith] = []; renderChat(); } }, "Clear view"),
      el("button", { class: "btn sm", onclick: () => go("characters", state.chatWith) }, "Profile")),
    scroll,
    el("div", { class: "chat-input-bar" },
      el("div", { class: "chat-input-box" }, ta, sendBtn),
      el("div", { class: "chat-note" }, "Conversation history lives in the character's Anamnesis memory — clearing this view forgets nothing.")));

  c.append(el("div", { class: "chat-wrap" }, side, main));
  ta.focus();
}

/* ═══════════════ AGENT (work) ═══════════════ */
let workTimer = null;

async function renderWork() {
  if (workTimer) { clearInterval(workTimer); workTimer = null; }
  loading();
  let profiles = [], settings = {};
  try {
    profiles = (await api.get("/api/profiles")).profiles;
    settings = (await api.get("/api/settings")).settings;
  } catch (e) { return err(e); }
  const c = $("#content"); c.innerHTML = "";

  /* composer */
  const ta = el("textarea", { class: "input mono", rows: "3", placeholder: `find the largest .py file in this repo and summarize it` });
  const charSel = el("select", { class: "select" }, el("option", { value: "" }, "No character (plain workspace)"));
  profiles.forEach(p => charSel.append(el("option", { value: p.name }, `as ${p.name} — its memory, vault & inbox`)));
  const tierSel = el("select", { class: "select" }, el("option", { value: "" }, `default (${settings.default_tier || "local"})`));
  Object.keys(settings.tiers || {}).forEach(t => tierSel.append(el("option", { value: t }, t)));
  const policySel = el("select", { class: "select" },
    ...["", "ask", "allow", "deny"].map(o => el("option", { value: o, selected: o === "" ? "selected" : null },
      o === "" ? `default (${settings.exec_policy || "ask"})` : o)));

  const runBtn = el("button", { class: "btn primary", onclick: async (e) => {
    const task = ta.value.trim();
    if (!task) return toast("Describe a task first");
    e.target.disabled = true;
    try {
      const r = await api.post("/api/work", { task, character: charSel.value, tier: tierSel.value, policy: policySel.value });
      state.workJob = r.id;
      renderWork();
    } catch (er) { e.target.disabled = false; err(er); }
  } }, "▶ Run task");

  c.append(el("div", { class: "card pad-lg", style: "margin-bottom:22px" },
    el("div", { class: "card-head" }, el("h3", {}, "New task"),
      el("span", { class: "pill violet" }, "60+ tools · permission gated")),
    field("Task", ta, "Plain language. The agent plans, calls tools, and reports back."),
    el("div", { class: "field-row tri" },
      field("Identity", charSel), field("Tier", tierSel), field("Policy", policySel,
        "ask = approve each gated tool call here in the UI")),
    el("div", { class: "inline-actions", style: "justify-content:flex-end" }, runBtn)));

  /* job list + live feed */
  const feedHost = el("div", {});
  c.append(feedHost);

  async function refresh() {
    let jobs;
    try { jobs = (await api.get("/api/work")).jobs; } catch { return; }
    feedHost.innerHTML = "";

    if (state.workJob) {
      let job;
      try { job = await api.get(`/api/work/${state.workJob}`); } catch { state.workJob = null; return; }
      feedHost.append(renderJob(job));
    }

    const others = jobs.filter(j => j.id !== state.workJob);
    if (others.length) {
      feedHost.append(el("div", { class: "section-label" }, "History"));
      const list = el("div", { class: "list" });
      others.forEach(j => list.append(el("div", { class: "list-row job-row", onclick: () => { state.workJob = j.id; refresh(); } },
        el("span", { class: `pill ${j.status === "running" ? "run" : j.status === "done" ? "up" : j.status === "error" ? "bad" : "down"}` }, j.status),
        el("div", {}, el("div", { class: "list-key" }, j.task.length > 90 ? j.task.slice(0, 90) + "…" : j.task),
          el("div", { class: "list-meta" }, `${j.character ? "as " + j.character + " · " : ""}${ago(j.started)}`)),
        el("span", { class: "list-spacer" }))));
      feedHost.append(el("div", { class: "card" }, list));
    } else if (!state.workJob) {
      feedHost.append(el("div", { class: "empty" },
        el("div", { class: "empty-ico" }, "⚡"),
        el("h3", {}, "No agent runs yet"),
        el("div", {}, "Describe a task above — the live tool-call feed appears here.")));
    }
  }

  function renderJob(job) {
    const statusPill = el("span", { class: `pill ${job.status === "running" ? "run" : job.status === "done" ? "up" : job.status === "error" ? "bad" : "down"}` }, job.status);
    const card = el("div", { class: "card pad-lg" },
      el("div", { class: "card-head" },
        el("div", { class: "inline-actions" }, statusPill, el("h3", {}, job.task.length > 80 ? job.task.slice(0, 80) + "…" : job.task)),
        el("div", { class: "inline-actions" },
          job.character ? el("span", { class: "badge-soft" }, `as ${job.character}`) : null,
          job.status === "running" ? el("button", { class: "btn sm danger", onclick: async () => {
            try { await api.post(`/api/work/${job.id}/cancel`); ok("Cancelling…"); } catch (e) { err(e); }
          } }, "■ Stop") : el("button", { class: "btn sm ghost", onclick: () => { state.workJob = null; refresh(); } }, "Dismiss"))));

    if (job.pending_approval) {
      card.append(el("div", { class: "approval" },
        el("span", {}, "⚠️"),
        el("div", { style: "flex:1;min-width:200px" },
          el("div", {}, "The agent wants to run ", el("span", { class: "tool" }, job.pending_approval.tool)),
          el("div", { class: "args" }, job.pending_approval.args)),
        el("button", { class: "btn sm good", onclick: () => api.post(`/api/work/${job.id}/approve`, { approve: true }).catch(err) }, "✓ Approve"),
        el("button", { class: "btn sm danger", onclick: () => api.post(`/api/work/${job.id}/approve`, { approve: false }).catch(err) }, "✕ Deny")));
    }

    const feed = el("div", { class: "work-feed" });
    (job.events || []).forEach(ev => {
      if (ev.kind === "tool_call") feed.append(el("div", { class: "evt" },
        el("span", { class: "evt-ico" }, "→"), el("span", { class: "name" }, ev.name)));
      else if (ev.kind === "tool_result") feed.append(el("div", { class: "evt" },
        el("span", { class: "evt-ico" }, ev.ok ? "✓" : "✕"),
        el("div", {}, el("span", { class: "name" }, ev.name || ""), el("div", { class: "out" }, (ev.output || "").split("\n").slice(0, 3).join("\n")))));
      else if (ev.kind === "text") feed.append(el("div", { class: "evt text" }, ev.text));
    });
    if (job.status === "running" && !job.pending_approval)
      feed.append(el("div", { class: "evt" }, el("span", { class: "spinner" }), el("span", { class: "muted" }, "working…")));
    card.append(feed);

    if (job.result) card.append(el("div", { class: "result-block" }, job.result));
    if (job.error) card.append(el("div", { class: "result-block", style: "color:var(--bad)" }, job.error));
    return card;
  }

  await refresh();
  workTimer = setInterval(() => { if (state.view === "work") refresh(); else { clearInterval(workTimer); workTimer = null; } }, 1500);
}

/* ═══════════════ CHARACTERS ═══════════════ */
async function renderCharacters() {
  if (state.detail) return renderCharacterDetail(state.detail);
  loading();
  let data;
  try { data = await api.get("/api/profiles"); } catch (e) { return err(e); }
  state.cache.models = (await api.get("/api/models").catch(() => ({ models: [] }))).models;
  const c = $("#content"); c.innerHTML = "";

  c.append(el("div", { class: "inline-actions", style: "justify-content:flex-end;margin-bottom:18px;gap:10px" },
    data.orphans.length ? el("button", { class: "btn", onclick: () => adoptDialog(data.orphans) }, `Adopt existing (${data.orphans.length})`) : null,
    el("button", { class: "btn primary", onclick: createCharacterDialog }, "＋ New character")));

  if (!data.profiles.length) {
    c.append(el("div", { class: "empty" },
      el("div", { class: "empty-ico" }, "✶"),
      el("h3", {}, "No characters yet"),
      el("div", {}, "Create one to bind email, vault, browser, and a model under a single identity."),
      el("div", { style: "margin-top:18px" }, el("button", { class: "btn primary", onclick: createCharacterDialog }, "＋ New character"))));
    return;
  }

  const grid = el("div", { class: "char-grid" });
  data.profiles.forEach(p => {
    const caps = [];
    caps.push(el("span", { class: `pill ${p.has_email ? "up" : "down"}` }, "✉ email"));
    caps.push(el("span", { class: `pill ${p.discord_enabled ? "violet" : "down"}` }, "discord"));
    caps.push(el("span", { class: `pill ${p.model ? "run" : "down"}` }, p.model ? `▤ ${p.model}` : "default model"));
    grid.append(el("div", { class: "char-card", onclick: () => go("characters", p.name) },
      el("div", { class: "char-top" },
        el("div", { class: "avatar" }, initials(p.name)),
        el("div", { style: "min-width:0" }, el("div", { class: "char-name" }, p.name),
          el("div", { class: "char-id" }, p.email_address || "no email configured"))),
      el("div", { class: "char-caps" }, ...caps),
      el("div", { class: "char-foot" },
        el("button", { class: "btn sm", onclick: (e) => { e.stopPropagation(); state.chatWith = p.name; go("chat"); } }, "💬 Chat"),
        el("button", { class: "btn sm ghost", onclick: (e) => { e.stopPropagation(); go("characters", p.name); } }, "Configure →"))));
  });
  c.append(grid);
}

function createCharacterDialog() {
  const name = textInput("", { placeholder: "alice" });
  const email = textInput("", { placeholder: "alice@example.com" });
  const persona = el("select", { class: "select" },
    ...["auto", "file", "inline", "disabled"].map(o => el("option", { value: o }, o)));
  const body = el("div", {},
    field("Character name", name, "Lowercase, no spaces. Creates the Anamnesis character + profile + vault."),
    field("Email address (optional)", email, "You can configure full email + password later in the character's Email tab."),
    field("Persona source", persona));
  modal.open({
    title: "New character", body,
    footer: [
      el("button", { class: "btn ghost", onclick: () => modal.close() }, "Cancel"),
      el("button", { class: "btn primary", onclick: async (e) => {
        if (!name.value.trim()) return toast("Name required");
        e.target.disabled = true;
        try {
          await api.post("/api/profiles", { name: name.value.trim(), email_address: email.value.trim(), persona_source: persona.value });
          modal.close(); ok("Character created", name.value.trim()); go("characters");
        } catch (er) { e.target.disabled = false; err(er); }
      } }, "Create"),
    ],
  });
}

function adoptDialog(orphans) {
  const sel = el("select", { class: "select" }, ...orphans.map(o => el("option", { value: o }, o)));
  modal.open({
    title: "Adopt an Anamnesis character",
    body: el("div", {},
      el("div", { class: "callout", style: "margin-bottom:16px" }, el("span", { class: "c-ico" }, "✶"),
        el("div", {}, "These Anamnesis characters exist but have no Pleiades profile. Adopting keeps their memory and history intact.")),
      field("Character", sel)),
    footer: [
      el("button", { class: "btn ghost", onclick: () => modal.close() }, "Cancel"),
      el("button", { class: "btn primary", onclick: async (e) => {
        e.target.disabled = true;
        try { await api.post("/api/adopt", { name: sel.value }); modal.close(); ok("Adopted", sel.value); go("characters"); }
        catch (er) { e.target.disabled = false; err(er); }
      } }, "Adopt"),
    ],
  });
}

/* ── character detail with tabs ── */
async function renderCharacterDetail(name) {
  loading();
  let p, models;
  try {
    p = await api.get(`/api/profiles/${encodeURIComponent(name)}`);
    models = (await api.get("/api/models")).models;
  } catch (e) { return err(e); }

  const c = $("#content"); c.innerHTML = "";
  c.append(el("button", { class: "back-link", onclick: () => go("characters") }, "← All characters"));
  c.append(el("div", { class: "detail-head" },
    el("div", { class: "avatar lg" }, initials(p.name)),
    el("div", {},
      el("h2", { style: "font-size:20px" }, p.name),
      el("div", { class: "muted mono", style: "font-size:12.5px" }, p.email_address || "no email configured")),
    el("div", { class: "list-spacer" }),
    el("button", { class: "btn sm", onclick: () => { state.chatWith = p.name; go("chat"); } }, "💬 Chat"),
    el("button", { class: "btn danger sm", onclick: () => deleteCharacter(p.name) }, "Delete")));

  const tabBar = el("div", { class: "tabs" });
  const pane = el("div", {});
  const TABS = {
    identity:    () => paneIdentity(p, models),
    email:       () => paneEmail(p),
    credentials: () => paneCredentials(p),
    discord:     () => paneDiscord(p),
    model:       () => paneModel(p, models),
    anamnesis:   () => paneAnamnesis(p),
  };
  const labels = { identity: "Identity", email: "Email", credentials: "Credentials", discord: "Discord", model: "Model", anamnesis: "Memory" };
  let active = (state.cache.charTab && TABS[state.cache.charTab]) ? state.cache.charTab : "identity";
  Object.keys(TABS).forEach(k => {
    tabBar.append(el("button", { class: `tab ${k === active ? "active" : ""}`, onclick: () => {
      active = k; state.cache.charTab = k;
      $$(".tab", tabBar).forEach((b, i) => b.classList.toggle("active", Object.keys(TABS)[i] === k));
      pane.innerHTML = ""; pane.append(TABS[k]());
    } }, labels[k]));
  });
  c.append(tabBar, pane);
  pane.append(TABS[active]());
}

function paneIdentity(p, models) {
  return el("div", { class: "grid cols-2" },
    el("div", { class: "card" },
      el("div", { class: "card-head" }, el("h3", {}, "Bindings")),
      el("div", { class: "list" },
        kvRow("Email", p.has_email ? p.email_address : "not configured", p.has_email ? "up" : "down"),
        kvRow("Model", p.model || "default engine", p.model ? "run" : "down"),
        kvRow("Discord", p.discord_enabled ? "enabled" : "disabled", p.discord_enabled ? "violet" : "down"),
        kvRow("Vault entries", String((p.vault || []).length), "up"),
        kvRow("Browser profile", "persistent", "up"))),
    el("div", { class: "card" },
      el("div", { class: "card-head" }, el("h3", {}, "Memory & persona")),
      el("div", { class: "callout" }, el("span", { class: "c-ico" }, "✶"),
        el("div", { html: "Persona and long-term memory are handled by <b>Anamnesis</b> in the background. Use the <b>Memory</b> tab to change the upstream model or browse stored memories." }))));
}
function kvRow(k, v, pill) {
  return el("div", { class: "list-row" },
    el("span", { class: "list-key" }, k),
    el("span", { class: "list-spacer" }),
    el("span", { class: `pill ${pill}` }, v));
}

/* ── Email tab ── */
function paneEmail(p) {
  if (!p.has_email) return emailSetupCard(p);

  const wrap = el("div", {});
  const listCard = el("div", { class: "card" });
  const unreadOnly = el("input", { type: "checkbox" });
  let loadedOnce = false;

  const load = async () => {
    listCard.innerHTML = ""; listCard.append(el("div", { class: "skeleton" }, "Checking the inbox…"));
    let r;
    try { r = await api.get(`/api/profiles/${p.name}/email/inbox?limit=25&unread=${unreadOnly.checked ? 1 : 0}`); }
    catch (e) { listCard.innerHTML = ""; listCard.append(el("div", { class: "svc-note", style: "padding:16px" }, String(e.message || e))); return; }
    loadedOnce = true;
    listCard.innerHTML = "";
    if (!r.messages.length) {
      listCard.append(el("div", { class: "empty", style: "padding:30px" }, el("div", { class: "empty-ico" }, "✉"),
        el("h3", {}, unreadOnly.checked ? "No unread mail" : "Inbox is empty")));
      return;
    }
    const list = el("div", { class: "list" });
    r.messages.forEach(m => list.append(el("div", { class: `list-row mail-row ${m.unread ? "unread" : ""}`, onclick: () => openMessage(m.id) },
      el("span", { class: `dot ${m.unread ? "up" : "idle"}`, title: m.unread ? "unread" : "read" }),
      el("span", { class: "m-from" }, m.from),
      el("span", { class: "m-subj" }, m.subject),
      el("span", { class: "badge-soft" }, (m.date || "").replace(/\s[+-]\d{4}.*$/, "")))));
    listCard.append(list);
  };

  const openMessage = async (mid) => {
    let msg;
    try { msg = await api.get(`/api/profiles/${p.name}/email/message/${encodeURIComponent(mid)}`); }
    catch (e) { return err(e); }
    modal.open({
      title: msg.subject,
      body: el("div", {},
        el("div", { class: "list", style: "margin-bottom:12px" },
          kvRow("From", msg.from, "down"), kvRow("To", msg.to, "down"), kvRow("Date", msg.date, "down")),
        el("div", { class: "mail-body" }, msg.body || "(empty body)")),
      footer: [
        el("button", { class: "btn", onclick: () => { modal.close(); composeDialog(p, msg.from.replace(/^.*<|>.*$/g, ""), `Re: ${msg.subject}`); } }, "↩ Reply"),
        el("button", { class: "btn ghost", onclick: () => { modal.close(); load(); } }, "Close"),
      ],
    });
  };

  unreadOnly.addEventListener("change", load);
  wrap.append(el("div", { class: "inline-actions", style: "justify-content:space-between;margin-bottom:14px" },
    el("div", { class: "inline-actions" },
      el("span", { class: "pill up" }, p.email_address),
      el("label", { class: "inline-actions", style: "gap:6px;color:var(--text-dim);font-size:13px" }, unreadOnly, "unread only")),
    el("div", { class: "inline-actions" },
      el("button", { class: "btn sm", onclick: load }, "⟳ Refresh"),
      el("button", { class: "btn sm primary", onclick: () => composeDialog(p) }, "✎ Compose"),
      el("button", { class: "btn sm", onclick: () => {
        const pane2 = wrap.parentElement; pane2.innerHTML = ""; pane2.append(emailSetupCard(p));
      } }, "⚙ Settings"))));
  wrap.append(listCard);
  load();
  return wrap;
}

function composeDialog(p, to = "", subject = "") {
  const toIn = textInput(to, { placeholder: "recipient@example.com", class: "input mono" });
  const subjIn = textInput(subject, { placeholder: "subject" });
  const bodyIn = el("textarea", { class: "input", rows: "8", placeholder: "write your message…" });
  modal.open({
    title: `New email · from ${p.email_address}`,
    body: el("div", {}, field("To", toIn), field("Subject", subjIn), field("Body", bodyIn)),
    footer: [
      el("button", { class: "btn ghost", onclick: () => modal.close() }, "Cancel"),
      el("button", { class: "btn primary", onclick: async (e) => {
        if (!toIn.value.trim()) return toast("Recipient required");
        e.target.disabled = true;
        try {
          await api.post(`/api/profiles/${p.name}/email/send`, { to: toIn.value.trim(), subject: subjIn.value, body: bodyIn.value });
          modal.close(); ok("Sent", toIn.value.trim());
        } catch (er) { e.target.disabled = false; err(er); }
      } }, "Send"),
    ],
  });
}

function emailSetupCard(p) {
  const email = textInput(p.email_address, { placeholder: "alice@example.com" });
  const imapH = textInput(p.imap_host, { placeholder: "imap.example.com", class: "input mono" });
  const imapP = textInput(p.imap_port, { type: "number", class: "input mono" });
  const smtpH = textInput(p.smtp_host, { placeholder: "smtp.example.com", class: "input mono" });
  const smtpP = textInput(p.smtp_port, { type: "number", class: "input mono" });
  const pass  = textInput("", { type: "password", placeholder: "app password (leave blank to keep current)" });

  const preset = el("select", { class: "select" }, el("option", { value: "" }, "Choose a provider preset…"));
  api.get("/api/email/presets").then(presets => {
    Object.entries(presets).forEach(([k]) => preset.append(el("option", { value: k }, k)));
    preset.addEventListener("change", async () => {
      if (!preset.value) return;
      const ps = (await api.get("/api/email/presets"))[preset.value];
      imapH.value = ps.imap_host || ""; imapP.value = ps.imap_port || 993;
      smtpH.value = ps.smtp_host || ""; smtpP.value = ps.smtp_port || 587;
      if (ps.note) toast(preset.value, ps.note);
    });
  });

  return el("div", { class: "card pad-lg", style: "max-width:660px" },
    el("div", { class: "card-head" }, el("h3", {}, "Email account"),
      el("span", { class: `pill ${p.has_email ? "up" : "down"}` }, p.has_email ? "configured" : "not set")),
    el("div", { class: "callout", style: "margin-bottom:20px" }, el("span", { class: "c-ico" }, "✉"),
      el("div", { html: "The character reads and sends from its own inbox — useful for receiving verification codes. The password is stored <b>encrypted</b> in this character's vault as <span class='kbd'>email.password</span>." })),
    field("Provider preset", preset),
    field("Email address", email),
    el("div", { class: "field-row" },
      field("IMAP host", imapH), field("IMAP port", imapP)),
    el("div", { class: "field-row" },
      field("SMTP host", smtpH), field("SMTP port", smtpP)),
    field("App password", pass, "Stored encrypted at rest. Never logged, never auto-injected into prompts."),
    el("button", { class: "btn primary", onclick: async (e) => {
      e.target.disabled = true;
      try {
        await api.post(`/api/profiles/${p.name}/email`, {
          email_address: email.value.trim(), imap_host: imapH.value.trim(), imap_port: +imapP.value || 993,
          smtp_host: smtpH.value.trim(), smtp_port: +smtpP.value || 587,
          password: pass.value || null,
        });
        ok("Email saved"); go("characters", p.name);
      } catch (er) { e.target.disabled = false; err(er); }
    } }, "Save email settings"));
}

/* ── Credentials / vault tab ── */
function paneCredentials(p) {
  const wrap = el("div", {});
  wrap.append(el("div", { class: "inline-actions", style: "justify-content:space-between;margin-bottom:16px" },
    el("div", { class: "muted" }, "Encrypted secrets for this character. Values are revealed only on explicit request."),
    el("button", { class: "btn primary sm", onclick: () => vaultEntryDialog(p.name) }, "＋ Add entry")));

  const card = el("div", { class: "card" });
  const list = el("div", { class: "list" });
  const entries = p.vault || [];
  if (!entries.length) {
    card.append(el("div", { class: "empty", style: "padding:34px" }, el("div", { class: "empty-ico" }, "🔒"),
      el("h3", {}, "Vault is empty"), el("div", {}, "Add email passwords, the Discord token, or per-site credentials.")));
  } else {
    entries.forEach(en => {
      const isReserved = en.reserved;
      const isSite = !!en.site;
      const valSpan = el("span", { class: "secret-dots" }, "••••••••••");
      const note = en.meta && en.meta.note ? en.meta.note : "";
      list.append(el("div", { class: "list-row" },
        el("span", { class: `pill ${isReserved ? "warn" : (isSite ? "violet" : "up")}` }, isReserved ? "managed" : (isSite ? "site" : "custom")),
        el("div", {}, el("div", { class: "list-key mono" }, en.key),
          el("div", { class: "list-meta" }, (note ? note + " · " : "") + `updated ${ago(en.updated_at)}`)),
        el("span", { class: "list-spacer" }),
        valSpan,
        el("button", { class: "btn sm", onclick: async (ev) => {
          if (valSpan.dataset.shown === "1") {
            valSpan.className = "secret-dots"; valSpan.textContent = "••••••••••"; valSpan.dataset.shown = "0"; ev.target.textContent = "Reveal";
          } else {
            try {
              const r = await api.get(`/api/profiles/${p.name}/vault/${encodeURIComponent(en.key)}`);
              valSpan.className = "reveal-val"; valSpan.textContent = r.value; valSpan.dataset.shown = "1"; ev.target.textContent = "Hide";
            } catch (er) { err(er); }
          }
        } }, "Reveal"),
        el("button", { class: "btn sm", onclick: () => vaultEntryDialog(p.name, en) }, "Edit"),
        el("button", { class: "btn sm danger", onclick: async () => {
          if (!confirm(`Delete vault entry "${en.key}"?`)) return;
          try { await api.del(`/api/profiles/${p.name}/vault/${encodeURIComponent(en.key)}`); ok("Deleted"); go("characters", p.name); }
          catch (er) { err(er); }
        } }, "✕")));
    });
    card.append(list);
  }
  wrap.append(card);
  return wrap;
}

function vaultEntryDialog(name, existing) {
  const isEdit = !!existing;
  const valInput = el("textarea", { class: "input mono", rows: "3", placeholder: isEdit ? "new secret value" : "secret value" });
  const noteInput = textInput(existing && existing.meta && existing.meta.note ? existing.meta.note : "", { placeholder: "optional label" });

  if (isEdit) {
    // Editing: the key is fixed; just take a new value/note.
    modal.open({
      title: `Edit ${existing.key}`,
      body: el("div", {},
        field("Value", valInput, "Encrypted with Fernet before it touches disk."),
        field("Note (optional)", noteInput)),
      footer: [
        el("button", { class: "btn ghost", onclick: () => modal.close() }, "Cancel"),
        el("button", { class: "btn primary", onclick: async (e) => {
          if (!valInput.value) return toast("Value required");
          e.target.disabled = true;
          try {
            await api.post(`/api/profiles/${name}/vault`, { key: existing.key, value: valInput.value, note: noteInput.value.trim() });
            modal.close(); ok("Updated", existing.key); go("characters", name);
          } catch (er) { e.target.disabled = false; err(er); }
        } }, "Update"),
      ],
    });
    return;
  }

  // New entry: pick a type, get the right fields for it.
  const siteIn = textInput("", { placeholder: "example.com  or  https://example.com/login", class: "input mono" });
  const userIn = textInput("", { placeholder: "username or email used to sign in" });
  const passIn = textInput("", { type: "password", placeholder: "password", class: "input mono" });
  const keyIn = textInput("", { placeholder: "my-api-key", class: "input mono" });
  const formHost = el("div", {});

  const forms = {
    site: () => el("div", {},
      field("Website", siteIn, "Stored under site:<domain> — the model finds it by domain."),
      field("Username / email", userIn),
      field("Password", passIn),
      field("Note (optional)", noteInput)),
    custom: () => el("div", {},
      field("Name", keyIn, "Any identifier, e.g. an API key name."),
      field("Secret value", valInput),
      field("Note (optional)", noteInput)),
  };
  const typeSel = el("select", { class: "select" },
    el("option", { value: "site" }, "Website login (site + username + password)"),
    el("option", { value: "custom" }, "Custom secret (name + value)"));
  typeSel.addEventListener("change", () => { formHost.innerHTML = ""; formHost.append(forms[typeSel.value]()); });
  formHost.append(forms.site());

  modal.open({
    title: "Add vault entry",
    body: el("div", {},
      el("div", { class: "callout", style: "margin-bottom:16px" }, el("span", { class: "c-ico" }, "🔒"),
        el("div", { html: "Everything is encrypted at rest. Email password and Discord token are managed from their own tabs." })),
      field("Type", typeSel), formHost),
    footer: [
      el("button", { class: "btn ghost", onclick: () => modal.close() }, "Cancel"),
      el("button", { class: "btn primary", onclick: async (e) => {
        let key, value, note = noteInput.value.trim();
        if (typeSel.value === "site") {
          const domain = siteIn.value.trim().replace(/^https?:\/\//, "").split("/")[0];
          if (!domain) return toast("Website required");
          if (!passIn.value) return toast("Password required");
          key = `site:${domain.toLowerCase()}`; value = passIn.value;
          if (userIn.value.trim()) note = note ? `${userIn.value.trim()} · ${note}` : userIn.value.trim();
        } else {
          if (!keyIn.value.trim()) return toast("Name required");
          if (!valInput.value) return toast("Value required");
          key = keyIn.value.trim(); value = valInput.value;
        }
        e.target.disabled = true;
        try {
          await api.post(`/api/profiles/${name}/vault`, { key, value, note });
          modal.close(); ok("Saved", key); go("characters", name);
        } catch (er) { e.target.disabled = false; err(er); }
      } }, "Save"),
    ],
  });
}

/* ── Discord tab ── */
function paneDiscord(p) {
  const toggle = switchToggle(p.discord_enabled, "Host this character as a Discord bot");
  const reqPing = switchToggle(p.discord_require_mention !== false, "In servers, only respond when pinged (DMs always work)");
  const botsToo = switchToggle(!!p.discord_respond_to_bots, "Respond to other bots' messages");
  const channels = textInput(p.discord_allowed_channels || "", { placeholder: "general, bot-chat, 123456789  (blank = all channels)", class: "input mono" });
  const token = textInput("", { type: "password", placeholder: "bot token (leave blank to keep current)", class: "input mono" });

  const infoCard = el("div", { class: "card" },
    el("div", { class: "card-head" }, el("h3", {}, "Bot status")),
    el("div", { class: "skeleton" }, "Checking the token with Discord…"));
  api.get(`/api/profiles/${p.name}/discord/info`).then(info => {
    infoCard.innerHTML = "";
    infoCard.append(el("div", { class: "card-head" }, el("h3", {}, "Bot status")));
    if (!info.configured) {
      infoCard.append(el("div", { class: "svc-note" }, "No token saved yet — add one below."));
      return;
    }
    if (!info.valid) {
      infoCard.append(el("div", { class: "list" }, kvRow("Token", info.error || "invalid", "down")));
      return;
    }
    const list = el("div", { class: "list" },
      kvRow("Bot account", info.bot.username, "violet"),
      kvRow("Servers", String(info.guilds.length), info.guilds.length ? "up" : "down"));
    info.guilds.forEach(g => list.append(el("div", { class: "list-row" },
      el("span", { class: "dot up" }), el("span", { class: "list-key" }, g.name),
      el("span", { class: "list-spacer" }), el("span", { class: "badge-soft" }, g.id))));
    if (!info.guilds.length) list.append(el("div", { class: "svc-note", style: "padding:10px 0 0" },
      "The bot isn't in any server yet — invite it from the Discord Developer Portal (OAuth2 → URL generator → bot scope)."));
    infoCard.append(list);
  }).catch(e => { infoCard.innerHTML = ""; infoCard.append(el("div", { class: "svc-note" }, String(e.message || e))); });

  const settingsCard = el("div", { class: "card pad-lg" },
    el("div", { class: "card-head" }, el("h3", {}, "Discord bot"),
      el("span", { class: `pill ${p.discord_enabled ? "violet" : "down"}` }, p.discord_enabled ? "enabled" : "disabled")),
    el("div", { class: "callout", style: "margin-bottom:18px" }, el("span", { class: "c-ico" }, "◈"),
      el("div", { html: "Create a bot in the Discord Developer Portal and <b>enable the Message Content intent</b>. Run it with <span class='kbd'>pleiades discord " + esc(p.name) + "</span>." })),
    el("div", { style: "display:flex;flex-direction:column;gap:12px;margin-bottom:18px" }, toggle, reqPing, botsToo),
    field("Allowed channels", channels, "Server channels (names or ids) the bot may speak in. Blank = everywhere it can see."),
    field("Bot token", token, "Stored encrypted in this character's vault."),
    el("button", { class: "btn primary", onclick: async (e) => {
      e.target.disabled = true;
      try {
        await api.post(`/api/profiles/${p.name}/discord`, {
          token: token.value || null,
          enabled: $("input", toggle).checked,
          require_mention: $("input", reqPing).checked,
          respond_to_bots: $("input", botsToo).checked,
          allowed_channels: channels.value.trim(),
        });
        ok("Discord saved"); go("characters", p.name);
      } catch (er) { e.target.disabled = false; err(er); }
    } }, "Save Discord settings"));

  return el("div", { class: "grid cols-2" }, settingsCard, infoCard);
}

/* ── Model assignment tab ── */
function paneModel(p, models) {
  const wrap = el("div", { class: "grid cols-2" });
  const sel = el("select", { class: "select" }, el("option", { value: "" }, "Default engine (PLEIADES_MODEL_PATH)"));
  models.forEach(m => sel.append(el("option", { value: m.name, selected: m.name === p.model ? "selected" : null },
    `${m.name}${m.running ? " · running" : ""}`)));

  const assignCard = el("div", { class: "card" },
    el("div", { class: "card-head" }, el("h3", {}, "Assigned model")),
    el("div", { class: "callout", style: "margin-bottom:18px" }, el("span", { class: "c-ico" }, "▤"),
      el("div", {}, "Each character can run a different model. If the chosen model's server is live, its proxy upstream is repointed immediately.")),
    field("Model for this character", sel),
    el("button", { class: "btn primary sm", onclick: async (e) => {
      e.target.disabled = true;
      try { await api.post(`/api/profiles/${p.name}/model`, { model: sel.value }); ok("Model assigned", sel.value || "default engine"); go("characters", p.name); }
      catch (er) { e.target.disabled = false; err(er); }
    } }, "Assign"));

  const infoCard = el("div", { class: "card" },
    el("div", { class: "card-head" }, el("h3", {}, "Status")),
    el("div", { class: "list" },
      kvRow("Current", p.model || "default engine", p.model ? "run" : "down"),
      kvRow("Registered", p.model_info ? (p.model_info.registered ? "yes" : "missing") : "—", p.model_info && p.model_info.registered ? "up" : "down"),
      kvRow("Server running", p.model_info ? (p.model_info.running ? "yes" : "no") : "—", p.model_info && p.model_info.running ? "run" : "down")),
    el("div", { class: "divider" }),
    el("button", { class: "btn sm", onclick: () => go("models") }, "Manage models →"));

  wrap.append(assignCard, infoCard);
  return wrap;
}

/* ── Anamnesis / Memory tab ── */
function paneAnamnesis(p) {
  const wrap = el("div", {});
  (async () => {
    let d;
    try { d = await api.get(`/api/profiles/${encodeURIComponent(p.name)}/anamnesis`); }
    catch (e) { wrap.append(el("div", { class: "svc-note", style: "padding:16px" }, String(e.message || e))); return; }

    const cfg = d.config || {};
    const upstream = cfg.upstream || {};
    const stats = d.stats || {};

    // ── Upstream config card ──────────────────────────────────────────────
    const urlIn = textInput(upstream.baseUrl || "", { class: "input mono", placeholder: "http://127.0.0.1:8091/v1" });

    const modelSel = el("select", { class: "select" },
      el("option", { value: "" }, "— pick a running Pleiades model —"),
      ...d.running_models.map(m => el("option",
        { value: m.url, selected: m.url === upstream.baseUrl ? "selected" : null },
        `${m.name}  ·  port ${m.port}`)));
    modelSel.onchange = () => { if (modelSel.value) urlIn.value = modelSel.value; };

    const saveBtn = el("button", { class: "btn primary sm" }, "Save & restart");
    saveBtn.onclick = async () => {
      if (!urlIn.value.trim()) return toast("URL required", "", "warn");
      saveBtn.disabled = true; saveBtn.innerHTML = `<span class="spinner"></span> Saving…`;
      try {
        await api.put(`/api/profiles/${encodeURIComponent(p.name)}/anamnesis/upstream`,
          { base_url: urlIn.value.trim() });
        ok("Upstream updated", `${p.name} restarted`);
      } catch (er) { err(er); }
      finally { saveBtn.disabled = false; saveBtn.textContent = "Save & restart"; }
    };

    // ── Stats card ────────────────────────────────────────────────────────
    const statsCard = el("div", { class: "card" },
      el("div", { class: "card-head" }, el("h3", {}, "Memory stats"),
        el("span", { class: `pill ${d.char_status === "running" ? "up" : "down"}` }, d.char_status)),
      el("div", { class: "list" },
        kvRow("Conversation turns", String(stats.turns ?? 0), "up"),
        kvRow("Memory scenes",      String(stats.scenes ?? 0), "up"),
        kvRow("Engrams",            String(stats.engrams ?? 0), "up"),
        kvRow("Observations",       String(stats.observations ?? 0), "up")));

    wrap.append(el("div", { class: "grid cols-2" },
      el("div", { class: "card" },
        el("div", { class: "card-head" }, el("h3", {}, "Upstream model")),
        el("div", { class: "callout", style: "margin-bottom:16px" }, el("span", { class: "c-ico" }, "↑"),
          el("div", {}, "The OpenAI-compatible endpoint this character's Anamnesis proxy forwards requests to. Pick a running Pleiades model or enter a URL manually.")),
        d.running_models.length ? field("Running model", modelSel, "Selects the URL below automatically") : null,
        field("Upstream URL", urlIn, "http://host:port/v1  (llama.cpp or Ollama)"),
        el("div", { class: "inline-actions", style: "justify-content:flex-end" }, saveBtn)),
      statsCard));

    // ── Memory browser ────────────────────────────────────────────────────
    wrap.append(el("div", { class: "section-label" }, "Memory browser"));
    const memBox = el("div", {});
    wrap.append(memBox);

    const loadMem = async (kind, page = 0) => {
      memBox.innerHTML = `<div class="skeleton">Loading ${kind}…</div>`;
      let m;
      try { m = await api.get(`/api/profiles/${encodeURIComponent(p.name)}/anamnesis/memory?kind=${kind}&page=${page}&per_page=25`); }
      catch (e) { memBox.innerHTML = ""; memBox.append(el("div", { class: "svc-note", style:"padding:16px" }, String(e.message || e))); return; }
      memBox.innerHTML = "";

      // Kind tabs
      const kindTabs = el("div", { class: "tabs", style: "margin-bottom:12px" });
      [["scenes","Scenes"],["engrams","Engrams"],["observations","Observations"]].forEach(([k, label]) => {
        const isActive = k === kind;
        const count = isActive ? ` (${m.total})` : "";
        kindTabs.append(el("button", { class: `tab${isActive?" active":""}`,
          onclick: () => loadMem(k) }, label + count));
      });
      memBox.append(kindTabs);

      if (!m.items.length) {
        memBox.append(el("div", { class: "empty", style: "padding:28px" },
          el("div", { class: "empty-ico" }, "◌"), el("h3", {}, "Nothing here yet")));
        return;
      }

      const list = el("div", { class: "list" });
      if (kind === "scenes") {
        m.items.forEach(s => list.append(el("div", { class: "list-row mem-row",
          onclick: () => {
            modal.open({ title: s.title,
              body: el("div", {},
                el("div", { class: "list", style:"margin-bottom:12px" },
                  kvRow("Scene ID", String(s.id), "down"),
                  kvRow("Recalled", `${s.recall_count}×`, "up"),
                  kvRow("Importance", (s.avg_importance||0).toFixed(3), "up"),
                  kvRow("Last updated", new Date(s.updated_at*1000).toLocaleString(), "down")),
                el("p", { style:"line-height:1.6;color:var(--text-dim)" }, s.summary)),
              footer: [el("button", { class:"btn ghost", onclick:()=>modal.close() }, "Close")] });
          } },
          el("div", { class: "mem-main" },
            el("span", { class: "mem-title" }, s.title),
            el("span", { class: "mem-summary" }, s.summary)),
          el("div", { class: "mem-meta" },
            el("span", { class: "badge-soft" }, `recalled ${s.recall_count}×`),
            el("span", { class: "muted", style: "font-size:11px" },
              new Date((s.updated_at||s.created_at)*1000).toLocaleDateString())))));
      } else if (kind === "engrams") {
        m.items.forEach(s => list.append(el("div", { class: "list-row mem-row" },
          el("div", { class: "mem-main" },
            el("span", { class: `badge-soft mem-cat cat-${(s.category||"other").toLowerCase()}` }, s.category || "other"),
            el("span", { class: "mem-summary" }, s.content)),
          el("div", { class: "mem-meta" },
            el("span", { class: "muted", style:"font-size:11px" },
              `imp ${(s.importance||0).toFixed(2)} · decay ${(s.decay_score||0).toFixed(2)}`)
          ))));
      } else {
        m.items.forEach(s => list.append(el("div", { class: "list-row" },
          el("span", { class: "mono", style: "font-size:12px;color:var(--text-dim)" },
            JSON.stringify(s).slice(0, 140)))));
      }
      memBox.append(el("div", { class: "card" }, list));

      // Pagination
      const pages = Math.ceil(m.total / m.per_page);
      if (pages > 1) {
        const pg = el("div", { class: "inline-actions", style: "justify-content:center;margin-top:10px;gap:8px" });
        if (page > 0) pg.append(el("button", { class: "btn sm ghost", onclick: () => loadMem(kind, page-1) }, "← Prev"));
        pg.append(el("span", { class: "muted", style: "font-size:13px" }, `${page+1} / ${pages}`));
        if (page < pages-1) pg.append(el("button", { class: "btn sm ghost", onclick: () => loadMem(kind, page+1) }, "Next →"));
        memBox.append(pg);
      }
    };

    loadMem("scenes");

    // ── Persona viewer ────────────────────────────────────────────────────
    const persona = cfg.persona;
    const inlineContent = persona?.source?.inline?.content;
    if (inlineContent) {
      wrap.append(el("div", { class: "section-label" }, "Persona"));
      wrap.append(el("div", { class: "card" },
        el("div", { class: "card-head" }, el("h3", {}, "Persona"),
          el("span", { class: `pill ${persona.enabled ? "up" : "down"}` }, persona.enabled ? "active" : "disabled"),
          el("span", { class: "muted", style: "font-size:12px;margin-left:6px" },
            `drift ${persona.drift?.enabled?"on":"off"} · evolution ${persona.evolution?.enabled?"on":"off"}`)),
        el("pre", { class: "persona-pre" }, inlineContent)));
    }
  })();
  return wrap;
}

async function deleteCharacter(name) {
  if (!confirm(`Delete character "${name}"? This removes its profile, vault, and (by default) its Anamnesis memory.`)) return;
  try { await api.del(`/api/profiles/${encodeURIComponent(name)}`); ok("Character deleted", name); state.detail = null; state.cache.charTab = "identity"; go("characters"); }
  catch (e) { err(e); }
}

/* ═══════════════ MODELS ═══════════════ */
async function renderModels() {
  loading();
  let data;
  try { data = await api.get("/api/models"); } catch (e) { return err(e); }
  const c = $("#content"); c.innerHTML = "";

  const q = textInput("", { placeholder: "Search Hugging Face for GGUF models… (e.g. llama 3.2 instruct)", class: "input mono", style: "flex:1" });
  const results = el("div", { class: "card", style: "margin-bottom:18px", hidden: "hidden" });
  const doSearch = async () => {
    if (!q.value.trim()) return;
    results.hidden = false; results.innerHTML = ""; results.append(el("div", { class: "skeleton" }, "Searching…"));
    let r;
    try { r = await api.get(`/api/models/hf-search?q=${encodeURIComponent(q.value.trim())}`); }
    catch (e) { results.innerHTML = ""; results.append(el("div", { class: "svc-note", style: "padding:14px" }, String(e.message || e))); return; }
    results.innerHTML = "";
    if (!r.results.length) { results.append(el("div", { class: "svc-note", style: "padding:14px" }, "No GGUF repos found.")); return; }
    const list = el("div", { class: "list" });
    r.results.forEach(m => list.append(el("div", { class: "list-row" },
      el("div", {}, el("div", { class: "list-key" }, m.id),
        el("div", { class: "list-meta" }, `${(m.downloads || 0).toLocaleString()} downloads · ${m.likes || 0} likes`)),
      el("span", { class: "list-spacer" }),
      el("button", { class: "btn sm primary", onclick: (e) => { e.target.disabled = true; startFetch(m.id); } }, "⇣ Download best fit"))));
    results.append(list);
  };
  q.addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });
  c.append(el("div", { class: "inline-actions", style: "margin-bottom:14px" },
    q,
    el("button", { class: "btn", onclick: doSearch }, "Search"),
    el("button", { class: "btn", title: "Fetch a repo you already know", onclick: () => fetchDialog() }, "⇣ By repo id"),
    el("button", { class: "btn primary", onclick: () => modelDialog() }, "＋ Add model")));
  c.append(results);

  if (!data.models.length) {
    c.append(el("div", { class: "empty" }, el("div", { class: "empty-ico" }, "▤"),
      el("h3", {}, "No models registered"),
      el("div", {}, "Fetch a GGUF from Hugging Face — Pleiades picks the best quantization for this machine — or register a local file."),
      el("div", { style: "margin-top:18px", class: "inline-actions", },
        el("button", { class: "btn primary", onclick: () => fetchDialog() }, "⇣ Fetch from Hugging Face"),
        el("button", { class: "btn", onclick: () => modelDialog() }, "Register local GGUF"))));
    return;
  }

  const grid = el("div", { class: "grid cols-2" });
  data.models.forEach(m => {
    const fname = (m.path || "").split(/[\\/]/).pop();
    grid.append(el("div", { class: "card hoverable" },
      el("div", { class: "card-head" },
        el("div", { class: "inline-actions" }, el("span", { class: "card-title-ico" }, "▤"), el("h3", {}, m.name)),
        el("span", { class: `pill ${ {running:"up",loading:"warn",crashed:"down"}[m.state] || "down" }` },
          { running: "● running", loading: "◌ loading…", crashed: "✕ crashed — see logs" }[m.state] || "stopped")),
      el("div", { class: "list" },
        kvRow("File", fname || "—", "down"),
        kvRow("Port", String(m.port), "up"),
        kvRow("Context", `${m.n_ctx} tok`, "up"),
        kvRow("GPU layers", String(m.n_gpu_layers) === "-1" ? "all" : String(m.n_gpu_layers), String(m.n_gpu_layers) !== "0" ? "run" : "down"),
        kvRow("Chat format", m.chat_format || "auto-detect", "down")),
      el("div", { class: "divider" }),
      el("div", { class: "inline-actions" },
        m.running
          ? el("button", { class: "btn sm", onclick: () => modelAction(m.name, "stop") }, "■ Stop")
          : el("button", { class: "btn sm primary", onclick: (e) => modelAction(m.name, "start", e.target) }, "▶ Start"),
        el("button", { class: "btn sm", onclick: () => showLogs(m.name) }, "Logs"),
        el("button", { class: "btn sm", onclick: () => modelDialog(m) }, "Edit"),
        el("span", { class: "list-spacer" }),
        el("button", { class: "btn sm danger", onclick: async () => {
          if (!confirm(`Remove model "${m.name}"? (the GGUF file is not deleted)`)) return;
          try { await api.del(`/api/models/${encodeURIComponent(m.name)}`); ok("Removed", m.name); renderModels(); }
          catch (er) { err(er); }
        } }, "Remove"))));
  });
  c.append(grid);
}

async function modelAction(name, action, btn) {
  if (btn) { btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>`; }
  try {
    await api.post(`/api/models/${encodeURIComponent(name)}/${action}`);
    ok(action === "start" ? "Model starting — watch its Logs" : "Model stopped", name);
  } catch (e) { err(e); }
  renderModels();
  if (action === "start") {  // re-render when it settles into running/crashed
    const probe = setInterval(async () => {
      try {
        const data = await api.get("/api/models");
        const m = data.models.find(x => x.name === name);
        if (!m || m.state !== "loading") { clearInterval(probe); if (state.view === "models") renderModels(); }
      } catch { clearInterval(probe); }
    }, 2500);
    setTimeout(() => clearInterval(probe), 300000);
  }
}

async function showLogs(name) {
  const pre = el("pre", { class: "log-pre" }, "loading…");
  const statusPill = el("span", { class: "pill down" }, "…");
  const refresh = async () => {
    try {
      const r = await api.get(`/api/models/${encodeURIComponent(name)}/logs`);
      pre.textContent = r.log || "(no log output yet)";
      statusPill.className = `pill ${ {running:"up",loading:"warn",crashed:"down"}[r.state] || "down" }`;
      statusPill.textContent = r.state;
      pre.scrollTop = pre.scrollHeight;
    } catch (e) { pre.textContent = String(e.message || e); }
  };
  modal.open({
    title: `Logs · ${name}`,
    body: el("div", {}, el("div", { style: "margin-bottom:10px" }, statusPill), pre),
    footer: [
      el("button", { class: "btn", onclick: refresh }, "⟳ Refresh"),
      el("button", { class: "btn ghost", onclick: () => modal.close() }, "Close"),
    ],
  });
  refresh();
}

function fetchDialog() {
  const repo = textInput("", { placeholder: "bartowski/Llama-3.2-3B-Instruct-GGUF", class: "input mono" });
  const name = textInput("", { placeholder: "(optional — derived from repo)", class: "input mono" });
  const quant = textInput("", { placeholder: "(optional — auto-picked for this machine)", class: "input mono" });
  const prog = el("div", { class: "svc-note", style: "margin-top:10px" }, "");
  modal.open({
    title: "Fetch a model from Hugging Face",
    body: el("div", {},
      el("div", { class: "callout", style: "margin-bottom:16px" }, el("span", { class: "c-ico" }, "⇣"),
        el("div", { html: "Pleiades picks the <b>best quantization that fits this machine</b> (checked against detected VRAM/RAM), downloads it, and registers it with auto GPU offload." })),
      field("Hugging Face repo", repo),
      el("div", { class: "field-row" }, field("Name", name), field("Force quant", quant)),
      prog),
    footer: [
      el("button", { class: "btn ghost", onclick: () => modal.close() }, "Cancel"),
      el("button", { class: "btn primary", onclick: (e) => {
        if (!repo.value.trim()) return toast("Repo required");
        modal.close();
        startFetch(repo.value.trim(), name.value.trim(), quant.value.trim());
      } }, "Fetch"),
    ],
  });
}

const gpuVal = (v) => { const t = String(v).trim().toLowerCase(); return (t === "" || t === "auto") ? "auto" : (parseInt(t, 10) || 0); };

function modelDialog(existing) {
  const isEdit = !!existing;
  const name = textInput(existing ? existing.name : "", { placeholder: "qwen", class: "input mono" });
  if (isEdit) name.setAttribute("readonly", "readonly");
  const path = textInput(existing ? existing.path : "", { placeholder: "~/models/qwen.gguf  or  C:\\models\\qwen.gguf", class: "input mono" });
  const nctx = textInput(existing ? existing.n_ctx : 8192, { type: "number", class: "input mono" });
  const gpu  = textInput(existing ? String(existing.n_gpu_layers) : "auto", { class: "input mono", placeholder: "auto" });
  const fmt  = el("select", { class: "select" },
    ...["", "chatml", "llama-3", "qwen", "mistral-instruct", "gemma"].map(o =>
      el("option", { value: o, selected: existing && existing.chat_format === o ? "selected" : null }, o || "auto-detect")));

  modal.open({
    title: isEdit ? `Edit model · ${existing.name}` : "Register a GGUF model",
    body: el("div", {},
      el("div", { class: "callout", style: "margin-bottom:16px" }, el("span", { class: "c-ico" }, "▤"),
        el("div", { html: "Any GGUF llama.cpp supports works. Use an absolute path (Linux or Windows). <span class='kbd'>GPU layers = auto</span> plans the GPU/CPU split from your hardware at launch; <span class='kbd'>-1</span> forces all layers, <span class='kbd'>0</span> forces CPU." })),
      field("Name", name, isEdit ? "Rename by removing and re-adding." : "A short handle you'll assign to characters."),
      field("Path to .gguf", path),
      el("div", { class: "field-row" },
        field("Context window", nctx), field("GPU layers", gpu)),
      field("Chat format", fmt, "Leave on auto-detect unless tool calls misbehave.")),
    footer: [
      el("button", { class: "btn ghost", onclick: () => modal.close() }, "Cancel"),
      el("button", { class: "btn primary", onclick: async (e) => {
        if (!name.value.trim() || !path.value.trim()) return toast("Name and path required");
        e.target.disabled = true;
        try {
          if (isEdit) {
            await api.put(`/api/models/${encodeURIComponent(existing.name)}`, {
              path: path.value.trim(), n_ctx: +nctx.value || 8192, n_gpu_layers: gpuVal(gpu.value), chat_format: fmt.value });
          } else {
            await api.post("/api/models", {
              name: name.value.trim(), path: path.value.trim(), n_ctx: +nctx.value || 8192,
              n_gpu_layers: gpuVal(gpu.value), chat_format: fmt.value });
          }
          modal.close(); ok(isEdit ? "Model updated" : "Model registered", name.value.trim()); renderModels();
        } catch (er) { e.target.disabled = false; err(er); }
      } }, isEdit ? "Save" : "Register"),
    ],
  });
}

/* ═══════════════ HARDWARE ═══════════════ */
async function renderHardware() {
  loading();
  let hw;
  try { hw = await api.get("/api/hardware"); } catch (e) { return err(e); }
  const c = $("#content"); c.innerHTML = "";

  const meter = (label, used, total, unit = "GiB") => {
    const pct = total ? Math.min(100, Math.round(100 * used / total)) : 0;
    return el("div", { class: "meter-row" },
      el("div", { class: "meter-head" },
        el("span", { class: "list-key" }, label),
        el("span", { class: "val" }, `${GiB(used)} / ${GiB(total)} ${unit} · ${pct}%`)),
      el("div", { class: `meter ${pct > 85 ? "warn" : ""}` }, el("span", { style: `width:${pct}%` })));
  };

  c.append(el("div", { class: "section-label" }, "This machine"));
  const sysCard = el("div", { class: "card pad-lg" });
  if (hw.gpus.length) {
    hw.gpus.forEach(g => {
      sysCard.append(el("div", { class: "inline-actions", style: "margin-bottom:6px" },
        el("span", { class: "pill run" }, g.vendor), el("span", { class: "list-key" }, g.name)));
      sysCard.append(meter("VRAM in use", g.vram_total - g.vram_free, g.vram_total));
    });
  } else {
    sysCard.append(el("div", { class: "callout warn", style: "margin-bottom:16px" }, el("span", { class: "c-ico" }, "▦"),
      el("div", {}, "No GPU detected — models run on CPU. The installer can rebuild llama.cpp with CUDA/ROCm/Metal if a GPU appears.")));
  }
  sysCard.append(meter("RAM in use", hw.ram_total - hw.ram_available, hw.ram_total));
  sysCard.append(el("div", { class: "inline-actions", style: "margin-top:14px" },
    el("span", { class: "badge-soft" }, `${hw.cpu_threads} CPU threads`),
    hw.unified_memory ? el("span", { class: "badge-soft" }, "unified memory") : null));
  c.append(sysCard);

  c.append(el("div", { class: "section-label" }, "Per-model offload plan"));
  if (!hw.plans.length) {
    c.append(el("div", { class: "card" }, el("div", { class: "empty", style: "padding:30px" },
      el("div", { class: "empty-ico" }, "▤"), el("h3", {}, "No models registered"),
      el("div", {}, "Register a model to see how its layers would be split between GPU and CPU."),
      el("div", { style: "margin-top:14px" }, el("button", { class: "btn primary sm", onclick: () => go("models") }, "Go to Models")))));
  } else {
    const grid = el("div", { class: "grid cols-2" });
    hw.plans.forEach(pl => {
      const onGpu = pl.n_gpu_layers === -1 ? pl.n_layers : pl.n_gpu_layers;
      const pct = pl.n_layers ? Math.round(100 * onGpu / pl.n_layers) : 0;
      grid.append(el("div", { class: "card" },
        el("div", { class: "card-head" }, el("h3", {}, pl.model),
          el("span", { class: `pill ${pl.fits_fully ? "up" : onGpu ? "warn" : "down"}` },
            pl.fits_fully ? "fits fully on GPU" : onGpu ? "partial offload" : "CPU only")),
        el("div", { class: "meter-row" },
          el("div", { class: "meter-head" },
            el("span", { class: "list-key" }, "GPU layers"),
            el("span", { class: "val" }, `${onGpu} / ${pl.n_layers} · ${pct}%`)),
          el("div", { class: "meter" }, el("span", { style: `width:${pct}%` }))),
        el("div", { class: "svc-note", style: "white-space:normal", title: pl.reason }, pl.reason)));
    });
    c.append(grid);
  }
}

/* ═══════════════ SETTINGS ═══════════════ */
async function renderSettings() {
  loading();
  let d;
  try { d = await api.get("/api/settings"); } catch (e) { return err(e); }
  const s = d.settings;
  const c = $("#content"); c.innerHTML = "";

  // engine
  const modelPath = textInput(s.model_path, { placeholder: "/path/to/model.gguf", class: "input mono" });
  const infHost = textInput(s.inference_host, { class: "input mono" });
  const infPort = textInput(s.inference_port, { type: "number", class: "input mono" });
  const nctx = textInput(s.n_ctx, { type: "number", class: "input mono" });
  const gpu = textInput(s.n_gpu_layers, { class: "input mono", placeholder: "auto" });
  const fmt = textInput(s.chat_format, { placeholder: "auto-detect", class: "input mono" });

  // services
  const anam = textInput(s.anamnesis_control_url, { class: "input mono" });
  const searx = textInput(s.searxng_url, { class: "input mono" });

  // harness
  const tierSel = el("select", { class: "select" }, ...Object.keys(s.tiers || {}).map(t =>
    el("option", { value: t, selected: t === s.default_tier ? "selected" : null }, t)));
  const policy = el("select", { class: "select" }, ...["ask", "allow", "deny"].map(o =>
    el("option", { value: o, selected: o === s.exec_policy ? "selected" : null }, o)));
  const steps = textInput(s.max_steps, { type: "number", class: "input mono" });

  const save = (collect, btn) => async () => {
    btn.disabled = true;
    try { await api.put("/api/settings", collect()); ok("Settings saved", "Written to .env"); }
    catch (e) { err(e); } finally { btn.disabled = false; }
  };
  const saveBtn = (collect) => { const b = el("button", { class: "btn primary sm" }, "Save"); b.onclick = save(collect, b); return b; };

  // master key status
  const mk = d.master_key;
  c.append(el("div", { class: "card", style: "margin-bottom:22px" },
    el("div", { class: "card-head" }, el("div", { class: "inline-actions" }, el("span", { class: "card-title-ico" }, "🔑"), el("h3", {}, "Vault master key")),
      el("span", { class: `pill ${mk.configured ? "up" : "warn"}` }, mk.configured ? "configured" : "not set")),
    el("div", { class: "callout warn" }, el("span", { class: "c-ico" }, "!"),
      el("div", { html: mk.from_env
        ? "Key supplied via <span class='kbd'>PLEIADES_MASTER_KEY</span> (environment). Losing it loses every stored secret."
        : (mk.keyfile_present
          ? `Generated key file in use: <span class='mono'>${esc(mk.keyfile_path)}</span> (mode 0600). Back it up — losing it loses every stored secret.`
          : "No master key yet. One is generated automatically the first time a vault is written.") }))));

  c.append(el("div", { class: "section-label" }, "Inference engine"));
  c.append(el("div", { class: "card pad-lg" },
    el("div", { class: "callout", style: "margin-bottom:18px" }, el("span", { class: "c-ico" }, "◎"),
      el("div", { html: "Pleiades runs the model itself via in-process llama.cpp. This is the <b>default</b> engine; per-character models (under Models) override it." })),
    field("Default model path", modelPath, "GGUF served when a character has no model assigned."),
    el("div", { class: "field-row" }, field("Inference host", infHost), field("Inference port", infPort)),
    el("div", { class: "field-row tri" },
      field("Context window", nctx), field("GPU layers", gpu, "auto = plan from hardware at launch"), field("Chat format", fmt)),
    el("div", { class: "inline-actions", style: "justify-content:flex-end" },
      saveBtn(() => ({ model_path: modelPath.value.trim(), inference_host: infHost.value.trim(), inference_port: +infPort.value || 8080,
        n_ctx: +nctx.value || 8192, n_gpu_layers: gpuVal(gpu.value), chat_format: fmt.value.trim() })))));

  c.append(el("div", { class: "section-label" }, "Services"));
  c.append(el("div", { class: "card pad-lg" },
    field("Anamnesis control URL", anam, "Memory-proxy control API (default :9000)."),
    field("SearXNG URL", searx, "Local web-search instance the model uses."),
    el("div", { class: "inline-actions", style: "justify-content:flex-end" },
      saveBtn(() => ({ anamnesis_control_url: anam.value.trim(), searxng_url: searx.value.trim() })))));

  c.append(el("div", { class: "section-label" }, "Agent harness"));
  c.append(el("div", { class: "card pad-lg" },
    el("div", { class: "callout", style: "margin-bottom:18px" }, el("span", { class: "c-ico" }, "⚡"),
      el("div", { html: "Controls the workspace harness (the <b>Agent</b> view and <span class='kbd'>pleiades work</span>)." })),
    el("div", { class: "field-row tri" },
      field("Default tier", tierSel), field("Exec policy", policy), field("Max steps", steps)),
    el("div", { class: "inline-actions", style: "justify-content:flex-end" },
      saveBtn(() => ({ default_tier: tierSel.value, exec_policy: policy.value, max_steps: +steps.value || 40 })))));

  c.append(el("div", { class: "section-label" }, "Paths"));
  c.append(el("div", { class: "card" },
    el("div", { class: "list" },
      kvRow("Pleiades home", d.pleiades_home, "down"),
      kvRow(".env file", d.env_file + (d.env_file_exists ? "" : "  (will be created on save)"), d.env_file_exists ? "up" : "warn"))));
}

/* ───────────── service mini status (sidebar) ───────────── */
async function loadStatusMini() {
  try {
    const s = await api.get("/api/status");
    const host = $("#svc-mini"); host.innerHTML = "";
    const row = (label, up) => el("div", { class: "row" }, el("span", { class: `dot ${up ? "up" : "down"}` }), el("span", {}, label));
    host.append(
      row("Memory", s.services.anamnesis.up),
      row("Inference", s.services.inference.up),
      row("Search", s.services.searxng.up));
  } catch (_) {}
}

/* ───────────── download dock (bottom-right, non-blocking) ───────────── */
let dockTimer = null;
function dockHide() { const d = $("#dl-dock"); d.hidden = true; d.innerHTML = ""; d.classList.remove("done"); d.onclick = null; }
function trackFetch() {
  if (dockTimer) return;
  const poll = async () => {
    let st;
    try { st = await api.get("/api/models/fetch/status"); } catch { return; }
    const d = $("#dl-dock");
    if (st.status === "downloading") {
      const pct = st.total ? Math.round(100 * st.done / st.total) : 0;
      d.hidden = false; d.classList.remove("done"); d.onclick = null;
      d.innerHTML = "";
      d.append(
        el("div", { class: "d-title" }, el("span", {}, `⇣ Downloading ${esc(st.repo || "model")}`),
          el("button", { class: "d-x", title: "Hide (download continues)", onclick: (e) => { e.stopPropagation(); dockHide(); } }, "✕")),
        el("div", { class: "d-file" }, `${st.file || "…"}${st.total ? ` — ${pct}% of ${(st.total / 1073741824).toFixed(1)} GiB` : ""}`),
        el("div", { class: "d-bar" }, el("div", { class: "d-fill", style: `width:${pct}%` })));
    } else if (st.status === "done" && st.result) {
      clearInterval(dockTimer); dockTimer = null;
      d.hidden = false; d.classList.add("done");
      d.innerHTML = "";
      d.append(
        el("div", { class: "d-title" }, el("span", {}, `✓ ${esc(st.result.name)} is ready`),
          el("button", { class: "d-x", onclick: (e) => { e.stopPropagation(); dockHide(); } }, "✕")),
        el("div", { class: "d-file" }, `${st.result.chosen} — click to open Models`));
      d.onclick = () => { dockHide(); go("models"); };
    } else if (st.status === "error") {
      clearInterval(dockTimer); dockTimer = null;
      d.hidden = false; d.classList.remove("done");
      d.innerHTML = "";
      d.append(
        el("div", { class: "d-title" }, el("span", {}, "✕ Download failed"),
          el("button", { class: "d-x", onclick: () => dockHide() }, "✕")),
        el("div", { class: "d-file" }, st.error || "unknown error"));
    }
  };
  poll();
  dockTimer = setInterval(poll, 900);
}
async function startFetch(repo, name = "", quant = "") {
  try { await api.post("/api/models/fetch", { repo, name, quant }); }
  catch (e) { return err(e); }
  toast("Download started", repo, "good");
  trackFetch();
}

/* ───────────── self-update ───────────── */
async function checkUpdate() {
  let c;
  try { c = await api.get("/api/update/check"); } catch { return; }
  const host = $("#update-host"); host.innerHTML = "";
  if (!c.supported || !c.behind) return;
  host.append(el("button", { class: "btn primary", id: "update-btn",
    title: `${c.installed} → ${c.latest}`, onclick: runUpdate }, "⬆ Update Pleiades"));
}

async function runUpdate() {
  const btn = $("#update-btn");
  btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> Updating…`;
  try { await api.post("/api/update"); } catch (e) { btn.disabled = false; return err(e); }
  const timer = setInterval(async () => {
    let st;
    try { st = await api.get("/api/update/status"); }
    catch { return waitForRestart(timer); }   // server went down to restart
    if (st.status === "error") {
      clearInterval(timer); btn.disabled = false; btn.textContent = "⬆ Update Pleiades";
      return err(new Error(st.error));
    }
    if (st.status === "up-to-date") { clearInterval(timer); $("#update-host").innerHTML = ""; ok("Already up to date"); }
    if (st.status === "restarting") { waitForRestart(timer); }
    if (st.status === "updating" && st.log && st.log.length) btn.innerHTML = `<span class="spinner"></span> ${st.log[st.log.length - 1]}`;
  }, 900);
}

function waitForRestart(timer) {
  clearInterval(timer);
  toast("Updating", "Pleiades is restarting — this page will reload.");
  const probe = setInterval(async () => {
    try { await api.get("/api/status"); clearInterval(probe); location.reload(); } catch {}
  }, 1200);
}

/* ───────────── boot ───────────── */
loadStatusMini();
go("dashboard");
setInterval(loadStatusMini, 20000);
checkUpdate();
setInterval(checkUpdate, 6 * 60 * 1000);
api.get("/api/models/fetch/status").then(st => { if (st.status === "downloading") trackFetch(); }).catch(() => {});
