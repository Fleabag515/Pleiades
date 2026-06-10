/* ═══════════════════════════════════════════════════════════════
   Pleiades control panel — frontend
   Single-page app over the FastAPI backend. No build step, no deps.
   ═══════════════════════════════════════════════════════════════ */
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
const state = { view: "dashboard", detail: null, cache: {} };

const VIEWS = {
  dashboard:  { title: "Overview",   sub: "Daemons, characters, and models at a glance.", render: renderDashboard },
  characters: { title: "Characters", sub: "Each character is an Anamnesis profile with its own email, vault, browser, and model.", render: renderCharacters },
  models:     { title: "Models",     sub: "Register GGUF models and run a local server for each.", render: renderModels },
  settings:   { title: "Settings",   sub: "Engine, services, and agent-harness configuration.", render: renderSettings },
};

function go(view, detail = null) {
  state.view = view; state.detail = detail;
  $$(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.view === view));
  const v = VIEWS[view];
  $("#view-title").textContent = v.title;
  $("#view-sub").textContent = v.sub;
  v.render();
}
$$(".nav-item").forEach(b => b.addEventListener("click", () => go(b.dataset.view)));
$("#refresh-btn").addEventListener("click", () => { loadStatusMini(); go(state.view, state.detail); });

function loading() { $("#content").innerHTML = `<div class="skeleton"><span class="spinner"></span> Loading…</div>`; }

/* ═══════════════ DASHBOARD ═══════════════ */
async function renderDashboard() {
  loading();
  let s;
  try { s = await api.get("/api/status"); } catch (e) { return err(e); }
  const c = $("#content"); c.innerHTML = "";

  const svc = (name, info, extra) => {
    const up = info.up;
    return el("div", { class: "svc" },
      el("div", { class: "svc-top" },
        el("div", { class: "inline-actions" }, el("span", { class: `dot ${up ? "up" : "down"}` }), el("span", { class: "svc-name" }, name)),
        el("span", { class: `pill ${up ? "up" : "down"}` }, up ? "online" : "offline")),
      el("div", { class: "svc-url" }, info.url || "—"),
      extra ? el("div", { class: "svc-note" }, extra) : null);
  };

  c.append(el("div", { class: "section-label" }, "Services"));
  c.append(el("div", { class: "svc-grid" },
    svc("Anamnesis", s.services.anamnesis, `${s.services.anamnesis.characters} character(s) registered`),
    svc("Inference engine", s.services.inference, s.services.inference.model_path ? `default model: ${s.services.inference.model_path.split(/[\\/]/).pop()}` : "no default model set"),
    svc("SearXNG", s.services.searxng, "local web search")));

  c.append(el("div", { class: "section-label" }, "Hardware"));
  const hwCard = el("div", { class: "card" }, el("div", { class: "skeleton" }, "Detecting hardware…"));
  c.append(hwCard);
  api.get("/api/hardware").then(hw => {
    hwCard.innerHTML = "";
    const list = el("div", { class: "list" });
    (hw.gpus.length ? hw.gpus : [null]).forEach(g => list.append(el("div", { class: "list-row" },
      el("span", { class: `pill ${g ? "run" : "down"}` }, g ? "● GPU" : "no GPU"),
      el("span", { class: "list-key" }, g ? `${g.name} (${g.vendor})` : "running on CPU"),
      el("span", { class: "list-spacer" }),
      g ? el("span", { class: "badge-soft" }, `${(g.vram_free / 1073741824).toFixed(1)} / ${(g.vram_total / 1073741824).toFixed(1)} GiB VRAM free`) : null)));
    list.append(el("div", { class: "list-row" },
      el("span", { class: "badge-soft" }, `RAM ${(hw.ram_available / 1073741824).toFixed(1)} / ${(hw.ram_total / 1073741824).toFixed(1)} GiB available${hw.unified_memory ? " (unified)" : ""}`),
      el("span", { class: "badge-soft" }, `${hw.cpu_threads} CPU threads`)));
    hw.plans.forEach(pl => list.append(el("div", { class: "list-row" },
      el("span", { class: `pill ${pl.n_gpu_layers !== 0 ? "run" : "down"}` },
        pl.n_gpu_layers === -1 ? "all on GPU" : pl.n_gpu_layers === 0 ? "CPU" : `${pl.n_gpu_layers}/${pl.n_layers} on GPU`),
      el("span", { class: "list-key" }, pl.model),
      el("span", { class: "list-spacer" }),
      el("span", { class: "svc-note", title: pl.reason }, pl.reason))));
    hwCard.append(list);
  }).catch(() => { hwCard.innerHTML = ""; hwCard.append(el("div", { class: "svc-note" }, "hardware detection unavailable")); });

  c.append(el("div", { class: "section-label" }, "At a glance"));
  const stat = (val, label, sub) => el("div", { class: "card stat" },
    el("div", { class: "stat-row" }, el("span", { class: "stat-val" }, String(val)), sub ? el("span", { class: "stat-sub" }, sub) : null),
    el("div", { class: "stat-label" }, label));
  c.append(el("div", { class: "grid cols-3" },
    stat(s.counts.profiles, "Characters", s.counts.orphans ? `${s.counts.orphans} adoptable` : ""),
    stat(s.counts.models, "Models registered", `${s.counts.models_running} running`),
    stat(s.counts.models_running, "Live model servers")));

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

  c.append(el("div", { class: "section-label" }, "Quick start"));
  c.append(el("div", { class: "card" },
    el("div", { class: "callout" }, el("span", { class: "c-ico" }, "✶"),
      el("div", { html: `Create a character under <b>Characters</b>, register a GGUF under <b>Models</b>, then assign the model to the character. Every tool — email, vault, browser, Discord — binds to that one character automatically.` }))));
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
        el("div", {}, el("div", { class: "char-name" }, p.name),
          el("div", { class: "char-id" }, p.email_address || "no email configured"))),
      el("div", { class: "char-caps" }, ...caps)));
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
    el("div", { class: "avatar" }, initials(p.name)),
    el("div", {},
      el("h2", { style: "font-size:20px" }, p.name),
      el("div", { class: "muted mono", style: "font-size:12.5px" }, p.email_address || "no email configured")),
    el("div", { class: "list-spacer" }),
    el("button", { class: "btn danger sm", onclick: () => deleteCharacter(p.name) }, "Delete")));

  const tabBar = el("div", { class: "tabs" });
  const pane = el("div", {});
  const TABS = {
    identity:    () => paneIdentity(p, models),
    email:       () => paneEmail(p),
    credentials: () => paneCredentials(p),
    discord:     () => paneDiscord(p),
    model:       () => paneModel(p, models),
  };
  const labels = { identity: "Identity", email: "Email", credentials: "Credentials", discord: "Discord", model: "Model" };
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
  const persona = el("select", { class: "select" },
    ...["auto", "file", "inline", "disabled"].map(o => el("option", { value: o, selected: o === p.persona_source ? "selected" : null }, o)));
  const grid = el("div", { class: "grid cols-2" },
    el("div", { class: "card" },
      el("div", { class: "card-head" }, el("h3", {}, "Profile")),
      field("Persona source", persona, "How Anamnesis sources this character's persona."),
      el("button", { class: "btn primary sm", onclick: async (e) => {
        e.target.disabled = true;
        try { await api.put(`/api/profiles/${p.name}`, { persona_source: persona.value }); ok("Saved"); }
        catch (er) { err(er); } finally { e.target.disabled = false; }
      } }, "Save")),
    el("div", { class: "card" },
      el("div", { class: "card-head" }, el("h3", {}, "Bindings")),
      el("div", { class: "list" },
        kvRow("Email", p.has_email ? p.email_address : "not configured", p.has_email ? "up" : "down"),
        kvRow("Model", p.model || "default engine", p.model ? "run" : "down"),
        kvRow("Discord", p.discord_enabled ? "enabled" : "disabled", p.discord_enabled ? "violet" : "down"),
        kvRow("Vault entries", String((p.vault || []).length), "up"),
        kvRow("Browser profile", "persistent", "up"))));
  return grid;
}
function kvRow(k, v, pill) {
  return el("div", { class: "list-row" },
    el("span", { class: "list-key" }, k),
    el("span", { class: "list-spacer" }),
    el("span", { class: `pill ${pill}` }, v));
}

/* ── Email tab ── */
function paneEmail(p) {
  const email = textInput(p.email_address, { placeholder: "alice@example.com" });
  const imapH = textInput(p.imap_host, { placeholder: "imap.example.com", class: "input mono" });
  const imapP = textInput(p.imap_port, { type: "number", class: "input mono" });
  const smtpH = textInput(p.smtp_host, { placeholder: "smtp.example.com", class: "input mono" });
  const smtpP = textInput(p.smtp_port, { type: "number", class: "input mono" });
  const pass  = textInput("", { type: "password", placeholder: "app password (leave blank to keep current)" });

  const preset = el("select", { class: "select" }, el("option", { value: "" }, "Choose a provider preset…"));
  api.get("/api/email/presets").then(presets => {
    Object.entries(presets).forEach(([k, v]) => preset.append(el("option", { value: k }, k)));
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
        el("span", { class: `pill ${isReserved ? "warn" : (isSite ? "violet" : "up")}` }, isReserved ? "reserved" : (isSite ? "site" : "custom")),
        el("div", {}, el("div", { class: "list-key" }, en.key),
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
  const RESERVED = ["email.password", "email.address", "discord.token"];
  const keyInput = textInput(existing ? existing.key : "", { placeholder: "site:example.com", class: "input mono" });
  if (isEdit) keyInput.setAttribute("readonly", "readonly");
  const valInput = el("textarea", { class: "input mono", rows: "3", placeholder: "secret value" });
  const noteInput = textInput(existing && existing.meta && existing.meta.note ? existing.meta.note : "", { placeholder: "optional label, e.g. GitHub login" });

  const quick = el("div", { class: "inline-actions", style: "flex-wrap:wrap;gap:6px;margin-bottom:8px" });
  if (!isEdit) {
    ["site:", ...RESERVED].forEach(k => quick.append(el("button", { class: "btn sm", onclick: () => { keyInput.value = k; keyInput.focus(); } }, k)));
  }

  modal.open({
    title: isEdit ? `Edit ${existing.key}` : "Add vault entry",
    body: el("div", {},
      el("div", { class: "callout warn", style: "margin-bottom:16px" }, el("span", { class: "c-ico" }, "🔒"),
        el("div", { html: "Reserved keys: <span class='kbd'>email.password</span>, <span class='kbd'>email.address</span>, <span class='kbd'>discord.token</span>. Site logins use the <span class='kbd'>site:&lt;domain&gt;</span> prefix." })),
      isEdit ? null : field("Key", el("div", {}, quick, keyInput)),
      isEdit ? field("Key", keyInput) : null,
      field("Value", valInput, "Encrypted with Fernet before it touches disk."),
      field("Note (optional)", noteInput)),
    footer: [
      el("button", { class: "btn ghost", onclick: () => modal.close() }, "Cancel"),
      el("button", { class: "btn primary", onclick: async (e) => {
        if (!keyInput.value.trim()) return toast("Key required");
        if (!valInput.value) return toast("Value required");
        e.target.disabled = true;
        try {
          await api.post(`/api/profiles/${name}/vault`, { key: keyInput.value.trim(), value: valInput.value, note: noteInput.value.trim() });
          modal.close(); ok(isEdit ? "Updated" : "Saved", keyInput.value.trim()); go("characters", name);
        } catch (er) { e.target.disabled = false; err(er); }
      } }, isEdit ? "Update" : "Save"),
    ],
  });
}

/* ── Discord tab ── */
function paneDiscord(p) {
  const toggle = switchToggle(p.discord_enabled, "Host this character as a Discord bot");
  const token = textInput("", { type: "password", placeholder: "bot token (leave blank to keep current)", class: "input mono" });
  return el("div", { class: "card pad-lg", style: "max-width:660px" },
    el("div", { class: "card-head" }, el("h3", {}, "Discord bot"),
      el("span", { class: `pill ${p.discord_enabled ? "violet" : "down"}` }, p.discord_enabled ? "enabled" : "disabled")),
    el("div", { class: "callout", style: "margin-bottom:20px" }, el("span", { class: "c-ico" }, "◈"),
      el("div", { html: "Create a bot in the Discord Developer Portal and <b>enable the Message Content intent</b>. The token is stored encrypted as <span class='kbd'>discord.token</span>. Run it with <span class='kbd'>pleiades discord " + esc(p.name) + "</span>." })),
    el("div", { style: "margin-bottom:18px" }, toggle),
    field("Bot token", token, "Stored encrypted in this character's vault."),
    el("button", { class: "btn primary", onclick: async (e) => {
      e.target.disabled = true;
      try {
        await api.post(`/api/profiles/${p.name}/discord`, { token: token.value || null, enabled: $("input", toggle).checked });
        ok("Discord saved"); go("characters", p.name);
      } catch (er) { e.target.disabled = false; err(er); }
    } }, "Save Discord settings"));
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

  c.append(el("div", { class: "inline-actions", style: "justify-content:flex-end;margin-bottom:18px" },
    el("button", { class: "btn", onclick: () => fetchDialog() }, "⇣ Fetch from Hugging Face"),
    el("button", { class: "btn primary", onclick: () => modelDialog() }, "＋ Add model")));

  if (!data.models.length) {
    c.append(el("div", { class: "empty" }, el("div", { class: "empty-ico" }, "▤"),
      el("h3", {}, "No models registered"),
      el("div", {}, "Register a GGUF file to run it as a local OpenAI-compatible server."),
      el("div", { style: "margin-top:18px" }, el("button", { class: "btn primary", onclick: () => modelDialog() }, "＋ Add model"))));
    return;
  }

  const grid = el("div", { class: "grid cols-2" });
  data.models.forEach(m => {
    const fname = (m.path || "").split(/[\\/]/).pop();
    grid.append(el("div", { class: "card" },
      el("div", { class: "card-head" },
        el("div", { class: "inline-actions" }, el("span", { class: "card-title-ico" }, "▤"), el("h3", {}, m.name)),
        el("span", { class: `pill ${m.running ? "run" : "down"}` }, m.running ? "● running" : "stopped")),
      el("div", { class: "list" },
        kvRow("File", fname || "—", "down"),
        kvRow("Port", String(m.port), "up"),
        kvRow("Context", `${m.n_ctx} tok`, "up"),
        kvRow("GPU layers", m.n_gpu_layers === -1 ? "all" : String(m.n_gpu_layers), m.n_gpu_layers !== 0 ? "run" : "down"),
        kvRow("Chat format", m.chat_format || "auto-detect", "down")),
      el("div", { class: "divider" }),
      el("div", { class: "inline-actions" },
        m.running
          ? el("button", { class: "btn sm", onclick: () => modelAction(m.name, "stop") }, "■ Stop")
          : el("button", { class: "btn sm primary", onclick: (e) => modelAction(m.name, "start", e.target) }, "▶ Start"),
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
    ok(action === "start" ? "Model started" : "Model stopped", name);
  } catch (e) { err(e); }
  renderModels();
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
      el("button", { class: "btn primary", onclick: async (e) => {
        if (!repo.value.trim()) return toast("Repo required");
        e.target.disabled = true;
        try { await api.post("/api/models/fetch", { repo: repo.value.trim(), name: name.value.trim(), quant: quant.value.trim() }); }
        catch (er) { e.target.disabled = false; return err(er); }
        const timer = setInterval(async () => {
          let st;
          try { st = await api.get("/api/models/fetch/status"); } catch { return; }
          if (st.status === "downloading") {
            const pct = st.total ? ` ${Math.round(100 * st.done / st.total)}%` : "";
            prog.textContent = `Downloading ${st.file || "…"}${pct}`;
          } else if (st.status === "done") {
            clearInterval(timer); modal.close();
            ok("Model fetched", `${st.result.name} — ${st.result.chosen}`); renderModels();
          } else if (st.status === "error") {
            clearInterval(timer); e.target.disabled = false;
            prog.textContent = ""; err(new Error(st.error));
          }
        }, 700);
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
  const gpu = textInput(s.n_gpu_layers, { type: "number", class: "input mono" });
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
      field("Context window", nctx), field("GPU layers", gpu), field("Chat format", fmt)),
    el("div", { class: "inline-actions", style: "justify-content:flex-end" },
      saveBtn(() => ({ model_path: modelPath.value.trim(), inference_host: infHost.value.trim(), inference_port: +infPort.value || 8080,
        n_ctx: +nctx.value || 8192, n_gpu_layers: +gpu.value, chat_format: fmt.value.trim() })))));

  c.append(el("div", { class: "section-label" }, "Services"));
  c.append(el("div", { class: "card pad-lg" },
    field("Anamnesis control URL", anam, "Memory-proxy control API (default :9000)."),
    field("SearXNG URL", searx, "Local web-search instance the model uses."),
    el("div", { class: "inline-actions", style: "justify-content:flex-end" },
      saveBtn(() => ({ anamnesis_control_url: anam.value.trim(), searxng_url: searx.value.trim() })))));

  c.append(el("div", { class: "section-label" }, "Agent harness"));
  c.append(el("div", { class: "card pad-lg" },
    el("div", { class: "callout", style: "margin-bottom:18px" }, el("span", { class: "c-ico" }, "⚙"),
      el("div", { html: "Controls the Claude-Code-style workspace harness (<span class='kbd'>pleiades work</span>)." })),
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
      row("Anamnesis", s.services.anamnesis.up),
      row("Inference", s.services.inference.up),
      row("SearXNG", s.services.searxng.up));
  } catch (_) {}
}

/* ───────────── boot ───────────── */
loadStatusMini();
go("dashboard");
setInterval(loadStatusMini, 20000);
