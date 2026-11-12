/* Manuscript Cleaner – page behaviour. Plain JavaScript, no libraries. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = { file: null, sample: null, result: null, tab: "ai", health: null, busy: false, thumbUrl: null };
  const TAB_NAMES = { ai: "AI cleaned", otsu: "Simple filter", sauvola: "Local filter", truth: "Expert answer" };
  const show = (el, on) => { el.hidden = !on; };
  const fmtBytes = (n) => (n > 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB");
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

  async function getJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(url + " " + res.status);
    return res.json();
  }

  /* ------------------------------------------------------------------ server status */
  async function loadHealth() {
    const pill = $("status");
    try {
      const h = await getJSON("/api/health");
      state.health = h;
      if (h.model_loaded) {
        pill.textContent = h.device === "cuda" ? "Ready · " + (h.gpu_name || "GPU") : "Ready · CPU (slower)";
        pill.className = "pill ok";
        if (h.device_note) { pill.title = h.device_note; }
      } else {
        pill.textContent = "Model not loaded";
        pill.className = "pill bad";
        const box = $("modelWarning");
        box.textContent = h.error || "The model is not loaded.";
        show(box, true);
      }
      $("maxMb").textContent = h.max_upload_mb;
      const ocr = $("ocrBox");
      if (!h.ocr_available) {
        ocr.disabled = true;
        $("ocrHint").textContent = "(not set up on this computer)";
      }
    } catch (err) {
      pill.textContent = "Server not reachable";
      pill.className = "pill bad";
      state.health = { model_loaded: false };
    }
    updateRunButton();
  }

  function updateRunButton() {
    $("runBtn").disabled = !(state.file && !state.busy && state.health && state.health.model_loaded);
  }

  /* ------------------------------------------------------------------ results charts */
  function drawBars(container, items, opts) {
    container.textContent = "";
    items.forEach((it) => {
      const row = el("div", "bar-row");
      row.appendChild(el("span", "name", it.name));
      const track = el("div", "track");
      const fill = el("div", "fill" + (it.key === "ai" ? " ai" : ""));
      fill.style.width = Math.max(1, (it.value / opts.max) * 100) + "%";
      track.appendChild(fill);
      if (opts.range && it.low != null) {
        const r = el("div", "range");
        r.style.left = (it.low / opts.max) * 100 + "%";
        r.style.width = ((it.high - it.low) / opts.max) * 100 + "%";
        track.appendChild(r);
      }
      row.appendChild(track);
      row.appendChild(el("span", "val", opts.fmt(it.value)));
      container.appendChild(row);
    });
  }

  async function loadResults() {
    try {
      const r = await getJSON("/api/results");
      if (!r.page_f1) return;
      $("pagesTested").textContent = r.pages_tested;
      const ai = r.page_f1.find((x) => x.key === "ai");
      if (ai) $("bigF1").textContent = ai.value.toFixed(2);
      $("bigBeat").textContent = r.beats_sauvola_pages + " of " + r.pages_tested;
      if (r.ocr) {
        const o = r.ocr.items.find((x) => x.key === "orig"), a = r.ocr.items.find((x) => x.key === "ai");
        if (o && a) $("bigOcr").textContent = Math.round(o.value * 100) + "% → " + Math.round(a.value * 100) + "%";
        drawBars($("chartOcr"), r.ocr.items, { max: 0.7, fmt: (v) => Math.round(v * 100) + "%" });
      }
      drawBars($("chartF1"), r.page_f1, { max: 1, fmt: (v) => v.toFixed(2), range: true });
    } catch (err) { /* the written text on the page still works without the charts */ }
  }

  /* ------------------------------------------------------------------ samples */
  async function loadSamples() {
    try {
      const list = await getJSON("/api/samples");
      if (!list.length) return;
      const box = $("samples");
      list.forEach((s) => {
        const b = el("button", "sample");
        b.type = "button"; b.title = s.title; b.setAttribute("aria-label", "Use sample: " + s.title);
        const img = el("img"); img.src = s.image; img.alt = s.title; img.loading = "lazy";
        b.appendChild(img);
        b.addEventListener("click", () => pickSample(s, b));
        box.appendChild(b);
      });
      $("sampleNote").textContent = list[0].note || "";
      show($("samplesBox"), true);
    } catch (err) { /* no samples available: uploading still works */ }
  }

  async function pickSample(sample, button) {
    try {
      const res = await fetch(sample.image);
      const blob = await res.blob();
      const name = sample.image.split("/").pop();
      setFile(new File([blob], name, { type: blob.type || "image/jpeg" }), sample);
      document.querySelectorAll(".sample").forEach((b) => b.classList.toggle("on", b === button));
    } catch (err) { showError("The sample could not be loaded."); }
  }

  /* ------------------------------------------------------------------ choosing a file */
  function setFile(file, sample) {
    const maxMb = (state.health && state.health.max_upload_mb) || 20;
    if (!file) return;
    if (file.size > maxMb * 1048576) { showError("This file is larger than " + maxMb + " MB. Please choose a smaller picture."); return; }
    if (file.type && !file.type.startsWith("image/")) { showError("Please choose a picture file (PNG, JPG, BMP, TIFF or WEBP)."); return; }
    showError(null);
    state.file = file; state.sample = sample || null;
    if (!sample) document.querySelectorAll(".sample").forEach((b) => b.classList.remove("on"));
    if (state.thumbUrl) URL.revokeObjectURL(state.thumbUrl);
    state.thumbUrl = URL.createObjectURL(file);
    const thumb = $("pickedThumb");
    thumb.onload = () => { $("pickedMeta").textContent = fmtBytes(file.size) + " · " + thumb.naturalWidth + "×" + thumb.naturalHeight + " px"; };
    thumb.src = state.thumbUrl;
    $("pickedName").textContent = file.name || "pasted-image.png";
    $("pickedMeta").textContent = fmtBytes(file.size);
    show($("picked"), true);
    updateRunButton();
  }

  function clearFile() {
    state.file = null; state.sample = null;
    if (state.thumbUrl) { URL.revokeObjectURL(state.thumbUrl); state.thumbUrl = null; }
    $("fileInput").value = "";
    document.querySelectorAll(".sample").forEach((b) => b.classList.remove("on"));
    show($("picked"), false);
    updateRunButton();
  }

  function showError(message) {
    const box = $("error");
    if (message) { box.textContent = message; show(box, true); show($("empty"), false); }
    else { show(box, false); }
  }

  /* ------------------------------------------------------------------ cleaning */
  async function runClean() {
    if (!state.file || state.busy) return;
    state.busy = true; updateRunButton();
    showError(null); show($("result"), false); show($("empty"), false); show($("busy"), true);
    $("busyText").textContent = "Cleaning… large pages can take a little while.";
    const slow = setTimeout(() => { $("busyText").textContent = "Still working… big pages take longer, especially without a graphics card."; }, 8000);
    try {
      const form = new FormData();
      form.append("file", state.file, state.file.name || "page.png");
      form.append("ocr", $("ocrBox").checked && !$("ocrBox").disabled ? "true" : "false");
      const res = await fetch("/api/clean", { method: "POST", body: form });
      if (!res.ok) {
        let msg = "Something went wrong (" + res.status + ").";
        try {
          const body = await res.json();
          if (typeof body.detail === "string") msg = body.detail;
          else if (Array.isArray(body.detail) && body.detail[0] && body.detail[0].msg) msg = body.detail[0].msg;
        } catch (e) { /* keep the generic message */ }
        throw new Error(msg);
      }
      render(await res.json());
    } catch (err) {
      showError(err instanceof TypeError ? "Could not reach the server. Is the app still running?" : err.message);
    } finally {
      clearTimeout(slow);
      state.busy = false; show($("busy"), false); updateRunButton();
    }
  }

  function afterSrc(tab) {
    if (tab === "truth") return state.sample && state.sample.truth;
    return state.result && state.result.images[tab];
  }

  function setTab(tab) {
    const src = afterSrc(tab);
    if (!src) return;
    state.tab = tab;
    document.querySelectorAll(".tab").forEach((b) => {
      const on = b.dataset.tab === tab;
      b.classList.toggle("on", on); b.setAttribute("aria-selected", on ? "true" : "false");
    });
    $("cmpAfter").src = src;
    $("afterTag").textContent = TAB_NAMES[tab];
  }

  function statBox(value, label) {
    const box = el("div", "stat-box");
    box.appendChild(el("b", null, value)); box.appendChild(el("span", null, label));
    return box;
  }

  function render(r) {
    state.result = r;
    $("cmpBefore").src = r.images.original;
    show($("truthTab"), !!(state.sample && state.sample.truth));
    setTab("ai");

    const notes = $("notes"); notes.textContent = "";
    (r.notes || []).forEach((n) => notes.appendChild(el("div", "alert info", n)));

    const s = r.stats, grid = $("statGrid"); grid.textContent = "";
    grid.appendChild(statBox(s.ink_ai_pct + "%", "of the page marked as ink by the AI"));
    grid.appendChild(statBox(s.ink_otsu_pct + "%", "a simple filter would mark"));
    grid.appendChild(statBox(s.ink_sauvola_pct + "%", "a local filter would mark"));
    grid.appendChild(statBox(s.seconds_ai + " s", "AI time on " + (s.device === "cuda" ? "the GPU" : "the CPU")));
    grid.appendChild(statBox(r.width + "×" + r.height, "pixels"));

    const v = $("verdict");
    const ratio = s.ink_ai_pct > 0 ? s.ink_otsu_pct / s.ink_ai_pct : 0;
    if (s.ink_ai_pct === 0) v.textContent = "The AI did not find any ink on this picture.";
    else if (ratio >= 1.3) v.textContent = "A simple filter marked about " + ratio.toFixed(1) + " times more of the page as ink than the AI did. The extra black is mostly stains, shadows and paper texture that the AI recognised as not being writing.";
    else if (ratio <= 0.8) v.textContent = "The AI found more ink than the simple filter, probably faint writing that a single brightness cut-off misses.";
    else v.textContent = "All methods found a similar amount of ink here, so this page is probably in fairly good condition.";

    const dl = $("dlBtn");
    dl.href = r.images.ai;
    dl.download = ((state.file && state.file.name) || "page").replace(/\.[^.]+$/, "") + "_cleaned.png";

    const ocrBox = $("ocrOut");
    if (r.ocr && r.ocr.available) {
      $("ocrOrig").textContent = r.ocr.original || "(no text found)";
      $("ocrClean").textContent = r.ocr.cleaned || "(no text found)";
      show(ocrBox, true);
    } else {
      show(ocrBox, false);
      if (r.ocr && !r.ocr.available) notes.appendChild(el("div", "alert info", "Text reading (OCR) is not set up on this computer, so it was skipped."));
    }
    show($("result"), true);
    $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  /* ------------------------------------------------------------------ wiring */
  function init() {
    const drop = $("drop"), input = $("fileInput");
    drop.addEventListener("click", () => input.click());
    drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
    input.addEventListener("change", () => { if (input.files[0]) setFile(input.files[0], null); });
    ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => { const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) setFile(f, null); });
    document.addEventListener("paste", (e) => {
      const item = [...(e.clipboardData ? e.clipboardData.items : [])].find((i) => i.type.startsWith("image/"));
      if (item) setFile(item.getAsFile(), null);
    });
    $("clearBtn").addEventListener("click", clearFile);
    $("runBtn").addEventListener("click", runClean);
    document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
    $("cmpRange").addEventListener("input", (e) => $("compare").style.setProperty("--pos", e.target.value + "%"));
    $("copyBtn").addEventListener("click", async () => {
      try { await navigator.clipboard.writeText($("ocrClean").textContent); $("copyBtn").textContent = "Copied!"; setTimeout(() => { $("copyBtn").textContent = "Copy this text"; }, 1500); }
      catch (e) { $("copyBtn").textContent = "Select the text and press Ctrl+C"; }
    });
    loadHealth(); loadResults(); loadSamples();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
