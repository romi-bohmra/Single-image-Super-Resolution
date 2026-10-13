(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const fileInput = $("fileInput"), dropzone = $("dropzone"), dropText = $("dropText"),
        enhanceBtn = $("enhanceBtn"), statusMsg = $("statusMsg"), viewer = $("viewer"),
        chips = $("chips"), bigStage = $("bigStage"), bigImg = $("bigImg"), lens = $("lens"),
        badge = $("stageBadge"), zoomSlider = $("zoomSlider"), zoomValue = $("zoomValue"),
        compareSelect = $("compareSelect"), canvasA = $("canvasA"), canvasB = $("canvasB"),
        capA = $("capA"), capB = $("capB"), downloadLink = $("downloadLink");

  let selectedFile = null;
  let results = {};          // key -> {label, url, img (HTMLImageElement)}
  let order = [];
  let current = null;        // model shown in the big image / right zoom
  let compare = "bicubic";   // model shown in the left zoom
  let center = { x: 0.5, y: 0.5 };  // lens centre as a fraction of the image
  let zoom = 4;              // on-screen magnification of the zoomed views

  const setStatus = (t, k) => { statusMsg.textContent = t || ""; statusMsg.className = "status" + (k ? " " + k : ""); };

  // ---------- upload ----------
  fileInput.addEventListener("change", () => fileInput.files[0] && pick(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((e) => dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((e) => dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.remove("dragover"); }));
  dropzone.addEventListener("drop", (e) => e.dataTransfer.files[0] && pick(e.dataTransfer.files[0]));

  function pick(file) {
    if (!file.type.startsWith("image/")) return setStatus("Please choose an image file.", "error");
    selectedFile = file; enhanceBtn.disabled = false;
    dropText.textContent = file.name; setStatus("");
  }

  enhanceBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    enhanceBtn.disabled = true; setStatus("Running the models… this can take a few seconds on CPU.");
    const fd = new FormData(); fd.append("file", selectedFile);
    try {
      const res = await fetch("/api/enhance", { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Request failed (${res.status})`);
      const data = await res.json();
      results = {}; order = [];
      await Promise.all(data.results.map(async (r) => {
        const img = new Image(); img.src = r.url; await img.decode();
        results[r.key] = { label: r.label, url: r.url, img };
      }));
      order = data.results.map((r) => r.key);
      const [iw, ih] = data.input_size, [ow, oh] = data.output_size;
      badge.dataset.info = `${iw}×${ih} → ${ow}×${oh}`;
      current = order.includes("srgan") ? "srgan" : order[order.length - 1];
      compare = "bicubic";
      center = { x: 0.5, y: 0.5 };
      buildControls();
      viewer.hidden = false;
      render();
      viewer.scrollIntoView({ behavior: "smooth", block: "start" });
      setStatus("Done.", "success");
    } catch (err) {
      console.error(err); setStatus(err.message || "Something went wrong.", "error");
    } finally { enhanceBtn.disabled = false; }
  });

  // ---------- controls ----------
  function buildControls() {
    chips.innerHTML = "";
    order.forEach((k) => {
      const b = document.createElement("button");
      b.className = "chip"; b.textContent = results[k].label; b.dataset.key = k; b.setAttribute("role", "tab");
      b.onclick = () => { current = k; render(); };
      chips.appendChild(b);
    });
    compareSelect.innerHTML = "";
    order.forEach((k) => {
      const o = document.createElement("option"); o.value = k; o.textContent = results[k].label; compareSelect.appendChild(o);
    });
    compareSelect.value = compare;
  }
  compareSelect.onchange = () => { compare = compareSelect.value; render(); };
  zoomSlider.oninput = () => setZoom(parseFloat(zoomSlider.value));
  function setZoom(z) {
    zoom = Math.max(1, Math.min(12, z)); zoomSlider.value = zoom; zoomValue.textContent = zoom.toFixed(1) + "×"; drawZoom();
  }

  // ---------- rendering ----------
  function render() {
    chips.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-selected", c.dataset.key === current));
    const r = results[current];
    bigImg.src = r.url; downloadLink.href = r.url;
    downloadLink.download = `sr_${current}.png`;
    badge.textContent = `${r.label} · ${badge.dataset.info}`;
    capA.textContent = results[compare].label; capB.textContent = r.label;
    if (bigImg.complete) updateLens(); else bigImg.onload = updateLens;
    drawZoom();
  }

  // Size of the region (in output-image pixels) shown by each zoom canvas.
  function regionSize() {
    const w = canvasA.clientWidth || 400;
    return { w: w / zoom, h: w / zoom };
  }

  function clampCenter(img) {
    const { w, h } = regionSize();
    const nx = img.naturalWidth, ny = img.naturalHeight;
    const hx = Math.min(0.5, w / 2 / nx), hy = Math.min(0.5, h / 2 / ny);
    center.x = Math.max(hx, Math.min(1 - hx, center.x));
    center.y = Math.max(hy, Math.min(1 - hy, center.y));
  }

  function updateLens() {
    const img = results[current].img;
    clampCenter(img);
    const { w, h } = regionSize();
    const sw = bigStage.clientWidth, sh = bigStage.clientHeight;
    const lw = Math.min(sw, (w / img.naturalWidth) * sw), lh = Math.min(sh, (h / img.naturalHeight) * sh);
    lens.style.width = lw + "px"; lens.style.height = lh + "px";
    lens.style.left = center.x * sw - lw / 2 + "px"; lens.style.top = center.y * sh - lh / 2 + "px";
  }

  function drawOne(canvas, img) {
    const dpr = window.devicePixelRatio || 1;
    const size = Math.round(canvas.clientWidth * dpr);
    if (canvas.width !== size) { canvas.width = size; canvas.height = size; }
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    const { w, h } = regionSize();
    const sx = center.x * img.naturalWidth - w / 2, sy = center.y * img.naturalHeight - h / 2;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, sx, sy, w, h, 0, 0, canvas.width, canvas.height);
  }

  function drawZoom() {
    if (!current) return;
    clampCenter(results[current].img);
    drawOne(canvasA, results[compare].img);
    drawOne(canvasB, results[current].img);
    updateLens();
  }

  // ---------- interaction ----------
  // 1) drag on the big image: lens follows the pointer
  let draggingBig = false;
  const moveToPointer = (e) => {
    const r = bigStage.getBoundingClientRect();
    center = { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
    drawZoom();
  };
  bigStage.addEventListener("pointerdown", (e) => { draggingBig = true; bigStage.setPointerCapture(e.pointerId); moveToPointer(e); });
  bigStage.addEventListener("pointermove", (e) => draggingBig && moveToPointer(e));
  ["pointerup", "pointercancel"].forEach((t) => bigStage.addEventListener(t, () => (draggingBig = false)));

  // 2) drag inside a zoomed view: pan; wheel: zoom
  [canvasA, canvasB].forEach((cv) => {
    let last = null;
    cv.addEventListener("pointerdown", (e) => { last = { x: e.clientX, y: e.clientY }; cv.setPointerCapture(e.pointerId); cv.classList.add("panning"); });
    cv.addEventListener("pointermove", (e) => {
      if (!last) return;
      const img = results[current].img;
      // one CSS pixel on the zoom view = 1/zoom output pixels
      center.x -= (e.clientX - last.x) / zoom / img.naturalWidth;
      center.y -= (e.clientY - last.y) / zoom / img.naturalHeight;
      last = { x: e.clientX, y: e.clientY };
      drawZoom();
    });
    const end = () => { last = null; cv.classList.remove("panning"); };
    cv.addEventListener("pointerup", end); cv.addEventListener("pointercancel", end);
    cv.addEventListener("wheel", (e) => { e.preventDefault(); setZoom(zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15)); }, { passive: false });
  });

  window.addEventListener("resize", () => current && drawZoom());
})();
