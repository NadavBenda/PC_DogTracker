(() => {
  "use strict";

  const state = {
    summary: null,
    frames: [],
    detections: [],
    detectionByFilename: new Map(),
    frameIndexByFilename: new Map(),
    visits: [],
    currentIndex: 0,
  };

  const el = (id) => document.getElementById(id);

  async function fetchJSON(url) {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`${url} -> ${res.status}`);
    }
    return res.json();
  }

  function formatElapsed(ms) {
    const totalSeconds = Math.max(0, Math.round(ms / 1000));
    const hh = Math.floor(totalSeconds / 3600);
    const mm = Math.floor((totalSeconds % 3600) / 60);
    const ss = totalSeconds % 60;
    return [hh, mm, ss].map((v) => String(v).padStart(2, "0")).join(":");
  }

  function statTile(label, value, opts = {}) {
    const tile = document.createElement("div");
    tile.className = "stat-tile";
    const labelEl = document.createElement("div");
    labelEl.className = "label";
    labelEl.textContent = label;
    const valueEl = document.createElement("div");
    valueEl.className = "value" + (opts.empty ? " empty" : "");
    valueEl.textContent = value;
    tile.append(labelEl, valueEl);
    return tile;
  }

  function renderKPIs() {
    const { summary, visits, detections } = state;
    const row = el("kpiRow");
    row.replaceChildren(
      statTile("Frames processed", summary.frame_count.toLocaleString()),
      statTile("Dog detections", detections.length.toLocaleString(), { empty: detections.length === 0 }),
      statTile("Visits", visits.length.toLocaleString()),
      statTile("Session duration", formatElapsed(summary.duration_ms))
    );
  }

  function elapsedOf(timestampMs) {
    return timestampMs - state.summary.first_timestamp_ms;
  }

  // Visits/areas are computed server-side purely over the detections array,
  // so their indices (first_index, mid_index, representative_index, ...)
  // refer to positions in state.detections -- not to state.frames, which
  // the frame browser now scrubs over in full. This resolves one to the
  // other via filename, the only thing both sides agree on.
  function frameIndexForDetectionIndex(detectionIndex) {
    const filename = state.detections[detectionIndex].filename;
    return state.frameIndexByFilename.get(filename) ?? 0;
  }

  function renderVisitsTable() {
    const body = el("visitsBody");
    body.replaceChildren();
    state.visits.forEach((visit, i) => {
      const row = document.createElement("tr");
      const midFrameIndex = frameIndexForDetectionIndex(visit.mid_index);
      row.dataset.firstIndex = String(frameIndexForDetectionIndex(visit.first_index));
      row.dataset.lastIndex = String(frameIndexForDetectionIndex(visit.last_index));
      row.dataset.midIndex = String(midFrameIndex);

      const cells = [
        String(i + 1),
        formatElapsed(elapsedOf(visit.start_ts)),
        formatElapsed(visit.duration_ms),
        String(visit.frame_count),
        `${Math.round(visit.centroid_x)}, ${Math.round(visit.centroid_y)}`,
      ];
      for (const text of cells) {
        const td = document.createElement("td");
        td.textContent = text;
        row.appendChild(td);
      }
      row.addEventListener("click", () => setCurrentIndex(midFrameIndex));
      body.appendChild(row);
    });
    updateActiveVisitRow();
  }

  function updateActiveVisitRow() {
    const body = el("visitsBody");
    const current = state.currentIndex;
    for (const row of body.children) {
      const first = Number(row.dataset.firstIndex);
      const last = Number(row.dataset.lastIndex);
      row.classList.toggle("active", current >= first && current <= last);
    }
  }

  function distanceSquared(ax, ay, bx, by) {
    return (ax - bx) ** 2 + (ay - by) ** 2;
  }

  function nearestDetectionIndex(x, y) {
    const dets = state.detections;
    let bestIndex = 0;
    let bestDist = Infinity;
    for (let i = 0; i < dets.length; i++) {
      const d = distanceSquared(dets[i].x, dets[i].y, x, y);
      if (d < bestDist) {
        bestDist = d;
        bestIndex = i;
      }
    }
    return { index: bestIndex, distance: Math.sqrt(bestDist) };
  }

  function setCurrentIndex(i) {
    const frames = state.frames;
    if (frames.length === 0) return;
    const clamped = Math.max(0, Math.min(frames.length - 1, i));
    state.currentIndex = clamped;
    const frame = frames[clamped];
    const det = state.detectionByFilename.get(frame.filename);

    const frameImg = el("frameImg");
    frameImg.src = `/frames/${encodeURIComponent(frame.filename)}`;
    if (det) {
      frameImg.dataset.hasDetection = "1";
      frameImg.dataset.w = det.w;
      frameImg.dataset.h = det.h;
      frameImg.dataset.x = det.x;
      frameImg.dataset.y = det.y;
    } else {
      frameImg.dataset.hasDetection = "0";
    }

    el("frameFilename").textContent = frame.filename;
    el("frameTimestamp").textContent = formatElapsed(elapsedOf(frame.timestamp_ms));
    el("frameConfidence").textContent = det ? `${Math.round(det.confidence * 100)}% confidence` : "No detection";

    el("frameSlider").value = String(clamped);
    el("prevBtn").disabled = clamped === 0;
    el("nextBtn").disabled = clamped === frames.length - 1;

    if (det) {
      positionCurrentMarker(det);
    } else {
      hideCurrentMarker();
    }
    updateRulerMarker();
    updateActiveVisitRow();
  }

  function ensureCurrentMarker() {
    let marker = el("currentMarker");
    if (!marker) {
      marker = document.createElement("div");
      marker.id = "currentMarker";
      marker.style.position = "absolute";
      marker.style.width = "10px";
      marker.style.height = "10px";
      marker.style.marginLeft = "-5px";
      marker.style.marginTop = "-5px";
      marker.style.borderRadius = "50%";
      marker.style.background = "var(--accent)";
      marker.style.boxShadow = "0 0 0 2px var(--surface-1)";
      marker.style.pointerEvents = "none";
      el("heatmapStage").appendChild(marker);
    }
    return marker;
  }

  function positionCurrentMarker(det) {
    const marker = ensureCurrentMarker();
    marker.style.display = "block";
    const { frame_width, frame_height } = state.summary;
    marker.style.left = `${(det.x / frame_width) * 100}%`;
    marker.style.top = `${(det.y / frame_height) * 100}%`;
  }

  function hideCurrentMarker() {
    ensureCurrentMarker().style.display = "none";
  }

  function drawBoundingBox() {
    const frameImg = el("frameImg");
    const canvas = el("bboxCanvas");
    const w = frameImg.naturalWidth;
    const h = frameImg.naturalHeight;
    if (!w || !h) return;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);

    if (frameImg.dataset.hasDetection !== "1") {
      return; // no dog detected in this frame -- nothing to draw
    }

    const bx = Number(frameImg.dataset.x);
    const by = Number(frameImg.dataset.y);
    const bw = Number(frameImg.dataset.w);
    const bh = Number(frameImg.dataset.h);
    ctx.strokeStyle = "#2a78d6";
    ctx.lineWidth = Math.max(2, Math.round(w / 200));
    ctx.strokeRect(bx - bw / 2, by - bh / 2, bw, bh);
  }

  // ======================================================
  // Detection-presence ruler: a clickable timeline showing which parts of
  // the whole session had a dog detected (colored) vs not (empty), plus a
  // marker for the currently-browsed frame. The coloring only needs
  // recomputing when the data loads or the ruler resizes; the marker moves
  // far more often, so it's a separate cheap overlay instead of a redraw.
  // ======================================================
  function renderDetectionRulerBackground() {
    const canvas = el("detectionRuler");
    const width = Math.max(Math.round(canvas.getBoundingClientRect().width), 1);
    const height = 22;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");

    const styles = getComputedStyle(document.body);
    const emptyColor = styles.getPropertyValue("--gridline").trim() || "#e1e0d9";
    const accentColor = styles.getPropertyValue("--accent").trim() || "#2a78d6";

    ctx.fillStyle = emptyColor;
    ctx.fillRect(0, 0, width, height);

    const total = state.frames.length;
    if (total === 0) return;

    ctx.fillStyle = accentColor;
    for (let x = 0; x < width; x++) {
      const startIdx = Math.floor((x / width) * total);
      const endIdx = Math.max(startIdx + 1, Math.floor(((x + 1) / width) * total));
      let hasDetection = false;
      for (let i = startIdx; i < endIdx && i < total; i++) {
        if (state.detectionByFilename.has(state.frames[i].filename)) {
          hasDetection = true;
          break;
        }
      }
      if (hasDetection) {
        ctx.fillRect(x, 0, 1, height);
      }
    }
  }

  function updateRulerMarker() {
    const marker = el("rulerMarker");
    const total = state.frames.length;
    marker.style.left = total > 1 ? `${(state.currentIndex / (total - 1)) * 100}%` : "0%";
  }

  function setupRulerInteraction() {
    const canvas = el("detectionRuler");
    canvas.addEventListener("click", (event) => {
      const rect = canvas.getBoundingClientRect();
      const frac = (event.clientX - rect.left) / rect.width;
      const index = Math.round(frac * (state.frames.length - 1));
      setCurrentIndex(index);
    });

    window.addEventListener("resize", debounce(renderDetectionRulerBackground, 150));
  }

  function setupHeatmapInteraction() {
    const stage = el("heatmapStage");
    const tooltip = el("heatmapTooltip");
    let rafPending = false;
    let lastEvent = null;

    function stageToFramePx(clientX, clientY) {
      const rect = stage.getBoundingClientRect();
      const { frame_width, frame_height } = state.summary;
      const fx = ((clientX - rect.left) / rect.width) * frame_width;
      const fy = ((clientY - rect.top) / rect.height) * frame_height;
      return { fx, fy, rect };
    }

    function updateTooltip() {
      rafPending = false;
      if (!lastEvent || state.detections.length === 0) return;
      const { fx, fy, rect } = stageToFramePx(lastEvent.clientX, lastEvent.clientY);
      const { index, distance } = nearestDetectionIndex(fx, fy);
      const maxDim = Math.max(state.summary.frame_width, state.summary.frame_height);
      if (distance > maxDim * 0.12) {
        tooltip.style.display = "none";
        return;
      }
      const det = state.detections[index];
      el("tooltipValue").textContent = formatElapsed(elapsedOf(det.timestamp_ms));
      el("tooltipSub").textContent = `${Math.round(det.x)}, ${Math.round(det.y)} · ${Math.round(det.confidence * 100)}%`;
      tooltip.style.left = `${lastEvent.clientX - rect.left}px`;
      tooltip.style.top = `${lastEvent.clientY - rect.top}px`;
      tooltip.style.display = "block";
    }

    stage.addEventListener("pointermove", (event) => {
      lastEvent = event;
      if (!rafPending) {
        rafPending = true;
        requestAnimationFrame(updateTooltip);
      }
    });

    stage.addEventListener("pointerleave", () => {
      tooltip.style.display = "none";
    });

    stage.addEventListener("click", (event) => {
      if (state.detections.length === 0) return;
      const { fx, fy } = stageToFramePx(event.clientX, event.clientY);
      const { index } = nearestDetectionIndex(fx, fy);
      setCurrentIndex(frameIndexForDetectionIndex(index));
    });
  }

  function debounce(fn, wait) {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), wait);
    };
  }

  async function refreshVisits() {
    const distance = Number(el("distanceInput").value);
    const gap = Number(el("gapInput").value) * 1000;
    state.visits = await fetchJSON(`/api/visits?distance=${distance}&gap=${gap}`);
    renderVisitsTable();
    renderKPIs();
  }

  function thumb(imgSrc, captionText, onClick) {
    const wrap = document.createElement("div");
    wrap.className = "thumb";
    const img = document.createElement("img");
    img.src = imgSrc;
    img.alt = captionText;
    img.loading = "lazy";
    const caption = document.createElement("div");
    caption.className = "caption";
    caption.textContent = captionText;
    wrap.append(img, caption);
    wrap.addEventListener("click", onClick);
    return wrap;
  }

  function renderTopArea(areas) {
    const card = el("topAreaCard");
    if (areas.length === 0) {
      card.hidden = true;
      return;
    }
    card.hidden = false;
    const top = areas[0];

    el("topAreaHeadline").textContent =
      top.visit_count === 1
        ? `Visited once, for ${formatElapsed(top.avg_duration_ms)}`
        : `Visited ${top.visit_count} times, ~${formatElapsed(top.avg_duration_ms)} each (${formatElapsed(top.total_duration_ms)} total)`;
    el("topAreaLocation").textContent = `Around (${Math.round(top.centroid_x)}, ${Math.round(top.centroid_y)})`;

    const strip = el("topAreaThumbs");
    strip.replaceChildren(
      ...top.visits.map((visit, i) =>
        thumb(
          `/frames/${encodeURIComponent(visit.representative_filename)}`,
          `Visit ${i + 1} · ${formatElapsed(elapsedOf(visit.start_ts))}`,
          () => setCurrentIndex(frameIndexForDetectionIndex(visit.representative_index))
        )
      )
    );
  }

  async function refreshAreas() {
    const distance = Number(el("distanceInput").value);
    const gap = Number(el("gapInput").value) * 1000;
    const areaRadius = Number(el("areaInput").value);
    const areas = await fetchJSON(`/api/areas?distance=${distance}&gap=${gap}&area_radius=${areaRadius}`);
    renderTopArea(areas);
  }

  function refreshHeatmap() {
    const blur = Number(el("blurInput").value);
    el("heatmapImg").src = `/api/heatmap.png?blur=${blur}&t=${Date.now()}`;
  }

  function setReferenceFrame(filename) {
    el("baseFrame").src = `/frames/${encodeURIComponent(filename)}`;
    el("refFrameLabel").textContent = `Reference: ${filename}`;
  }

  function setupFilterControls() {
    const distanceInput = el("distanceInput");
    const gapInput = el("gapInput");
    const blurInput = el("blurInput");
    const areaInput = el("areaInput");
    const defaults = state.summary.defaults;

    distanceInput.value = defaults.distance_threshold_px;
    gapInput.value = defaults.time_gap_threshold_ms / 1000;
    blurInput.value = defaults.blur_radius_px;
    areaInput.value = defaults.area_radius_px;

    const syncOutputs = () => {
      el("distanceOutput").textContent = `${distanceInput.value}px`;
      el("gapOutput").textContent = `${Number(gapInput.value).toFixed(1)}s`;
      el("blurOutput").textContent = `${blurInput.value}px`;
      el("areaOutput").textContent = `${areaInput.value}px`;
    };
    syncOutputs();

    const debouncedVisits = debounce(() => {
      syncOutputs();
      refreshVisits();
      refreshAreas();
    }, 150);
    const debouncedHeatmap = debounce(() => {
      syncOutputs();
      refreshHeatmap();
    }, 150);
    const debouncedAreas = debounce(() => {
      syncOutputs();
      refreshAreas();
    }, 150);

    distanceInput.addEventListener("input", debouncedVisits);
    gapInput.addEventListener("input", debouncedVisits);
    blurInput.addEventListener("input", debouncedHeatmap);
    areaInput.addEventListener("input", debouncedAreas);
  }

  function setupFrameControls() {
    el("prevBtn").addEventListener("click", () => setCurrentIndex(state.currentIndex - 1));
    el("nextBtn").addEventListener("click", () => setCurrentIndex(state.currentIndex + 1));
    el("frameSlider").addEventListener("input", (event) => {
      setCurrentIndex(Number(event.target.value));
    });
    el("frameImg").addEventListener("load", drawBoundingBox);
    el("useCurrentFrameBtn").addEventListener("click", () => {
      setReferenceFrame(state.frames[state.currentIndex].filename);
    });
  }

  async function init() {
    const summary = await fetchJSON("/api/summary");
    state.summary = summary;
    el("folderLabel").textContent = summary.folder;

    if (summary.frame_count === 0) {
      el("emptyState").hidden = false;
      el("emptyState").textContent = "No JPEG frames were found in this folder.";
      return;
    }

    state.frames = await fetchJSON("/api/frames");
    state.detections = await fetchJSON("/api/detections");
    state.detectionByFilename = new Map(state.detections.map((d) => [d.filename, d]));
    state.frameIndexByFilename = new Map(state.frames.map((f, i) => [f.filename, i]));

    el("dashboard").hidden = false;
    setReferenceFrame(summary.reference_frame || state.frames[0].filename);

    el("frameSlider").max = String(state.frames.length - 1);

    setupFilterControls();
    setupFrameControls();
    setupHeatmapInteraction();
    setupRulerInteraction();
    renderDetectionRulerBackground();

    if (state.detections.length === 0) {
      // Frames exist but nothing was detected -- still show the dashboard so
      // the raw footage can be browsed (e.g. to check focus/framing); the
      // heatmap/visits/most-visited-spot just end up empty, which they
      // already handle on their own.
      el("emptyState").hidden = false;
      el("emptyState").textContent =
        "No dog detections found in this session. The frames were processed, but YOLOv8s did not find a dog in any of them. " +
        "You can still browse the raw frames below -- the ruler will be empty throughout.";
    }

    await Promise.all([refreshVisits(), refreshAreas()]);
    refreshHeatmap();
    setCurrentIndex(0);
  }

  document.addEventListener("DOMContentLoaded", () => {
    init().catch((err) => {
      console.error(err);
      const empty = el("emptyState");
      empty.hidden = false;
      empty.textContent = `Failed to load session data: ${err.message}`;
    });
  });
})();
