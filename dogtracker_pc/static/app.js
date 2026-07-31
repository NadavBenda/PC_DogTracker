(() => {
  "use strict";

  const state = {
    summary: null,
    detections: [],
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

  function renderVisitsTable() {
    const body = el("visitsBody");
    body.replaceChildren();
    state.visits.forEach((visit, i) => {
      const row = document.createElement("tr");
      row.dataset.firstIndex = String(visit.first_index);
      row.dataset.lastIndex = String(visit.last_index);
      row.dataset.midIndex = String(visit.mid_index);

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
      row.addEventListener("click", () => setCurrentIndex(visit.mid_index));
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
    const dets = state.detections;
    if (dets.length === 0) return;
    const clamped = Math.max(0, Math.min(dets.length - 1, i));
    state.currentIndex = clamped;
    const det = dets[clamped];

    const frameImg = el("frameImg");
    frameImg.src = `/frames/${encodeURIComponent(det.filename)}`;
    frameImg.dataset.w = det.w;
    frameImg.dataset.h = det.h;
    frameImg.dataset.x = det.x;
    frameImg.dataset.y = det.y;

    el("frameFilename").textContent = det.filename;
    el("frameTimestamp").textContent = formatElapsed(elapsedOf(det.timestamp_ms));
    el("frameConfidence").textContent = `${Math.round(det.confidence * 100)}% confidence`;

    el("frameSlider").value = String(clamped);
    el("prevBtn").disabled = clamped === 0;
    el("nextBtn").disabled = clamped === dets.length - 1;

    positionCurrentMarker(det);
    updateActiveVisitRow();
  }

  function positionCurrentMarker(det) {
    const stage = el("heatmapStage");
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
      stage.appendChild(marker);
    }
    const { frame_width, frame_height } = state.summary;
    marker.style.left = `${(det.x / frame_width) * 100}%`;
    marker.style.top = `${(det.y / frame_height) * 100}%`;
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

    const bx = Number(frameImg.dataset.x);
    const by = Number(frameImg.dataset.y);
    const bw = Number(frameImg.dataset.w);
    const bh = Number(frameImg.dataset.h);
    ctx.strokeStyle = "#2a78d6";
    ctx.lineWidth = Math.max(2, Math.round(w / 200));
    ctx.strokeRect(bx - bw / 2, by - bh / 2, bw, bh);
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
      setCurrentIndex(index);
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
          () => setCurrentIndex(visit.representative_index)
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
      setReferenceFrame(state.detections[state.currentIndex].filename);
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

    state.detections = await fetchJSON("/api/detections");

    if (state.detections.length === 0) {
      el("emptyState").hidden = false;
      el("emptyState").textContent =
        "No dog detections found in this session. The frames were processed, but YOLOv8s did not find a dog in any of them.";
      renderKPIs();
      return;
    }

    el("dashboard").hidden = false;
    setReferenceFrame(summary.reference_frame || state.detections[0].filename);

    el("frameSlider").max = String(state.detections.length - 1);

    setupFilterControls();
    setupFrameControls();
    setupHeatmapInteraction();

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
