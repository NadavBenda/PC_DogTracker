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
    currentHighlightedAreas: [],
    drawMode: { active: false, points: [] },
  };

  const el = (id) => document.getElementById(id);

  async function fetchJSON(url) {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`${url} -> ${res.status}`);
    }
    return res.json();
  }

  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      throw new Error(`${url} -> ${res.status}`);
    }
    return res.json();
  }

  async function deleteRequest(url) {
    const res = await fetch(url, { method: "DELETE" });
    if (!res.ok) {
      throw new Error(`${url} -> ${res.status}`);
    }
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

    const styles = getComputedStyle(document.body);
    const accent = styles.getPropertyValue("--accent").trim() || "#2a78d6";
    const surface = styles.getPropertyValue("--surface-1").trim() || "#fcfcfb";

    ctx.strokeStyle = accent;
    ctx.lineWidth = Math.max(2, Math.round(w / 200));
    ctx.strokeRect(bx - bw / 2, by - bh / 2, bw, bh);

    // Center point -- the same position used as "the dog's location" for the
    // heatmap and visits/areas, drawn here too so it reads as the same spot.
    const dotRadius = Math.max(4, Math.round(w / 120));
    ctx.beginPath();
    ctx.arc(bx, by, dotRadius, 0, Math.PI * 2);
    ctx.fillStyle = accent;
    ctx.fill();
    ctx.lineWidth = Math.max(2, Math.round(w / 300));
    ctx.strokeStyle = surface;
    ctx.stroke();
  }

  // ======================================================
  // Presence ruler: a clickable timeline over the whole session showing
  // which parts had some condition true (colored) vs not (empty). Used both
  // for "a dog was detected at all" (the main ruler) and, per region, "the
  // dog was in this specific region" -- same rendering, different predicate
  // and color.
  // ======================================================
  function renderPresenceRuler(canvas, isPresentAtIndex, presentColorCss, height) {
    const width = Math.max(Math.round(canvas.getBoundingClientRect().width), 1);
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");

    const emptyColor = getComputedStyle(document.body).getPropertyValue("--gridline").trim() || "#e1e0d9";
    ctx.fillStyle = emptyColor;
    ctx.fillRect(0, 0, width, height);

    const total = state.frames.length;
    if (total === 0) return;

    ctx.fillStyle = presentColorCss;
    for (let x = 0; x < width; x++) {
      const startIdx = Math.floor((x / width) * total);
      const endIdx = Math.max(startIdx + 1, Math.floor(((x + 1) / width) * total));
      let present = false;
      for (let i = startIdx; i < endIdx && i < total; i++) {
        if (isPresentAtIndex(i)) {
          present = true;
          break;
        }
      }
      if (present) {
        ctx.fillRect(x, 0, 1, height);
      }
    }
  }

  // The coloring only needs recomputing when the data loads or the ruler
  // resizes; the current-frame marker moves far more often, so it's a
  // separate cheap overlay instead of a redraw (see updateRulerMarker).
  function renderDetectionRulerBackground() {
    const accentColor = getComputedStyle(document.body).getPropertyValue("--accent").trim() || "#2a78d6";
    renderPresenceRuler(
      el("detectionRuler"),
      (i) => state.detectionByFilename.has(state.frames[i].filename),
      accentColor,
      22
    );
  }

  // Per-region presence ruler: which parts of the whole session the dog
  // spent inside this specific region, using the region's own visit time
  // ranges (start_ts .. start_ts + duration_ms) rather than exact frames,
  // since that's what /api/areas already reports per visit.
  function renderAreaRuler(canvas, region) {
    const intervals = region.visits.map((v) => [v.start_ts, v.start_ts + v.duration_ms]);
    const isInRegion = (i) => {
      const ts = state.frames[i].timestamp_ms;
      return intervals.some(([start, end]) => ts >= start && ts <= end);
    };
    const styles = getComputedStyle(document.body);
    const slot = ((region.displayRank - 1) % REGION_COLOR_COUNT) + 1;
    const color = styles.getPropertyValue(`--region-${slot}`).trim() || "#2a78d6";
    renderPresenceRuler(canvas, isInRegion, color, 14);
  }

  function redrawAreaRulers() {
    const cards = el("areaList").children;
    for (let i = 0; i < cards.length && i < state.currentHighlightedAreas.length; i++) {
      const canvas = cards[i].querySelector(".area-ruler");
      if (canvas) renderAreaRuler(canvas, state.currentHighlightedAreas[i]);
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

    window.addEventListener(
      "resize",
      debounce(() => {
        renderDetectionRulerBackground();
        redrawAreaRulers();
      }, 150)
    );
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
      if (state.drawMode.active) {
        addDrawPoint(event, stageToFramePx);
        return;
      }
      if (state.detections.length === 0) return;
      const { fx, fy } = stageToFramePx(event.clientX, event.clientY);
      const { index } = nearestDetectionIndex(fx, fy);
      setCurrentIndex(frameIndexForDetectionIndex(index));
    });
  }

  // ======================================================
  // Manual region drawing: click points on the heatmap to trace an arbitrary
  // polygon, closed either by clicking back near the first point or via the
  // explicit Finish button, then persisted server-side (see
  // dogtracker_pc/regions.py) alongside the automatically-clustered areas.
  // ======================================================
  const DRAW_CLOSE_THRESHOLD_PX = 12;

  function renderDraftOverlay() {
    const svg = el("drawOverlay");
    const { frame_width, frame_height } = state.summary;
    svg.setAttribute("viewBox", `0 0 ${frame_width} ${frame_height}`);
    svg.setAttribute("preserveAspectRatio", "none");
    svg.replaceChildren();

    const points = state.drawMode.points;
    if (points.length >= 2) {
      const path = document.createElementNS(SVG_NS, "polyline");
      path.setAttribute("points", points.map(([x, y]) => `${x},${y}`).join(" "));
      path.setAttribute("class", "draw-path");
      path.setAttribute("fill", "none");
      svg.appendChild(path);
    }
    for (const [x, y] of points) {
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", String(x));
      dot.setAttribute("cy", String(y));
      dot.setAttribute("r", String(Math.max(3, frame_width / 160)));
      dot.setAttribute("class", "draw-vertex");
      svg.appendChild(dot);
    }
  }

  function addDrawPoint(event, stageToFramePx) {
    const { fx, fy, rect } = stageToFramePx(event.clientX, event.clientY);
    const points = state.drawMode.points;

    if (points.length >= 3) {
      const { frame_width, frame_height } = state.summary;
      const [firstFx, firstFy] = points[0];
      const firstClientX = rect.left + (firstFx / frame_width) * rect.width;
      const firstClientY = rect.top + (firstFy / frame_height) * rect.height;
      const dist = Math.hypot(event.clientX - firstClientX, event.clientY - firstClientY);
      if (dist <= DRAW_CLOSE_THRESHOLD_PX) {
        finishDrawMode();
        return;
      }
    }

    points.push([fx, fy]);
    renderDraftOverlay();
    el("finishDrawBtn").disabled = points.length < 3;
  }

  function startDrawMode() {
    state.drawMode = { active: true, points: [] };
    el("drawModeBar").hidden = false;
    el("drawRegionBtn").disabled = true;
    el("finishDrawBtn").disabled = true;
    renderDraftOverlay();
  }

  function exitDrawMode() {
    state.drawMode = { active: false, points: [] };
    el("drawModeBar").hidden = true;
    el("drawRegionBtn").disabled = false;
    el("drawOverlay").replaceChildren();
  }

  async function finishDrawMode() {
    const points = state.drawMode.points;
    if (points.length < 3) return;
    exitDrawMode();
    try {
      await postJSON("/api/manual-regions", { points });
      await refreshAreas();
    } catch (err) {
      console.error("Failed to save the drawn region:", err);
    }
  }

  async function removeManualRegion(id) {
    try {
      await deleteRequest(`/api/manual-regions/${encodeURIComponent(id)}`);
      await refreshAreas();
    } catch (err) {
      console.error("Failed to remove the region:", err);
    }
  }

  function setupDrawModeControls() {
    el("drawRegionBtn").addEventListener("click", startDrawMode);
    el("cancelDrawBtn").addEventListener("click", exitDrawMode);
    el("finishDrawBtn").addEventListener("click", finishDrawMode);
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

  // Fixed count of distinct region colors (see --region-1..6 in style.css) --
  // ranks past this wrap around, since "regions to highlight" tops out at 6.
  const REGION_COLOR_COUNT = 6;
  const SVG_NS = "http://www.w3.org/2000/svg";

  function regionColorVar(displayRank) {
    const slot = ((displayRank - 1) % REGION_COLOR_COUNT) + 1;
    return `var(--region-${slot})`;
  }

  function centroidOfPoints(points) {
    const n = points.length;
    const sx = points.reduce((s, p) => s + p[0], 0);
    const sy = points.reduce((s, p) => s + p[1], 0);
    return [sx / n, sy / n];
  }

  // Convex-hull polygon overlays on the reference/heatmap image, one per
  // region (automatic or manually drawn), each in its own color -- an SVG
  // (not absolutely positioned divs) so the polygon points map directly onto
  // frame pixel coordinates via viewBox, matching how the frame's own aspect
  // ratio scales on screen. Purely decorative (no pointer events): they sit
  // on top of the heatmap, and capturing clicks there would shadow clicks on
  // the heatmap's own points underneath.
  function renderAreaOverlays(regions) {
    const container = el("areaOverlays");
    const { frame_width, frame_height } = state.summary;

    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${frame_width} ${frame_height}`);
    svg.setAttribute("preserveAspectRatio", "none");

    for (const region of regions) {
      if (!region.hull || region.hull.length < 3) continue;
      const color = regionColorVar(region.displayRank);
      const polygon = document.createElementNS(SVG_NS, "polygon");
      polygon.setAttribute("points", region.hull.map(([x, y]) => `${x},${y}`).join(" "));
      polygon.setAttribute("class", "area-hull");
      polygon.style.stroke = color;
      polygon.style.fill = color;
      svg.appendChild(polygon);
    }

    container.replaceChildren(svg);
  }

  function areaCard(region) {
    const isManual = region.kind === "manual";
    const [centroidX, centroidY] = isManual ? centroidOfPoints(region.hull) : [region.centroid_x, region.centroid_y];

    const card = document.createElement("div");
    card.className = "area-card";
    card.style.setProperty("--region-color", regionColorVar(region.displayRank));

    const header = document.createElement("div");
    header.className = "area-card-header";

    const titleGroup = document.createElement("div");
    titleGroup.className = "area-card-title-group";
    const title = document.createElement("div");
    title.className = "area-card-title";
    title.textContent = `Region ${region.displayRank}`;
    titleGroup.appendChild(title);
    if (isManual) {
      const badge = document.createElement("span");
      badge.className = "area-card-manual-badge";
      badge.textContent = "manual";
      titleGroup.appendChild(badge);
    }

    const location = document.createElement("div");
    location.className = "area-card-location";
    location.textContent = `Around (${Math.round(centroidX)}, ${Math.round(centroidY)})`;
    header.append(titleGroup, location);

    if (isManual) {
      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "ghost area-card-remove";
      removeBtn.textContent = "Remove";
      removeBtn.addEventListener("click", () => removeManualRegion(region.id));
      header.appendChild(removeBtn);
    }

    const headline = document.createElement("div");
    headline.className = "area-card-headline";
    headline.textContent =
      region.visit_count === 0
        ? "No visits in this region yet"
        : region.visit_count === 1
        ? `Visited once, for ${formatElapsed(region.avg_duration_ms)}`
        : `Visited ${region.visit_count} times, ${formatElapsed(region.total_duration_ms)} total`;

    const lengths = document.createElement("div");
    lengths.className = "area-card-lengths";
    lengths.textContent =
      region.visits.length > 0
        ? `Visit lengths: ${region.visits.map((v) => formatElapsed(v.duration_ms)).join(", ")}`
        : "";

    // Mini presence ruler -- same idea as the main detection-presence ruler
    // under the frame browser, scoped to just this region: colored (in this
    // region's own color) where the dog was here, empty everywhere else
    // across the whole session timeline. Drawn later, once the canvas is
    // actually in the DOM and has a real width (see redrawAreaRulers).
    const rulerWrap = document.createElement("div");
    rulerWrap.className = "area-ruler-wrap";
    const rulerCanvas = document.createElement("canvas");
    rulerCanvas.className = "area-ruler";
    rulerCanvas.addEventListener("click", (event) => {
      const rect = rulerCanvas.getBoundingClientRect();
      const frac = (event.clientX - rect.left) / rect.width;
      setCurrentIndex(Math.round(frac * (state.frames.length - 1)));
    });
    rulerWrap.appendChild(rulerCanvas);

    const rulerLegend = document.createElement("div");
    rulerLegend.className = "ruler-legend";
    const onSwatch = document.createElement("span");
    onSwatch.className = "swatch";
    onSwatch.style.background = regionColorVar(region.displayRank);
    const onLabel = document.createElement("span");
    onLabel.append(onSwatch, document.createTextNode("in this region"));
    const offSwatch = document.createElement("span");
    offSwatch.className = "swatch swatch-off";
    const offLabel = document.createElement("span");
    offLabel.append(offSwatch, document.createTextNode("elsewhere / not detected"));
    rulerLegend.append(onLabel, offLabel);

    const strip = document.createElement("div");
    strip.className = "thumb-strip";
    strip.append(
      ...region.visits.map((visit, i) =>
        thumb(
          `/frames/${encodeURIComponent(visit.representative_filename)}`,
          `Visit ${i + 1} · ${formatElapsed(elapsedOf(visit.start_ts))}`,
          () => setCurrentIndex(frameIndexForDetectionIndex(visit.representative_index))
        )
      )
    );

    card.append(header, headline, lengths, rulerWrap, rulerLegend, strip);
    return card;
  }

  // Automatic (clustered) and manually-drawn regions are both "preferred
  // locations" and share one combined, continuously color-numbered list --
  // there's no reason to keep them in visually separate systems, since a
  // manual region is just a stand-in for a spot the automatic clustering
  // missed or split awkwardly. Automatic areas are listed first (in their
  // ranked order), manual ones after (in creation order).
  function renderAreas(data) {
    const card = el("areasCard");
    const highlightedAuto = data.areas.filter((a) => a.is_highlighted).map((a) => ({ ...a, kind: "auto" }));
    const manual = (data.manual_regions || []).map((r) => ({ ...r, kind: "manual" }));
    const combined = [...highlightedAuto, ...manual].map((r, i) => ({ ...r, displayRank: i + 1 }));

    if (combined.length === 0) {
      card.hidden = true;
      state.currentHighlightedAreas = [];
      renderAreaOverlays([]);
      return;
    }
    card.hidden = false;

    const overview = el("areasOverview");
    overview.replaceChildren(
      statTile("Time in highlighted regions", formatElapsed(data.total_visit_duration_ms - data.elsewhere_duration_ms)),
      statTile("Elsewhere (in transit)", formatElapsed(data.elsewhere_duration_ms))
    );

    state.currentHighlightedAreas = combined;
    el("areaList").replaceChildren(...combined.map(areaCard));
    redrawAreaRulers();
    renderAreaOverlays(combined);
  }

  async function refreshAreas() {
    const distance = Number(el("distanceInput").value);
    const gap = Number(el("gapInput").value) * 1000;
    const areaRadius = Number(el("areaInput").value);
    const topN = Number(el("topNInput").value);
    const data = await fetchJSON(
      `/api/areas?distance=${distance}&gap=${gap}&area_radius=${areaRadius}&top_n=${topN}`
    );
    renderAreas(data);
  }

  function refreshHeatmap() {
    const blur = Number(el("blurInput").value);
    el("heatmapImg").src = `/api/heatmap.png?blur=${blur}&t=${Date.now()}`;
  }

  function setReferenceFrame(filename) {
    const src = `/frames/${encodeURIComponent(filename)}`;
    el("baseFrame").src = src;
    el("trajectoryBaseFrame").src = src;
    el("refFrameLabel").textContent = `Reference: ${filename}`;
  }

  function hexToRgb(hex) {
    const h = hex.trim().replace("#", "");
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }

  function trajectoryColorAt(t, stops) {
    for (let i = 0; i < stops.length - 1; i++) {
      const a = stops[i];
      const b = stops[i + 1];
      if (t <= b.t) {
        const localT = (t - a.t) / (b.t - a.t || 1);
        return a.rgb.map((v, ch) => Math.round(v + (b.rgb[ch] - v) * localT));
      }
    }
    return stops[stops.length - 1].rgb;
  }

  // Fraction of the session elapsed at a given timestamp (0 at the first
  // frame, 1 at the last) -- used for the trajectory's time gradient and its
  // 0-10 tick marks, so both track wall-clock time rather than detection
  // sample index (which would compress/stretch wherever detections happen
  // to be denser or sparser).
  function elapsedFractionOf(timestampMs) {
    const duration = state.summary.duration_ms || 1;
    return Math.min(1, Math.max(0, elapsedOf(timestampMs) / duration));
  }

  // Linearly interpolated (x, y) at a given elapsed-time fraction, between
  // whichever two consecutive detections straddle that instant.
  function positionAtElapsedFraction(dets, frac) {
    const targetTs = state.summary.first_timestamp_ms + frac * (state.summary.duration_ms || 0);
    if (targetTs <= dets[0].timestamp_ms) return [dets[0].x, dets[0].y];
    for (let i = 0; i < dets.length - 1; i++) {
      if (targetTs <= dets[i + 1].timestamp_ms) {
        const span = dets[i + 1].timestamp_ms - dets[i].timestamp_ms;
        const localT = span > 0 ? (targetTs - dets[i].timestamp_ms) / span : 0;
        return [
          dets[i].x + (dets[i + 1].x - dets[i].x) * localT,
          dets[i].y + (dets[i + 1].y - dets[i].y) * localT,
        ];
      }
    }
    return [dets[dets.length - 1].x, dets[dets.length - 1].y];
  }

  // The dog's full path, drawn as a Catmull-Rom spline through every
  // detection center in chronological order, colored along its length from
  // --traj-start (earliest) through --traj-mid to --traj-end (latest) so
  // "when" is readable directly off the line, not just "where". Numbered
  // badges 0-10 mark every 10% of elapsed session time, matching the ticks
  // under the legend below.
  function renderTrajectory() {
    const card = el("trajectoryCard");
    const dets = state.detections;
    if (dets.length === 0) {
      card.hidden = true;
      return;
    }
    card.hidden = false;

    const { frame_width, frame_height } = state.summary;
    const canvas = el("trajectoryCanvas");
    canvas.width = frame_width;
    canvas.height = frame_height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, frame_width, frame_height);

    const styles = getComputedStyle(document.body);
    const stops = [
      { t: 0, rgb: hexToRgb(styles.getPropertyValue("--traj-start") || "#2a78d6") },
      { t: 0.5, rgb: hexToRgb(styles.getPropertyValue("--traj-mid") || "#4a3aa7") },
      { t: 1, rgb: hexToRgb(styles.getPropertyValue("--traj-end") || "#e34948") },
    ];
    const surface = styles.getPropertyValue("--surface-1").trim() || "#fcfcfb";

    const n = dets.length;
    ctx.lineWidth = Math.max(2, Math.round(frame_width / 220));
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (n === 1) {
      const [r, g, b] = stops[0].rgb;
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.beginPath();
      ctx.arc(dets[0].x, dets[0].y, ctx.lineWidth, 0, Math.PI * 2);
      ctx.fill();
    } else {
      const pointAt = (i) => {
        const idx = Math.max(0, Math.min(n - 1, i));
        return [dets[idx].x, dets[idx].y];
      };

      for (let i = 0; i < n - 1; i++) {
        const [x0, y0] = pointAt(i - 1);
        const [x1, y1] = pointAt(i);
        const [x2, y2] = pointAt(i + 1);
        const [x3, y3] = pointAt(i + 2);
        // Catmull-Rom -> cubic Bezier control points for the segment [P1, P2].
        const cp1x = x1 + (x2 - x0) / 6;
        const cp1y = y1 + (y2 - y0) / 6;
        const cp2x = x2 - (x3 - x1) / 6;
        const cp2y = y2 - (y3 - y1) / 6;

        const midT = (elapsedFractionOf(dets[i].timestamp_ms) + elapsedFractionOf(dets[i + 1].timestamp_ms)) / 2;
        const [r, g, b] = trajectoryColorAt(midT, stops);
        ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2);
        ctx.stroke();
      }
    }

    // 0-10 badges at every 10% of elapsed time, drawn last so they sit on
    // top of the path lines.
    const badgeRadius = Math.max(8, Math.round(frame_width / 45));
    ctx.font = `${Math.round(badgeRadius * 1.1)}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let k = 0; k <= 10; k++) {
      const frac = k / 10;
      const [x, y] = positionAtElapsedFraction(dets, frac);
      const [r, g, b] = trajectoryColorAt(frac, stops);
      ctx.beginPath();
      ctx.arc(x, y, badgeRadius, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fill();
      ctx.lineWidth = Math.max(1.5, badgeRadius / 8);
      ctx.strokeStyle = surface;
      ctx.stroke();

      // Pick black or white text by background luminance so the number
      // stays legible across the whole blue -> violet -> red gradient.
      const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
      ctx.fillStyle = luminance > 140 ? "#0b0b0b" : "#ffffff";
      ctx.fillText(String(k), x, y + 1);
    }
  }

  function setupFilterControls() {
    const distanceInput = el("distanceInput");
    const gapInput = el("gapInput");
    const blurInput = el("blurInput");
    const areaInput = el("areaInput");
    const topNInput = el("topNInput");
    const defaults = state.summary.defaults;

    distanceInput.value = defaults.distance_threshold_px;
    gapInput.value = defaults.time_gap_threshold_ms / 1000;
    blurInput.value = defaults.blur_radius_px;
    areaInput.value = defaults.area_radius_px;
    topNInput.value = defaults.top_areas_count;

    const syncOutputs = () => {
      el("distanceOutput").textContent = `${distanceInput.value}px`;
      el("gapOutput").textContent = `${Number(gapInput.value).toFixed(1)}s`;
      el("blurOutput").textContent = `${blurInput.value}px`;
      el("areaOutput").textContent = `${areaInput.value}px`;
      el("topNOutput").textContent = topNInput.value;
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
    topNInput.addEventListener("input", debouncedAreas);
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
    setupDrawModeControls();
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
    renderTrajectory();
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
