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

  // Fixed count of distinct region colors (see --region-1..6 in style.css) --
  // ranks past this wrap around, since "regions to highlight" tops out at 6.
  const REGION_COLOR_COUNT = 6;
  const SVG_NS = "http://www.w3.org/2000/svg";

  function regionColorVar(rank) {
    const slot = ((rank - 1) % REGION_COLOR_COUNT) + 1;
    return `var(--region-${slot})`;
  }

  // Convex-hull polygon overlays on the reference/heatmap image, one per
  // highlighted area, each in its own color -- an SVG (not absolutely
  // positioned divs) so the polygon points map directly onto frame pixel
  // coordinates via viewBox, matching how the frame's own aspect ratio
  // scales on screen.
  function renderAreaOverlays(areas) {
    const container = el("areaOverlays");
    const { frame_width, frame_height } = state.summary;
    const highlighted = areas.filter((a) => a.is_highlighted && a.hull.length >= 3);

    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${frame_width} ${frame_height}`);
    svg.setAttribute("preserveAspectRatio", "none");

    for (const area of highlighted) {
      const color = regionColorVar(area.rank);
      const polygon = document.createElementNS(SVG_NS, "polygon");
      polygon.setAttribute("points", area.hull.map(([x, y]) => `${x},${y}`).join(" "));
      polygon.setAttribute("class", "area-hull");
      polygon.style.stroke = color;
      polygon.style.fill = color;

      const titleEl = document.createElementNS(SVG_NS, "title");
      titleEl.textContent = `Region ${area.rank} · ${area.visit_count} visit${area.visit_count === 1 ? "" : "s"}, ${formatElapsed(area.total_duration_ms)} total`;
      polygon.appendChild(titleEl);

      const jumpToIndex = frameIndexForDetectionIndex(area.visits[0].representative_index);
      polygon.addEventListener("click", (event) => {
        event.stopPropagation();
        setCurrentIndex(jumpToIndex);
      });

      svg.appendChild(polygon);
    }

    container.replaceChildren(svg);
  }

  function areaCard(area) {
    const card = document.createElement("div");
    card.className = "area-card";
    card.style.setProperty("--region-color", regionColorVar(area.rank));

    const header = document.createElement("div");
    header.className = "area-card-header";
    const title = document.createElement("div");
    title.className = "area-card-title";
    title.textContent = `Region ${area.rank}`;
    const location = document.createElement("div");
    location.className = "area-card-location";
    location.textContent = `Around (${Math.round(area.centroid_x)}, ${Math.round(area.centroid_y)})`;
    header.append(title, location);

    const headline = document.createElement("div");
    headline.className = "area-card-headline";
    headline.textContent =
      area.visit_count === 1
        ? `Visited once, for ${formatElapsed(area.avg_duration_ms)}`
        : `Visited ${area.visit_count} times, ${formatElapsed(area.total_duration_ms)} total`;

    const lengths = document.createElement("div");
    lengths.className = "area-card-lengths";
    lengths.textContent = `Visit lengths: ${area.visits.map((v) => formatElapsed(v.duration_ms)).join(", ")}`;

    const strip = document.createElement("div");
    strip.className = "thumb-strip";
    strip.append(
      ...area.visits.map((visit, i) =>
        thumb(
          `/frames/${encodeURIComponent(visit.representative_filename)}`,
          `Visit ${i + 1} · ${formatElapsed(elapsedOf(visit.start_ts))}`,
          () => setCurrentIndex(frameIndexForDetectionIndex(visit.representative_index))
        )
      )
    );

    card.append(header, headline, lengths, strip);
    return card;
  }

  function renderAreas(data) {
    const card = el("areasCard");
    if (data.areas.length === 0) {
      card.hidden = true;
      renderAreaOverlays([]);
      return;
    }
    card.hidden = false;

    const highlighted = data.areas.filter((a) => a.is_highlighted);
    const overview = el("areasOverview");
    overview.replaceChildren(
      statTile("Time in highlighted regions", formatElapsed(data.total_visit_duration_ms - data.elsewhere_duration_ms)),
      statTile("Elsewhere (in transit)", formatElapsed(data.elsewhere_duration_ms))
    );

    el("areaList").replaceChildren(...highlighted.map(areaCard));
    renderAreaOverlays(data.areas);
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

  // The dog's full path, drawn as a Catmull-Rom spline through every
  // detection center in chronological order, colored along its length from
  // --traj-start (earliest) through --traj-mid to --traj-end (latest) so
  // "when" is readable directly off the line, not just "where".
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
      return;
    }

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

      const [r, g, b] = trajectoryColorAt(i / (n - 2 || 1), stops);
      ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2);
      ctx.stroke();
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
