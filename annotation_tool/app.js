import { GOAL_PLANE_POINTS, projectPoint, solveHomography } from "./geometry.js";
import { parseMp4Fps } from "./mp4fps.js";

const video = document.querySelector("#video");
const canvas = document.querySelector("#overlay");
const context = canvas.getContext("2d");
const scrubber = document.querySelector("#scrubber");
const frameReadout = document.querySelector("#frameReadout");
const validation = document.querySelector("#validation");
const fpsStatus = document.querySelector("#fpsStatus");
const markerLabels = {
  left_post_base: "left post base",
  right_post_base: "right post base",
  left_crossbar: "left crossbar corner",
  right_crossbar: "right crossbar corner",
  ball_crossing: "ball crossing",
  keeper_strike: "keeper COM at strike",
  keeper_crossing: "keeper COM at crossing",
  contact_location: "contact location",
};
const calibrationOrder = ["left_post_base", "right_post_base", "left_crossbar", "right_crossbar"];
let clipFile = null;
let fps = null;
let fpsMetadata = null;
let currentFrame = 0;
let strikeFrame = null;
let crossingFrame = null;
let activeMarker = null;
let markers = {};
let jsonlHandle = null;

function maxFrame() {
  return fps && Number.isFinite(video.duration) ? Math.max(0, Math.floor(video.duration * fps) - 1) : 0;
}

function resizeCanvas() {
  const rect = video.parentElement.getBoundingClientRect();
  canvas.width = Math.round(rect.width * devicePixelRatio);
  canvas.height = Math.round(rect.height * devicePixelRatio);
  context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  drawMarkers();
}

function videoDisplayGeometry() {
  const wrap = video.parentElement.getBoundingClientRect();
  const scale = Math.min(wrap.width / video.videoWidth, wrap.height / video.videoHeight);
  const width = video.videoWidth * scale;
  const height = video.videoHeight * scale;
  return { scale, offsetX: (wrap.width - width) / 2, offsetY: (wrap.height - height) / 2, width, height };
}

function eventToVideoPixel(event) {
  const rect = canvas.getBoundingClientRect();
  const localX = event.clientX - rect.left;
  const localY = event.clientY - rect.top;
  const geometry = videoDisplayGeometry();
  const x = (localX - geometry.offsetX) / geometry.scale;
  const y = (localY - geometry.offsetY) / geometry.scale;
  if (x < 0 || y < 0 || x > video.videoWidth || y > video.videoHeight) throw new Error("Click inside the video image, not the letterbox area");
  return [x, y];
}

function videoPixelToDisplay([x, y]) {
  const geometry = videoDisplayGeometry();
  return [geometry.offsetX + x * geometry.scale, geometry.offsetY + y * geometry.scale];
}

function drawMarkers() {
  context.clearRect(0, 0, canvas.width / devicePixelRatio, canvas.height / devicePixelRatio);
  if (!video.videoWidth) return;
  Object.entries(markers).forEach(([name, point], index) => {
    const [x, y] = videoPixelToDisplay(point);
    context.beginPath();
    context.arc(x, y, 6, 0, 2 * Math.PI);
    context.fillStyle = name.startsWith("keeper") ? "#00e5ff" : name.includes("post") || name.includes("crossbar") ? "#ffe600" : "#ff3b30";
    context.fill();
    context.strokeStyle = "#000";
    context.stroke();
    context.fillStyle = "white";
    context.font = "bold 13px system-ui";
    context.fillText(String(index + 1), x + 8, y - 8);
  });
}

function updateFrameUi() {
  scrubber.value = String(currentFrame);
  frameReadout.textContent = `frame ${currentFrame} / ${maxFrame()}  (${(currentFrame / (fps || 1)).toFixed(3)} s)`;
}

function seekFrame(frame) {
  if (!fps) return;
  currentFrame = Math.max(0, Math.min(maxFrame(), Math.round(frame)));
  video.currentTime = currentFrame / fps;
  updateFrameUi();
}

function nearestFrame() {
  if (!fps) return 0;
  return Math.max(0, Math.min(maxFrame(), Math.round(video.currentTime * fps)));
}

function calibrationPoints() {
  return calibrationOrder.map((name) => markers[name]);
}

function homography() {
  const points = calibrationPoints();
  if (points.some((point) => !point)) throw new Error("All four goal-frame reference points are required");
  return solveHomography(points, GOAL_PLANE_POINTS);
}

function projected(name) {
  if (!markers[name]) throw new Error(`${markerLabels[name]} is not marked`);
  return projectPoint(homography(), markers[name]);
}

function updateCoordinateReadout() {
  const lines = [];
  try {
    const h = homography();
    for (const name of ["ball_crossing", "keeper_strike", "keeper_crossing", "contact_location"]) {
      if (markers[name]) {
        const [x, y] = projectPoint(h, markers[name]);
        lines.push(`${markerLabels[name]}: (${x.toFixed(3)}, ${y.toFixed(3)}) m`);
      }
    }
  } catch (error) {
    lines.push(error.message);
  }
  document.querySelector("#coordinateReadout").textContent = lines.join("\n");
}

function qualityFlags() {
  return [...document.querySelectorAll(".quality:checked")].map((input) => input.value);
}

function validateState() {
  const errors = [];
  if (!clipFile) errors.push("Load an MP4 clip.");
  if (!fps) errors.push("FPS metadata is unavailable.");
  if (!("showSaveFilePicker" in window)) errors.push("Direct append requires Chromium File System Access API.");
  if (!jsonlHandle) errors.push("Choose or create the append-only JSONL destination.");
  if (!document.querySelector("#keeperId").value.trim()) errors.push("Keeper ID is required.");
  if (!document.querySelector("#annotator").value.trim()) errors.push("Annotator is required.");
  if (strikeFrame === null) errors.push("Mark the strike frame.");
  if (crossingFrame === null) errors.push("Mark the crossing/contact frame.");
  if (strikeFrame !== null && crossingFrame !== null && crossingFrame <= strikeFrame) errors.push("Crossing frame must follow strike frame.");
  for (const name of [...calibrationOrder, "ball_crossing", "keeper_strike", "keeper_crossing"]) {
    if (!markers[name]) errors.push(`Mark ${markerLabels[name]}.`);
  }
  const outcome = document.querySelector("#outcome").value;
  if (outcome === "save" && !markers.contact_location) errors.push("Save outcome requires a contact location.");
  if (outcome === "excluded") errors.push("Off-target clips are excluded and cannot be appended.");
  try { if (calibrationPoints().every(Boolean)) homography(); } catch (error) { errors.push(error.message); }
  validation.textContent = errors.length ? errors.map((error) => `• ${error}`).join("\n") : "Ready to append. Geometry, timing, and censoring logic are internally consistent.";
  return errors;
}

function buildRecord() {
  const errors = validateState();
  if (errors.some((error) => !error.includes("fallback"))) throw new Error("Resolve validation errors before saving");
  const outcome = document.querySelector("#outcome").value;
  const contact = outcome === "save";
  const ball = projected("ball_crossing");
  const keeperStrike = projected("keeper_strike");
  const keeperCrossing = projected("keeper_crossing");
  const flightTime = (crossingFrame - strikeFrame) / fps;
  return {
    id: crypto.randomUUID(),
    keeper_id: document.querySelector("#keeperId").value.trim(),
    source_url: document.querySelector("#sourceUrl").value.trim(),
    clip_file: clipFile.name,
    fps,
    strike_frame: strikeFrame,
    crossing_frame: crossingFrame,
    flight_time_s: flightTime,
    ball_crossing_xy_m: ball,
    keeper_pos_at_strike_xy_m: keeperStrike,
    keeper_pos_at_crossing_xy_m: keeperCrossing,
    displacement_m: [keeperCrossing[0] - keeperStrike[0], keeperCrossing[1] - keeperStrike[1]],
    contact,
    contact_xy_m: contact ? projected("contact_location") : null,
    time_to_contact_s: contact ? flightTime : null,
    censored: !contact,
    outcome,
    dive_direction: document.querySelector("#diveDirection").value,
    contact_body_part: document.querySelector("#bodyPart").value.trim() || null,
    quality_flags: qualityFlags(),
    annotator: document.querySelector("#annotator").value.trim(),
    annotated_at: new Date().toISOString(),
    annotation_metadata: {
      homography_pixel_to_goal: homography(),
      goal_reference_pixels: Object.fromEntries(calibrationOrder.map((name) => [name, markers[name]])),
      fps_metadata: fpsMetadata,
    },
  };
}

async function appendJsonl(record) {
  const line = `${JSON.stringify(record)}\n`;
  if (!jsonlHandle) throw new Error("Choose the JSONL destination first");
  const file = await jsonlHandle.getFile();
  const writer = await jsonlHandle.createWritable({ keepExistingData: true });
  await writer.seek(file.size);
  await writer.write(line);
  await writer.close();
}

function resetAnnotation() {
  strikeFrame = null;
  crossingFrame = null;
  activeMarker = null;
  markers = {};
  document.querySelector("#strikeReadout").textContent = "not marked";
  document.querySelector("#crossingReadout").textContent = "not marked";
  document.querySelector("#activeMarker").textContent = "Select a marker, then click the video.";
  document.querySelectorAll("[data-marker]").forEach((button) => button.classList.remove("active"));
  drawMarkers(); updateCoordinateReadout(); validateState();
}

document.querySelector("#videoFile").addEventListener("change", async (event) => {
  clipFile = event.target.files[0] || null;
  fps = null; fpsMetadata = null;
  if (!clipFile) return;
  video.src = URL.createObjectURL(clipFile);
  try {
    fpsMetadata = parseMp4Fps(await clipFile.arrayBuffer());
    document.querySelector("#vfrFlag").checked = fpsMetadata.variableFrameRate;
    if (fpsMetadata.variableFrameRate) {
      fpsStatus.textContent = `BLOCKED: variable-frame-rate MP4 (${fpsMetadata.fps.toFixed(6)} average fps)`;
      validation.textContent = "Frame-index timing is unreliable for VFR footage. Transcode to constant frame rate before annotation.";
      fps = null;
      return;
    }
    fps = fpsMetadata.fps;
    fpsStatus.textContent = `${fps.toFixed(6)} fps — ${fpsMetadata.timingSource}`;
  } catch (error) {
    fpsStatus.textContent = `BLOCKED: ${error.message}`;
    validation.textContent = "FPS must come from file metadata. This clip cannot be annotated until its timing metadata is repaired or transcoded.";
  }
});

video.addEventListener("loadedmetadata", () => {
  if (fps) {
    scrubber.max = String(maxFrame());
    seekFrame(0);
  }
  resizeCanvas(); validateState();
});
video.addEventListener("seeked", () => { currentFrame = nearestFrame(); updateFrameUi(); drawMarkers(); });
video.addEventListener("timeupdate", () => { if (!video.paused) { currentFrame = nearestFrame(); updateFrameUi(); } });
window.addEventListener("resize", resizeCanvas);

document.querySelectorAll("[data-step]").forEach((button) => button.addEventListener("click", () => { video.pause(); seekFrame(currentFrame + Number(button.dataset.step)); }));
document.querySelector("#playPause").addEventListener("click", () => { if (video.paused) video.play(); else video.pause(); });
scrubber.addEventListener("input", () => { video.pause(); seekFrame(Number(scrubber.value)); });
document.addEventListener("keydown", (event) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (event.key === "ArrowLeft") { event.preventDefault(); seekFrame(currentFrame - (event.shiftKey ? 5 : 1)); }
  if (event.key === "ArrowRight") { event.preventDefault(); seekFrame(currentFrame + (event.shiftKey ? 5 : 1)); }
  if (event.key === " ") { event.preventDefault(); if (video.paused) video.play(); else video.pause(); }
});

document.querySelector("#markStrike").addEventListener("click", () => { strikeFrame = currentFrame; document.querySelector("#strikeReadout").textContent = `frame ${strikeFrame}`; validateState(); });
document.querySelector("#markCrossing").addEventListener("click", () => { crossingFrame = currentFrame; document.querySelector("#crossingReadout").textContent = `frame ${crossingFrame}`; validateState(); });

document.querySelectorAll("[data-marker]").forEach((button) => button.addEventListener("click", () => {
  activeMarker = button.dataset.marker;
  document.querySelectorAll("[data-marker]").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
  document.querySelector("#activeMarker").textContent = `Click ${markerLabels[activeMarker]} on the video.`;
  if (activeMarker === "keeper_strike" && strikeFrame !== null) seekFrame(strikeFrame);
  if (["ball_crossing", "keeper_crossing", "contact_location"].includes(activeMarker) && crossingFrame !== null) seekFrame(crossingFrame);
}));

canvas.addEventListener("click", (event) => {
  if (!activeMarker) return;
  try {
    markers[activeMarker] = eventToVideoPixel(event);
    const index = Object.keys(markerLabels).indexOf(activeMarker);
    const nextName = Object.keys(markerLabels)[index + 1];
    activeMarker = nextName || null;
    document.querySelectorAll("[data-marker]").forEach((button) => button.classList.toggle("active", button.dataset.marker === activeMarker));
    document.querySelector("#activeMarker").textContent = activeMarker ? `Click ${markerLabels[activeMarker]} on the video.` : "All marker slots have been visited.";
    if (activeMarker === "keeper_strike" && strikeFrame !== null) seekFrame(strikeFrame);
    if (["ball_crossing", "keeper_crossing", "contact_location"].includes(activeMarker) && crossingFrame !== null) seekFrame(crossingFrame);
    drawMarkers(); updateCoordinateReadout(); validateState();
  } catch (error) { validation.textContent = error.message; }
});

document.querySelector("#chooseJsonl").addEventListener("click", async () => {
  if (!("showSaveFilePicker" in window)) { validation.textContent = "Direct append requires a Chromium browser with File System Access API."; return; }
  jsonlHandle = await window.showSaveFilePicker({ suggestedName: "penalties.jsonl", types: [{ description: "JSON Lines", accept: { "application/x-ndjson": [".jsonl"] } }] });
  document.querySelector("#chooseJsonl").textContent = `JSONL: ${jsonlHandle.name}`;
  validateState();
});

document.querySelector("#saveRecord").addEventListener("click", async () => {
  try {
    const record = buildRecord();
    await appendJsonl(record);
    validation.textContent = `Appended ${record.id}\nflight=${record.flight_time_s.toFixed(4)} s; censored=${record.censored}`;
    resetAnnotation();
  } catch (error) { validation.textContent = error.message; }
});
document.querySelector("#resetRecord").addEventListener("click", resetAnnotation);
document.querySelectorAll("input,select").forEach((element) => element.addEventListener("change", validateState));
