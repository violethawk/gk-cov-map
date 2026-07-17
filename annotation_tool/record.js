// Annotation logic with no DOM dependency, so it can be exercised in node the way
// geometry.js and mp4fps.js already are. app.js owns the elements and events and
// hands the state in; everything that decides what reaches the JSONL lives here.
import { GOAL_PLANE_POINTS, projectPoint, solveHomography } from "./geometry.js";

export const MARKER_LABELS = {
  left_post_base: "left post base",
  right_post_base: "right post base",
  left_crossbar: "left crossbar corner",
  right_crossbar: "right crossbar corner",
  ball_crossing: "ball crossing",
  keeper_strike: "keeper COM at strike",
  keeper_crossing: "keeper COM at crossing",
  contact_location: "contact location",
};

export const CALIBRATION_ORDER = ["left_post_base", "right_post_base", "left_crossbar", "right_crossbar"];
export const REQUIRED_MARKERS = [...CALIBRATION_ORDER, "ball_crossing", "keeper_strike", "keeper_crossing"];
export const READY_MESSAGE = "Ready to append. Geometry, timing, and censoring logic are internally consistent.";

export function annotationHomography(markers) {
  const points = CALIBRATION_ORDER.map((name) => markers[name]);
  if (points.some((point) => !point)) throw new Error("All four goal-frame reference points are required");
  return solveHomography(points, GOAL_PLANE_POINTS);
}

export function projectMarker(markers, name) {
  if (!markers[name]) throw new Error(`${MARKER_LABELS[name]} is not marked`);
  return projectPoint(annotationHomography(markers), markers[name]);
}

export function validateAnnotation(state) {
  const errors = [];
  const markers = state.markers || {};
  if (!state.clipFile) errors.push("Load an MP4 clip.");
  if (!state.fps) errors.push("FPS metadata is unavailable.");
  if (!state.filePickerAvailable) errors.push("Direct append requires Chromium File System Access API.");
  if (!state.jsonlChosen) errors.push("Choose or create the append-only JSONL destination.");
  if (!String(state.keeperId || "").trim()) errors.push("Keeper ID is required.");
  if (!String(state.annotator || "").trim()) errors.push("Annotator is required.");
  if (state.strikeFrame === null || state.strikeFrame === undefined) errors.push("Mark the strike frame.");
  if (state.crossingFrame === null || state.crossingFrame === undefined) errors.push("Mark the crossing/contact frame.");
  if (
    state.strikeFrame !== null && state.strikeFrame !== undefined &&
    state.crossingFrame !== null && state.crossingFrame !== undefined &&
    state.crossingFrame <= state.strikeFrame
  ) {
    errors.push("Crossing frame must follow strike frame.");
  }
  for (const name of REQUIRED_MARKERS) {
    if (!markers[name]) errors.push(`Mark ${MARKER_LABELS[name]}.`);
  }
  if (state.outcome === "save" && !markers.contact_location) errors.push("Save outcome requires a contact location.");
  if (state.outcome === "excluded") errors.push("Off-target clips are excluded and cannot be appended.");
  // Four clicks can be present but collinear or coincident, which only the solve catches.
  try {
    if (CALIBRATION_ORDER.map((name) => markers[name]).every(Boolean)) annotationHomography(markers);
  } catch (error) {
    errors.push(error.message);
  }
  return errors;
}

export function buildRecord(state, overrides = {}) {
  if (validateAnnotation(state).length) throw new Error("Resolve validation errors before saving");
  const { markers, outcome } = state;
  const contact = outcome === "save";
  const ball = projectMarker(markers, "ball_crossing");
  const keeperStrike = projectMarker(markers, "keeper_strike");
  const keeperCrossing = projectMarker(markers, "keeper_crossing");
  // The crossing/contact frame is the terminal event, so for a save this is the
  // time to contact and for a non-contact shot it is the censoring limit.
  const flightTime = (state.crossingFrame - state.strikeFrame) / state.fps;
  return {
    id: overrides.id ?? crypto.randomUUID(),
    keeper_id: state.keeperId.trim(),
    source_url: String(state.sourceUrl || "").trim(),
    clip_file: state.clipFile.name,
    fps: state.fps,
    strike_frame: state.strikeFrame,
    crossing_frame: state.crossingFrame,
    flight_time_s: flightTime,
    ball_crossing_xy_m: ball,
    keeper_pos_at_strike_xy_m: keeperStrike,
    keeper_pos_at_crossing_xy_m: keeperCrossing,
    displacement_m: [keeperCrossing[0] - keeperStrike[0], keeperCrossing[1] - keeperStrike[1]],
    contact,
    contact_xy_m: contact ? projectMarker(markers, "contact_location") : null,
    time_to_contact_s: contact ? flightTime : null,
    censored: !contact,
    outcome,
    dive_direction: state.diveDirection,
    contact_body_part: String(state.bodyPart || "").trim() || null,
    quality_flags: state.qualityFlags || [],
    annotator: state.annotator.trim(),
    annotated_at: overrides.annotatedAt ?? new Date().toISOString(),
    annotation_metadata: {
      homography_pixel_to_goal: annotationHomography(markers),
      goal_reference_pixels: Object.fromEntries(CALIBRATION_ORDER.map((name) => [name, markers[name]])),
      fps_metadata: state.fpsMetadata,
    },
  };
}
