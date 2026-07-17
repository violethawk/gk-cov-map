function readType(view, offset) {
  return String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1), view.getUint8(offset + 2), view.getUint8(offset + 3));
}

function boxes(view, start, end) {
  const result = [];
  let offset = start;
  while (offset + 8 <= end) {
    let size = view.getUint32(offset);
    const type = readType(view, offset + 4);
    let header = 8;
    if (size === 1) {
      const high = view.getUint32(offset + 8);
      const low = view.getUint32(offset + 12);
      size = high * 2 ** 32 + low;
      header = 16;
    } else if (size === 0) {
      size = end - offset;
    }
    if (size < header || offset + size > end) break;
    result.push({ type, start: offset, dataStart: offset + header, end: offset + size });
    offset += size;
  }
  return result;
}

function child(parent, view, type) {
  return boxes(view, parent.dataStart, parent.end).find((box) => box.type === type);
}

function handlerType(view, mdia) {
  const hdlr = child(mdia, view, "hdlr");
  if (!hdlr || hdlr.dataStart + 12 > hdlr.end) return null;
  return readType(view, hdlr.dataStart + 8);
}

function mediaTimescale(view, mdia) {
  const mdhd = child(mdia, view, "mdhd");
  if (!mdhd) throw new Error("MP4 video track has no mdhd box");
  const version = view.getUint8(mdhd.dataStart);
  const offset = version === 1 ? mdhd.dataStart + 20 : mdhd.dataStart + 12;
  if (offset + 4 > mdhd.end) throw new Error("Malformed mdhd box");
  return view.getUint32(offset);
}

function sampleTiming(view, mdia) {
  const minf = child(mdia, view, "minf");
  const stbl = minf && child(minf, view, "stbl");
  const stts = stbl && child(stbl, view, "stts");
  if (!stts) throw new Error("MP4 video track has no stts timing table");
  let offset = stts.dataStart + 4;
  const entryCount = view.getUint32(offset);
  offset += 4;
  let samples = 0;
  let ticks = 0;
  const deltas = new Set();
  for (let i = 0; i < entryCount; i += 1) {
    if (offset + 8 > stts.end) throw new Error("Malformed stts timing table");
    const count = view.getUint32(offset);
    const delta = view.getUint32(offset + 4);
    samples += count;
    ticks += count * delta;
    deltas.add(delta);
    offset += 8;
  }
  if (samples <= 0 || ticks <= 0) throw new Error("Empty MP4 timing table");
  return { samples, ticks, variableFrameRate: deltas.size > 1 };
}

export function parseMp4Fps(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const top = boxes(view, 0, view.byteLength);
  const moov = top.find((box) => box.type === "moov");
  if (!moov) throw new Error("No MP4 moov metadata found");
  const tracks = boxes(view, moov.dataStart, moov.end).filter((box) => box.type === "trak");
  for (const track of tracks) {
    const mdia = child(track, view, "mdia");
    if (!mdia || handlerType(view, mdia) !== "vide") continue;
    const timescale = mediaTimescale(view, mdia);
    const timing = sampleTiming(view, mdia);
    return {
      fps: (timing.samples * timescale) / timing.ticks,
      variableFrameRate: timing.variableFrameRate,
      sampleCount: timing.samples,
      timingSource: "MP4 mdhd timescale + stts sample table",
    };
  }
  throw new Error("No MP4 video track found");
}
