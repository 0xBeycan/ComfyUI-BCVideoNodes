import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// The pack's video player, drawn straight onto the node canvas so it moves with the node and
// never lags behind it. Used by the Save Video and Load Video previews and the Video Comparer.
//
// One hidden <video> per node. Nothing plays until asked: the play button at the left of the top
// row or a click on the picture toggles playback, the seek bar at the bottom scrubs, the clip
// loops, a speaker button mutes it when it has sound. The picture is letterboxed inside the
// node; the node keeps whatever size the user gives it.
//
// Two options for Load Video: `range` loops playback inside [start, end] seconds (the seek bar
// then spans the range), and `step` samples the picture at that many frames per second: the frame
// under the playhead at each tick, as a loader that keeps or drops source frames would see it.

export const TOP_H = 22;
export const BAR_H = 22;
const TAG_FONT = "bold 12px sans-serif";

export function viewUrl(info) {
	const q = new URLSearchParams({ filename: info.filename, subfolder: info.subfolder ?? "", type: info.type ?? "temp" });
	return api.apiURL(`/view?${q}`);
}

// The node's last output, keyed the way the frontend stores it: the node id at the root,
// "<subgraph id>:<node id>" inside a subgraph. Filled by the "executed" event, restored with the
// workflow tab and by a click in the queue history, empty after a restart.
export function storedOutput(node) {
	const g = node.graph;
	const key = !g || g === (g.rootGraph ?? g) ? String(node.id) : `${g.id}:${node.id}`;
	return app.nodeOutputs?.[key];
}

// Letterbox [w, h] into the box [x, y, bw, bh].
export function fitRect(w, h, x, y, bw, bh) {
	if (!w || !h) return null;
	const scale = Math.min(bw / w, bh / h);
	return [x + (bw - w * scale) / 2, y + (bh - h * scale) / 2, w * scale, h * scale];
}

export function drawTag(ctx, text, x, y, align) {
	ctx.font = TAG_FONT;
	ctx.textBaseline = "top";
	ctx.textAlign = align;
	const w = ctx.measureText(text).width + 10;
	ctx.fillStyle = "rgba(0,0,0,.6)";
	ctx.fillRect(align === "left" ? x - 2 : x - w + 2, y - 2, w, 17);
	ctx.fillStyle = "#fff";
	ctx.fillText(text, align === "left" ? x + 3 : x - 3, y);
}

export function drawMessage(ctx, text, [bx, by, bw, bh]) {
	ctx.fillStyle = "#666";
	ctx.font = "12px sans-serif";
	ctx.textAlign = "center";
	ctx.textBaseline = "middle";
	ctx.fillText(text, bx + bw / 2, by + bh / 2);
}

// Never let a widget problem escape into the canvas or graph teardown: an exception in onRemoved
// aborts LGraph.clear() half way and corrupts the next workflow.
export function guard(fn) {
	try {
		return fn();
	} catch (e) {
		console.error("[BCVideoNodes] player:", e);
	}
}

export class Player {
	constructor(node) {
		this.node = node;
		this.el = null;
		this.url = null;
		this.failed = false;
		this.muted = false;
		this.raf = null;
		this.hits = [];
		this.range = null; // [start, end] seconds, or null for the whole clip
		this.step = null; // frames per second the picture is sampled at, or null for every frame
		this.still = null; // the sampled picture while step is set
		this.tick = -1;
		this.frameTimes = []; // media times of presented frames, to measure the frame rate
	}

	dirty() {
		this.node.setDirtyCanvas?.(true, false);
	}

	// ---- source ---------------------------------------------------------------------------

	load(url) {
		if (url === this.url) return;
		this.unload();
		this.url = url ?? null;
		if (!url) return;
		const v = document.createElement("video");
		v.playsInline = true;
		v.preload = "auto";
		v.loop = true;
		v.muted = this.muted;
		v.addEventListener("loadeddata", () => this.resample());
		v.addEventListener("seeked", () => this.resample());
		v.addEventListener("error", () => {
			this.failed = true;
			this.dirty();
		});
		v.src = url;
		this.el = v;
	}

	// A source that failed to load (a temp file gone after a restart, typically) is drawn as
	// "not available" instead of loading forever; its element is released either way.
	unload() {
		this.stop();
		if (this.el) {
			this.el.removeAttribute("src");
			this.el.load();
		}
		this.el = null;
		this.url = null;
		this.failed = false;
		this.still = null;
		this.tick = -1;
		this.frameTimes = [];
	}

	// The <video> once it has metadata, else null.
	clip() {
		return this.el && this.el.readyState >= 1 ? this.el : null;
	}

	size() {
		const v = this.clip();
		return v ? [v.videoWidth, v.videoHeight] : [0, 0];
	}

	pending() {
		return !!this.url && !this.failed && !this.clip();
	}

	// ---- playback range and sampling --------------------------------------------------------

	setRange(range) {
		const same = range && this.range && range[0] === this.range[0] && range[1] === this.range[1];
		if (same || (!range && !this.range)) return;
		this.range = range;
		const v = this.clip();
		if (v && range) v.currentTime = range[0];
		this.resample();
	}

	setStep(fps) {
		if (fps === this.step) return;
		this.step = fps || null;
		this.resample();
	}

	bounds() {
		const d = this.clip()?.duration || 0;
		if (!this.range) return [0, d];
		return [Math.min(this.range[0], d), Math.min(this.range[1], d)];
	}

	resample() {
		this.tick = -1;
		this.dirty();
	}

	// Called on every paint: keeps playback inside the range and, with a step, takes the frame
	// under the playhead into the still when the playhead enters a new tick.
	update() {
		const v = this.clip();
		if (!v) return;
		const [start, end] = this.bounds();
		if (this.range && end > start && (v.currentTime < start - 1e-3 || v.currentTime >= end)) {
			v.currentTime = start;
			return;
		}
		if (!this.step || v.readyState < 2) return;
		const tick = Math.floor((v.currentTime - start) * this.step + 1e-6);
		if (tick === this.tick && this.still) return;
		this.tick = tick;
		this.still ??= document.createElement("canvas");
		if (this.still.width !== v.videoWidth || this.still.height !== v.videoHeight) {
			this.still.width = v.videoWidth;
			this.still.height = v.videoHeight;
		}
		this.still.getContext("2d").drawImage(v, 0, 0);
	}

	// What to paint: the still with a step, else the <video> itself.
	picture() {
		const v = this.clip();
		if (!v || v.readyState < 2) return null;
		return this.step && this.still ? this.still : v;
	}

	// The source's frame rate, measured from the frames the browser presents while it plays
	// (null until a few have been seen). The browser does not expose it otherwise.
	measureFrameRate() {
		const v = this.el;
		if (!v?.requestVideoFrameCallback || v.bcvMeasuring) return;
		v.bcvMeasuring = true;
		const seen = (_, meta) => {
			if (v !== this.el) return;
			this.frameTimes.push(meta.mediaTime);
			if (this.frameTimes.length > 32) this.frameTimes.shift();
			v.requestVideoFrameCallback(seen);
		};
		v.requestVideoFrameCallback(seen);
	}

	frameRate() {
		const t = this.frameTimes;
		const gaps = [];
		for (let i = 1; i < t.length; i++) if (t[i] > t[i - 1]) gaps.push(t[i] - t[i - 1]);
		if (gaps.length < 4) return null;
		// the smallest gap is one frame; larger ones are frames the display skipped
		return 1 / Math.min(...gaps);
	}

	// ---- controls ------------------------------------------------------------------------

	// Top row: play / pause at the left, the time, the speaker when there is sound, a note at the
	// right.
	drawTopRow(ctx, y, width, { audio = false, note = "" } = {}) {
		this.hits = [];
		const v = this.clip();
		if (!v) return;
		const [start, end] = this.bounds();
		const t = Math.max(0, (v.currentTime || 0) - start);
		const cy = y + TOP_H / 2;

		ctx.fillStyle = "#ddd";
		const cx = 18;
		if (!v.paused) {
			ctx.fillRect(cx - 5, cy - 6, 3, 12);
			ctx.fillRect(cx + 2, cy - 6, 3, 12);
		} else {
			ctx.beginPath();
			ctx.moveTo(cx - 5, cy - 6);
			ctx.lineTo(cx + 6, cy);
			ctx.lineTo(cx - 5, cy + 6);
			ctx.fill();
		}
		this.hits.push({ x: 4, y: y - 2, w: 30, h: TOP_H + 2, action: () => this.toggle() });

		ctx.font = "11px sans-serif";
		ctx.textBaseline = "middle";
		ctx.textAlign = "left";
		ctx.fillStyle = "#bbb";
		const time = `${t.toFixed(2)} / ${(end - start).toFixed(2)}`;
		ctx.fillText(time, 34, cy);

		if (audio) {
			const sx = 34 + ctx.measureText(time).width + 12;
			this.drawSpeaker(ctx, sx, cy);
			this.hits.push({ x: sx - 4, y: y - 2, w: 24, h: TOP_H + 2, action: () => this.toggleMute() });
		}
		if (note) {
			ctx.textAlign = "right";
			ctx.fillStyle = "#e0b060";
			ctx.font = "10px sans-serif";
			ctx.fillText(note, width - 6, cy);
		}
	}

	// A small speaker: body + cone, two arcs when sounding, a slash when muted.
	drawSpeaker(ctx, x, cy) {
		ctx.fillStyle = this.muted ? "#777" : "#ddd";
		ctx.strokeStyle = ctx.fillStyle;
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		ctx.moveTo(x, cy - 3);
		ctx.lineTo(x + 3, cy - 3);
		ctx.lineTo(x + 7, cy - 6);
		ctx.lineTo(x + 7, cy + 6);
		ctx.lineTo(x + 3, cy + 3);
		ctx.lineTo(x, cy + 3);
		ctx.closePath();
		ctx.fill();
		if (this.muted) {
			ctx.beginPath();
			ctx.moveTo(x + 9, cy - 4);
			ctx.lineTo(x + 15, cy + 4);
			ctx.stroke();
		} else {
			for (const r of [4, 7]) {
				ctx.beginPath();
				ctx.arc(x + 7, cy, r, -Math.PI / 3, Math.PI / 3);
				ctx.stroke();
			}
		}
	}

	// Bottom: the seek bar alone, full width, over the range.
	drawSeekBar(ctx) {
		const v = this.clip();
		if (!v) return;
		const [w, h] = this.node.size;
		const [start, end] = this.bounds();
		const d = end - start;
		const t = Math.min(Math.max((v.currentTime || 0) - start, 0), d);
		const top = h - BAR_H;
		const cy = top + BAR_H / 2;
		const tx = 10;
		const tw = w - 20;
		const px = tx + (d > 0 ? (tw * t) / d : 0);
		ctx.fillStyle = "#333";
		ctx.fillRect(tx, cy - 2, tw, 4);
		ctx.fillStyle = "#4a90e2";
		ctx.fillRect(tx, cy - 2, px - tx, 4);
		ctx.beginPath();
		ctx.arc(px, cy, 5, 0, Math.PI * 2);
		ctx.fillStyle = "#e6e6e6";
		ctx.fill();
		this.hits.push({ x: 4, y: top, w: w - 8, h: BAR_H, seek: (x) => this.seek(start + ((Math.min(Math.max(x, tx), tx + tw) - tx) / tw) * d) });
	}

	// ---- pointer ---------------------------------------------------------------------------

	hitAt(pos) {
		return this.hits.find((h) => pos[0] >= h.x && pos[0] <= h.x + h.w && pos[1] >= h.y && pos[1] <= h.y + h.h);
	}

	// A click on a control runs it; a click on the picture (inPicture) toggles playback, like a
	// video player. Returns whether the click was taken.
	click(pos, inPicture) {
		const h = this.hitAt(pos);
		if (h) {
			if (h.seek) h.seek(pos[0]);
			else h.action?.();
			return true;
		}
		if (inPicture && this.clip()) {
			this.toggle();
			return true;
		}
		return false;
	}

	// Dragging with the button held scrubs when the pointer is on the seek bar.
	drag(event, pos) {
		if (!(event.buttons & 1)) return;
		const h = this.hitAt(pos);
		if (h?.seek) h.seek(pos[0]);
	}

	// ---- playback --------------------------------------------------------------------------

	toggle() {
		const v = this.clip();
		if (!v) return;
		if (!v.paused) return this.stop();
		v.muted = this.muted;
		v.play().catch(() => {});
		// Repaint on every frame while it plays.
		const step = () => {
			this.dirty();
			this.raf = v.paused || v !== this.el ? null : requestAnimationFrame(step);
		};
		if (this.raf) cancelAnimationFrame(this.raf);
		this.raf = requestAnimationFrame(step);
	}

	stop() {
		this.el?.pause();
		if (this.raf) cancelAnimationFrame(this.raf);
		this.raf = null;
		this.dirty();
	}

	seek(t) {
		const v = this.clip();
		if (v) v.currentTime = Math.max(0, t);
		this.dirty();
	}

	toggleMute() {
		this.muted = !this.muted;
		if (this.el) this.el.muted = this.muted;
		this.dirty();
	}
}

// A canvas widget that spans the node below the node's other widgets: litegraph gives it a small
// row, and `draw` paints from that row down to the node's bottom edge, so the preview scales
// with the node and never makes the node grow. `paint(ctx, box)` draws the picture.
export class PlayerWidget {
	constructor(node, name) {
		this.type = "BCV_PLAYER";
		this.name = name;
		this.node = node;
		this.value = null;
		this.serialize = false; // not written to the workflow file
		this.options = { serialize: false }; // not sent in the prompt
		this.player = new Player(node);
	}

	computeSize(width) {
		return [width, TOP_H];
	}

	// The picture box: below the top row, above the seek bar.
	box(y) {
		const [w, h] = this.node.size;
		const top = y + TOP_H;
		return [4, top, w - 8, Math.max(20, h - BAR_H - top)];
	}

	inBox(pos) {
		if (typeof this.y !== "number") return false;
		const [bx, by, bw, bh] = this.box(this.y);
		return pos[0] >= bx && pos[0] <= bx + bw && pos[1] >= by && pos[1] <= by + bh;
	}

	draw(ctx, node, width, y) {
		// On the node canvas the widget spans the node. The Vue-nodes legacy renderer leaves its
		// own width on the widget, which would otherwise stick after switching back.
		if (ctx.canvas === app.canvas?.canvas) width = node.size[0];
		ctx.save();
		guard(() => {
			this.sync?.();
			this.player.update();
			this.player.drawTopRow(ctx, y, width, this.controls?.() ?? {});
			const box = this.box(y);
			ctx.fillStyle = "#111";
			ctx.fillRect(...box);
			this.paint(ctx, box);
			this.player.drawSeekBar(ctx);
		});
		ctx.restore();
	}

	paint(ctx, box) {}

	// Clicks inside the widget's own row arrive here; the rest come from the node's onMouseDown.
	mouse(event, pos) {
		if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
		return this.player.click(pos, this.inBox(pos));
	}
}

// Wires a PlayerWidget into a node type: created with the node (at a minimum size for a new node;
// a loaded node keeps its saved size), clicks and drags routed from anywhere on the node, the
// clip stopped and released when the node is removed. `make(node)` returns the widget.
export function installPlayer(nodeType, make, minSize = [320, 300]) {
	const onNodeCreated = nodeType.prototype.onNodeCreated;
	nodeType.prototype.onNodeCreated = function (...args) {
		const r = onNodeCreated?.apply(this, args);
		this.bcvPlayer = make(this);
		this.addCustomWidget(this.bcvPlayer);
		this.setSize([Math.max(this.size[0], minSize[0]), Math.max(this.size[1], minSize[1])]);
		return r;
	};

	const onMouseDown = nodeType.prototype.onMouseDown;
	nodeType.prototype.onMouseDown = function (event, pos, ...rest) {
		const w = this.bcvPlayer;
		if (w && guard(() => w.player.click(pos, w.inBox(pos)))) return true;
		return onMouseDown?.apply(this, [event, pos, ...rest]);
	};

	const onMouseMove = nodeType.prototype.onMouseMove;
	nodeType.prototype.onMouseMove = function (event, pos, ...rest) {
		const r = onMouseMove?.apply(this, [event, pos, ...rest]);
		const w = this.bcvPlayer;
		if (w) guard(() => {
			w.hover?.(w.inBox(pos) ? pos[0] : null);
			w.player.drag(event, pos);
		});
		return r;
	};

	const onMouseEnter = nodeType.prototype.onMouseEnter;
	nodeType.prototype.onMouseEnter = function (event, pos, ...rest) {
		const r = onMouseEnter?.apply(this, [event, pos, ...rest]);
		const w = this.bcvPlayer;
		if (w && pos) guard(() => w.hover?.(w.inBox(pos) ? pos[0] : null));
		return r;
	};

	const onMouseLeave = nodeType.prototype.onMouseLeave;
	nodeType.prototype.onMouseLeave = function (...args) {
		const r = onMouseLeave?.apply(this, args);
		guard(() => this.bcvPlayer?.hover?.(null));
		return r;
	};

	const onRemoved = nodeType.prototype.onRemoved;
	nodeType.prototype.onRemoved = function (...args) {
		guard(() => this.bcvPlayer?.player.unload());
		return onRemoved?.apply(this, args);
	};
}
