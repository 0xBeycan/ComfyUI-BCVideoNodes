import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { PlayerWidget, drawMessage, drawTag, fitRect, guard, installPlayer, viewUrl } from "./player.js";

// Load Video: the resolution labels of the chosen model, the upload of a video into input/, a
// video dragged from the queue or the media assets panel, the preview of the source file as the
// loader will take it, and what an empty force_fps or frame_count stands for.
//
// A video dragged from the queue or the media assets panel is selected where it is, as the
// frontend's Load Image selects an image dragged from there: the widget gets core's annotated
// file path, "name [output]" for a file of output/, "name [temp]" for one of temp/ (a file of
// input/ keeps its bare name), which the loader, its validation and PLAN_ROUTE resolve inside that
// folder only. The node claims such a drag; without that the canvas takes the drop and opens the
// workflow the video carries.
//
// resolution offers the labels of the chosen model; the labels come with the node definition (the
// resolution input's "bcv_sizes": model -> label -> [width, height], portrait; null for source,
// the video's own pixels, cut and never resized).
//
// What the loader will load comes from the server (PLAN_ROUTE, nodes/video_input.py), which
// answers from the loader's own checks and frame selection without loading a frame: the loaded
// size and frame count, the source's frame rate, the frames an empty frame_count stands for, and
// the loader's error, word for word. The node asks when the file or a widget changes (after
// ASK_DELAY of quiet); the label and the error are shown only for an answer to the current
// values, so the preview never shows a count the loader would not load. Without an answer (no
// route, a failed request) there is no label.
//
// The preview plays the source file. With force_fps the picture is sampled at that rate (the frame
// under the playhead at each tick); playback loops over the loaded frames' time span; the part the
// crop to the model's size cuts away is dimmed. The browser's frame at a tick can differ by one
// from the loader's.

const NODE = "BCVLoadVideo";
const PLAN_ROUTE = "/bcvideonodes/load_video/plan";
// the widgets the route is asked with: Load Video's inputs, in order
const PLAN_PARAMS = ["video", "model", "resolution", "orientation", "force_fps", "start_frame", "frame_count", "precision"];
// the drag data of an item of the queue or the media assets panel: JSON {filename, subfolder,
// type, media_kind, ...}, as the frontend's own nodes read it
const ASSET_INFO = "application/x-comfy-asset-info";
// the folders a widget value may name (core's folder_paths.annotated_filepath)
const FOLDERS = ["input", "output", "temp"];
// ms without a widget change before the route is asked
const ASK_DELAY = 120;

function widget(node, name) {
	return node.widgets?.find((w) => w.name === name);
}

// ---- resolution by model -------------------------------------------------------------------

// The labels of `model` that have a size (source has none).
function sized(sizes, model) {
	return Object.entries(sizes?.[model] ?? {}).filter(([, size]) => size).map(([label]) => label);
}

// Narrows resolution to the model's labels. On a model change the label stays when the new model
// has it (720p, source), else it keeps its place among the sized labels (480p <-> 512p, 720p <->
// 704p, 1080p -> the largest); a loaded workflow keeps its value.
function applyModel(node, sizes, previousModel) {
	const model = widget(node, "model")?.value;
	const resolution = widget(node, "resolution");
	const labels = Object.keys(sizes?.[model] ?? {});
	if (!resolution || !labels.length) return;
	resolution.options.values = labels;
	if (previousModel !== undefined && previousModel !== model && !labels.includes(resolution.value)) {
		const place = sized(sizes, previousModel).indexOf(resolution.value);
		const next = sized(sizes, model);
		resolution.value = next.length ? next[Math.min(Math.max(place, 0), next.length - 1)] : labels[0];
	}
	node.setDirtyCanvas?.(true, false);
}

// ---- upload --------------------------------------------------------------------------------

function notify(summary, detail) {
	const toast = app.extensionManager?.toast;
	if (toast?.add) toast.add({ severity: "error", summary, detail, life: 6000 });
	else alert(`${summary}\n${detail}`);
}

// Adds `value` to the video widget's choices when it is not one of them and selects it.
function select(node, value) {
	const video = widget(node, "video");
	if (!video.options.values.includes(value)) video.options.values.push(value);
	video.value = value;
	video.callback?.(value);
	node.setDirtyCanvas?.(true, false);
}

// Uploads `file` into input/ (the endpoint the frontend's own uploads use) and selects it.
async function upload(node, file) {
	const body = new FormData();
	body.append("image", file);
	body.append("type", "input");
	const resp = await api.fetchApi("/upload/image", { method: "POST", body });
	if (resp.status !== 200) {
		notify("Load Video: upload failed", `${file.name}: ${resp.status} ${resp.statusText}`);
		return false;
	}
	const { name, subfolder } = await resp.json();
	select(node, subfolder ? `${subfolder}/${name}` : name);
	return true;
}

// The asset info of a video dragged from the queue or the media assets panel, or null.
function draggedVideo(event) {
	try {
		const asset = JSON.parse(event?.dataTransfer?.getData(ASSET_INFO) || "null");
		return asset?.filename && asset.media_kind === "video" && FOLDERS.includes(asset.type || "input") ? asset : null;
	} catch {
		return null; // not JSON
	}
}

// The widget value of an asset: its path in its folder, annotated outside input/ (as the
// frontend writes it for Load Image).
function assetValue({ filename, subfolder, type }) {
	const path = subfolder ? `${subfolder}/${filename}` : filename;
	return type && type !== "input" ? `${path} [${type}]` : path;
}

function chooseFile(node) {
	const input = document.createElement("input");
	input.type = "file";
	input.accept = "video/*";
	input.onchange = () => input.files?.length && upload(node, input.files[0]);
	input.click();
}

// ---- placeholders --------------------------------------------------------------------------

// An empty force_fps or frame_count shows, greyed, what empty stands for. The text sits in the
// widget's options.placeholder, never in its value, which stays "" (that is what the prompt and
// the saved workflow get). The Vue-nodes renderer binds a widget's options to its <input>, so it
// shows the text as the input's own placeholder. On the classic canvas a text widget draws
// `_displayValue` in `text_color`: while the value is empty, those give the placeholder in the
// theme's disabled text colour, so the frontend's own layout places and cuts it like a value.

function protoGetter(object, name) {
	for (let p = Object.getPrototypeOf(object); p; p = Object.getPrototypeOf(p)) {
		const getter = Object.getOwnPropertyDescriptor(p, name)?.get;
		if (getter) return getter;
	}
	return null;
}

function installPlaceholder(w) {
	const value = w && protoGetter(w, "_displayValue");
	const color = w && protoGetter(w, "text_color");
	if (!value || !color) return; // another canvas widget: the Vue renderer still shows the placeholder
	const shown = () => !w.computedDisabled && String(w.value ?? "").trim() === "" && !!w.options?.placeholder;
	Object.defineProperty(w, "_displayValue", { configurable: true, get: () => (shown() ? String(w.options.placeholder) : value.call(w)) });
	Object.defineProperty(w, "text_color", { configurable: true, get: () => (shown() ? w.disabledTextColor : color.call(w)) });
}

function setPlaceholder(w, text) {
	const placeholder = text == null ? undefined : String(text);
	if (w?.options && w.options.placeholder !== placeholder) w.options.placeholder = placeholder;
}

// ---- preview -------------------------------------------------------------------------------

// [the path inside its folder, the folder] of a widget value, read as core reads it
// (folder_paths.annotated_filepath: a value ending in "[output]", "[input]" or "[temp]" loses that
// and the character before it; input/ without one).
function folderOf(value) {
	const type = FOLDERS.find((folder) => value.endsWith(`[${folder}]`));
	return type ? [value.slice(0, -(type.length + 3)), type] : [value, "input"];
}

function sourceUrl(value) {
	if (!value) return null;
	const [path, type] = folderOf(value);
	const cut = path.lastIndexOf("/");
	return viewUrl({ filename: path.slice(cut + 1), subfolder: cut < 0 ? "" : path.slice(0, cut), type });
}

// The centred region of a width x height frame with the target's aspect; the cut side rounded,
// as the loader cuts it.
function cropBox(width, height, tw, th) {
	const aspect = width / height;
	const target = tw / th;
	const [w, h] = aspect > target ? [Math.round(height * target), height] : [width, Math.round(width / target)];
	return [Math.floor((width - w) / 2), Math.floor((height - h) / 2), w, h];
}

// The region a resolution source load keeps, in a width x height picture of the source: the centred
// loaded_width x loaded_height of the source's pixels, as the loader cuts it (no resize).
function cutBox(width, height, info) {
	const sx = width / info.source_width;
	const sy = height / info.source_height;
	const x = Math.floor((info.source_width - info.loaded_width) / 2);
	const y = Math.floor((info.source_height - info.loaded_height) / 2);
	return [x * sx, y * sy, info.loaded_width * sx, info.loaded_height * sy];
}

function rate(fps) {
	return String(+fps.toFixed(3));
}

// `text` in ctx's current font, broken into lines of at most `width` pixels at the spaces.
function wrap(ctx, text, width) {
	const lines = [];
	let line = "";
	for (const word of text.split(/\s+/)) {
		const next = line ? `${line} ${word}` : word;
		if (line && ctx.measureText(next).width > width) {
			lines.push(line);
			line = word;
		} else line = next;
	}
	if (line) lines.push(line);
	return lines;
}

// The loader's error over the top of the picture box, wrapped to its width, cut to its height.
function drawError(ctx, text, [bx, by, bw, bh]) {
	const lineH = 15;
	ctx.save();
	ctx.beginPath();
	ctx.rect(bx, by, bw, bh);
	ctx.clip();
	ctx.font = "12px sans-serif";
	const lines = wrap(ctx, text, bw - 24).slice(0, Math.max(1, Math.floor((bh - 22) / lineH)));
	ctx.fillStyle = "rgba(0,0,0,.75)";
	ctx.fillRect(bx + 6, by + 6, bw - 12, lines.length * lineH + 10);
	ctx.fillStyle = "#e0b060";
	ctx.textAlign = "left";
	ctx.textBaseline = "top";
	lines.forEach((line, i) => ctx.fillText(line, bx + 12, by + 11 + i * lineH));
	ctx.restore();
}

class SourceWidget extends PlayerWidget {
	constructor(node, sizes) {
		super(node, "preview");
		this.sizes = sizes; // bcv_sizes: a null size is source, which cuts instead of resizing
		this.asked = null; // the query of the widget values last asked about ("" without a file)
		this.timer = null;
		this.answer = null; // the route's answer to `answered`, or null
		this.answered = null;
	}

	// The widget values as the route's query; "" when no file is chosen.
	query() {
		if (!widget(this.node, "video")?.value) return "";
		return new URLSearchParams(PLAN_PARAMS.map((name) => [name, String(widget(this.node, name)?.value ?? "")])).toString();
	}

	// Called on every paint: loads the chosen file and, when the values changed, asks the route
	// once they stay unchanged for ASK_DELAY. Another file drops the last answer at once.
	sync() {
		const video = widget(this.node, "video")?.value;
		const url = sourceUrl(video);
		if (url !== this.player.url) this.apply(null, null);
		this.player.load(url);
		const query = this.query();
		if (query === this.asked) return;
		this.asked = query;
		clearTimeout(this.timer);
		if (query) this.timer = setTimeout(() => guard(() => this.ask(query)), ASK_DELAY);
	}

	async ask(query) {
		let answer = null;
		try {
			const resp = await api.fetchApi(`${PLAN_ROUTE}?${query}`);
			if (resp.ok) answer = await resp.json();
			else console.warn(`[BCVideoNodes] Load Video: ${PLAN_ROUTE} answered ${resp.status}: ${await resp.text()}`);
		} catch (e) {
			console.warn(`[BCVideoNodes] Load Video: ${PLAN_ROUTE} failed:`, e);
		}
		if (query === this.asked) guard(() => this.apply(query, answer)); // else a newer question is on its way
	}

	// Takes `answer` (to `query`): the playback range and sampling, the placeholders.
	apply(query, answer) {
		this.answer = answer;
		this.answered = query;
		const info = answer?.info;
		const p = this.player;
		// force_fps samples the picture only when the loader keeps frames on its grid
		p.setStep(info && info.loaded_fps !== info.source_fps ? info.loaded_fps : null);
		// the loaded frames' time span, as the loader cuts the audio
		const start = info ? (Number(new URLSearchParams(query).get("start_frame")) - 1) / info.loaded_fps : 0;
		p.setRange(info ? [start, start + info.loaded_duration] : null);
		setPlaceholder(widget(this.node, "force_fps"), answer?.source ? rate(answer.source.fps) : null);
		setPlaceholder(widget(this.node, "frame_count"), answer?.available);
		this.node.setDirtyCanvas?.(true, false);
	}

	// The answer to the current widget values, or null while a newer one is pending.
	current() {
		return this.answered === this.asked ? this.answer : null;
	}

	controls() {
		return { audio: !!this.answer?.source?.audio };
	}

	paint(ctx, box) {
		const p = this.player;
		const pic = p.picture();
		const [sw, sh] = p.size();
		const rect = pic ? fitRect(sw, sh, ...box) : null;
		const current = this.current();
		if (rect) ctx.drawImage(pic, ...rect);
		if (current?.error) {
			drawError(ctx, current.error, box);
			return;
		}
		if (!rect) {
			const empty = !widget(this.node, "video")?.value;
			drawMessage(ctx, p.failed ? "the browser cannot play this file" : empty ? "choose a video" : "loading…", box);
			return;
		}
		// the crop of the last answer while a newer one is pending (same file: another file drops it)
		const info = this.answer?.info;
		if (!info) return;
		// dim what the crop cuts away, outline what it keeps
		const asked = new URLSearchParams(this.answered ?? "");
		const cut = this.sizes?.[asked.get("model")]?.[asked.get("resolution")] === null;
		const [cx, cy, cw, ch] = cut ? cutBox(sw, sh, info) : cropBox(sw, sh, info.loaded_width, info.loaded_height);
		const s = rect[2] / sw;
		const [kx, ky, kw, kh] = [rect[0] + cx * s, rect[1] + cy * s, cw * s, ch * s];
		ctx.fillStyle = "rgba(0,0,0,.55)";
		ctx.fillRect(rect[0], rect[1], rect[2], ky - rect[1]);
		ctx.fillRect(rect[0], ky + kh, rect[2], rect[1] + rect[3] - ky - kh);
		ctx.fillRect(rect[0], ky, kx - rect[0], kh);
		ctx.fillRect(kx + kw, ky, rect[0] + rect[2] - kx - kw, kh);
		ctx.strokeStyle = "rgba(255,255,255,.8)";
		ctx.lineWidth = 1;
		ctx.strokeRect(kx + 0.5, ky + 0.5, kw - 1, kh - 1);
		const size = `${info.loaded_width}x${info.loaded_height}`;
		const label = current?.info ? ` · ${current.info.loaded_frame_count} frames @ ${rate(current.info.loaded_fps)} fps` : "";
		drawTag(ctx, size + label, box[0] + 8, box[1] + 8, "left", box[2] - 16);
	}
}

app.registerExtension({
	name: "BCVideoNodes.LoadVideo",
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE) return;
		const sizes = nodeData.input?.required?.resolution?.[1]?.bcv_sizes;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const r = onNodeCreated?.apply(this, args);
			const model = widget(this, "model");
			if (model) {
				let previous = model.value;
				const callback = model.callback;
				model.callback = (...a) => {
					const out = callback?.apply(model, a);
					guard(() => applyModel(this, sizes, previous));
					previous = model.value;
					return out;
				};
			}
			guard(() => applyModel(this, sizes));
			guard(() => ["force_fps", "frame_count"].forEach((name) => installPlaceholder(widget(this, name))));
			// The upload button after the inputs' widgets (so the saved widget values stay in input
			// order), unless the frontend added its own for this input.
			if (!widget(this, "upload")) {
				const button = this.addWidget("button", "choose video to upload", null, () => chooseFile(this), { serialize: false });
				button.serialize = false;
				this.bcvUpload = true;
			}
			return r;
		};

		// wraps the onNodeCreated above, which runs first: the preview comes after the upload button
		// and spans the rest of the node
		installPlayer(nodeType, (node) => new SourceWidget(node, sizes), [320, 560]);

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function (...args) {
			const r = onConfigure?.apply(this, args);
			guard(() => applyModel(this, sizes));
			return r;
		};

		// A video file dropped on the node is uploaded and selected; a video dragged from the queue
		// or the media assets panel is selected where it is. The drag's data can be read only on the
		// drop: the node claims every file or asset drag, and a drop it does not take (not a
		// video) goes on to the canvas.
		const onDragOver = nodeType.prototype.onDragOver;
		nodeType.prototype.onDragOver = function (e) {
			const types = e?.dataTransfer?.types;
			if (this.bcvUpload && (types?.includes?.("Files") || types?.includes?.(ASSET_INFO))) return true;
			return onDragOver?.apply(this, arguments) ?? false;
		};
		const onDragDrop = nodeType.prototype.onDragDrop;
		nodeType.prototype.onDragDrop = async function (e) {
			if (this.bcvUpload) {
				const file = [...(e?.dataTransfer?.files ?? [])].find((f) => f.type.startsWith("video/"));
				if (file) return await upload(this, file);
				const asset = draggedVideo(e);
				if (asset) {
					select(this, assetValue(asset));
					return true;
				}
			}
			return (await onDragDrop?.apply(this, arguments)) ?? false;
		};
	},
});
