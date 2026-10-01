import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { PlayerWidget, drawMessage, drawTag, fitRect, guard, installPlayer } from "./player.js";

// Load Video: the resolution labels of the chosen model, the upload of a video into input/, and
// the preview of the source file as the loader will take it. All in the browser, from the file
// itself: no server work.
//
// resolution offers the labels of the chosen model; the sizes come with the node definition (the
// resolution input's "bcv_sizes": model -> label -> [width, height], portrait; and "bcv_frames":
// model -> n of its n*k + 1 frame rule).
//
// The preview plays the source file. With force_fps the picture is sampled at that rate (the frame
// under the playhead at each tick); playback loops over start_frame / frame_count; the part the
// crop to the model's size cuts away is dimmed. The browser's frame at a tick can differ by one
// from the loader's. Without force_fps the source's frame rate is needed to place start_frame and
// frame_count, and the browser only tells it while the clip plays: until then the whole clip loops.

const NODE = "BCVLoadVideo";

function widget(node, name) {
	return node.widgets?.find((w) => w.name === name);
}

// ---- resolution by model -------------------------------------------------------------------

// Narrows resolution to the model's labels. On a model change the label keeps its place in the
// list (480p <-> 512p, 720p <-> 704p); a loaded workflow keeps its value.
function applyModel(node, sizes, previousModel) {
	const model = widget(node, "model")?.value;
	const resolution = widget(node, "resolution");
	const labels = Object.keys(sizes?.[model] ?? {});
	if (!resolution || !labels.length) return;
	resolution.options.values = labels;
	if (previousModel !== undefined && previousModel !== model) {
		const place = Object.keys(sizes[previousModel] ?? {}).indexOf(resolution.value);
		resolution.value = labels[Math.min(Math.max(place, 0), labels.length - 1)];
	}
	node.setDirtyCanvas?.(true, false);
}

// ---- upload --------------------------------------------------------------------------------

function notify(summary, detail) {
	const toast = app.extensionManager?.toast;
	if (toast?.add) toast.add({ severity: "error", summary, detail, life: 6000 });
	else alert(`${summary}\n${detail}`);
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
	const value = subfolder ? `${subfolder}/${name}` : name;
	const video = widget(node, "video");
	if (!video.options.values.includes(value)) video.options.values.push(value);
	video.value = value;
	video.callback?.(value);
	node.setDirtyCanvas?.(true, false);
	return true;
}

function chooseFile(node) {
	const input = document.createElement("input");
	input.type = "file";
	input.accept = "video/*";
	input.onchange = () => input.files?.length && upload(node, input.files[0]);
	input.click();
}

// ---- preview -------------------------------------------------------------------------------

function sourceUrl(value) {
	if (!value) return null;
	const cut = value.lastIndexOf("/");
	const q = new URLSearchParams({ filename: value.slice(cut + 1), subfolder: cut < 0 ? "" : value.slice(0, cut), type: "input" });
	return api.apiURL(`/view?${q}`);
}

// The centred region of a width x height frame with the target's aspect; the cut side rounded,
// as the loader cuts it.
function cropBox(width, height, tw, th) {
	const aspect = width / height;
	const target = tw / th;
	const [w, h] = aspect > target ? [Math.round(height * target), height] : [width, Math.round(width / target)];
	return [Math.floor((width - w) / 2), Math.floor((height - h) / 2), w, h];
}

// "" -> null; else a number, or NaN when it is not one.
function parseOptional(text) {
	const t = String(text ?? "").trim();
	return t === "" ? null : Number(t);
}

class SourceWidget extends PlayerWidget {
	constructor(node, sizes, frames) {
		super(node, "preview");
		this.sizes = sizes;
		this.frames = frames;
		this.label = "";
		this.error = "";
	}

	sync() {
		const p = this.player;
		p.load(sourceUrl(widget(this.node, "video")?.value));
		p.measureFrameRate();
		const forceFps = parseOptional(widget(this.node, "force_fps")?.value);
		const startFrame = widget(this.node, "start_frame")?.value ?? 1;
		const frameCount = parseOptional(widget(this.node, "frame_count")?.value);
		this.error = "";
		if (forceFps !== null && !(forceFps > 0)) this.error = "force_fps: empty or a number > 0";
		else if (frameCount !== null && !(Number.isInteger(frameCount) && frameCount >= 1)) this.error = "frame_count: empty or a whole number >= 1";
		const fps = this.error ? null : forceFps ?? p.frameRate();
		p.setStep(this.error ? null : forceFps);

		// Frames are counted on the force_fps grid (the source's frames without it).
		const duration = p.clip()?.duration || 0;
		if (!fps || !duration) {
			p.setRange(null);
			this.label = fps || this.error ? "" : "play to measure the frame rate";
			return;
		}
		const available = Math.max(0, Math.floor(duration * fps + 1e-6) - (startFrame - 1));
		const selected = frameCount ?? available;
		if (startFrame - 1 + selected > Math.floor(duration * fps + 1e-6)) {
			this.error = `start_frame + frame_count past the video's ~${Math.floor(duration * fps + 1e-6)} frames`;
		}
		const rule = this.frames?.[widget(this.node, "model")?.value];
		const kept = rule && selected >= 1 ? rule * Math.floor((selected - 1) / rule) + 1 : 0;
		const start = (startFrame - 1) / fps;
		p.setRange([start, start + kept / fps]);
		this.label = `${kept} frames @ ${+fps.toFixed(3)} fps`;
	}

	controls() {
		return { audio: true, note: this.error };
	}

	// The model's size, oriented as the widget says (auto: as the source).
	target(sw, sh) {
		const size = this.sizes?.[widget(this.node, "model")?.value]?.[widget(this.node, "resolution")?.value];
		if (!size) return null;
		const orientation = widget(this.node, "orientation")?.value;
		const portrait = orientation === "portrait" || (orientation !== "landscape" && sh > sw);
		return portrait ? size : [size[1], size[0]];
	}

	paint(ctx, box) {
		const p = this.player;
		const pic = p.picture();
		const [sw, sh] = p.size();
		const rect = pic ? fitRect(sw, sh, ...box) : null;
		if (!rect) {
			const empty = !widget(this.node, "video")?.value;
			drawMessage(ctx, p.failed ? "the browser cannot play this file" : empty ? "choose a video" : "loading…", box);
			return;
		}
		ctx.drawImage(pic, ...rect);
		const target = this.target(sw, sh);
		if (!target) {
			drawTag(ctx, "resolution does not belong to model", box[0] + 8, box[1] + 8, "left");
			return;
		}
		// dim what the crop cuts away, outline what it keeps
		const [cx, cy, cw, ch] = cropBox(sw, sh, ...target);
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
		drawTag(ctx, `${target[0]}x${target[1]}${this.label ? " · " + this.label : ""}`, box[0] + 8, box[1] + 8, "left");
	}
}

app.registerExtension({
	name: "BCVideoNodes.LoadVideo",
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE) return;
		const sizes = nodeData.input?.required?.resolution?.[1]?.bcv_sizes;
		const frames = nodeData.input?.required?.resolution?.[1]?.bcv_frames;

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
		installPlayer(nodeType, (node) => new SourceWidget(node, sizes, frames), [320, 560]);

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function (...args) {
			const r = onConfigure?.apply(this, args);
			guard(() => applyModel(this, sizes));
			return r;
		};

		// A video file dropped on the node is uploaded and selected.
		const onDragOver = nodeType.prototype.onDragOver;
		nodeType.prototype.onDragOver = function (e) {
			if (this.bcvUpload && e?.dataTransfer?.types?.includes?.("Files")) return true;
			return onDragOver?.apply(this, arguments) ?? false;
		};
		const onDragDrop = nodeType.prototype.onDragDrop;
		nodeType.prototype.onDragDrop = async function (e) {
			const file = this.bcvUpload ? [...(e?.dataTransfer?.files ?? [])].find((f) => f.type.startsWith("video/")) : null;
			if (file) return await upload(this, file);
			return (await onDragDrop?.apply(this, arguments)) ?? false;
		};
	},
});
