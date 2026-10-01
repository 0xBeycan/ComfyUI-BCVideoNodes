import { app } from "../../../scripts/app.js";
import { PlayerWidget, drawMessage, fitRect, guard, installPlayer, storedOutput, viewUrl } from "./player.js";

// Save Video: the codec's widgets and the preview of the saved file.
//
// The values crf, preset and pix_fmt offer depend on the codec. The table lives in Python and
// comes with the node definition (the codec input's "bcv_codecs"); changing the codec narrows the
// three widgets to that codec's values and resets them to its defaults. A loaded workflow keeps
// its saved values.
//
// The preview plays the file the last run saved (the node's "bcv_video" output).

const NODE = "BCVSaveVideo";
const UI_KEY = "bcv_video";

function widget(node, name) {
	return node.widgets?.find((w) => w.name === name);
}

function applyCodec(node, table, reset) {
	const spec = table?.[widget(node, "codec")?.value];
	if (!spec) return;
	const crf = widget(node, "crf");
	if (crf) {
		crf.options.min = spec.crf.min;
		crf.options.max = spec.crf.max;
		if (reset) crf.value = spec.crf.default;
	}
	for (const name of ["preset", "pix_fmt"]) {
		const w = widget(node, name);
		if (!w) continue;
		w.options.values = [...spec[name].values];
		if (reset) w.value = spec[name].default;
	}
	node.setDirtyCanvas?.(true, false);
}

class SavedVideoWidget extends PlayerWidget {
	constructor(node) {
		super(node, "preview");
		this.seen = undefined; // the stored output the preview was built from
		this.info = null;
	}

	// Follow the stored output: a new run, a tab switch, a history click, a fresh workflow.
	sync() {
		const out = storedOutput(this.node);
		if (out === this.seen) return;
		this.seen = out;
		this.info = out?.[UI_KEY]?.[0] ?? null;
		this.player.load(this.info ? viewUrl(this.info) : null);
	}

	controls() {
		return { audio: !!this.info?.audio };
	}

	paint(ctx, box) {
		const p = this.player;
		const pic = p.picture();
		const [w, h] = p.size();
		const rect = pic ? fitRect(w, h, ...box) : null;
		if (rect) ctx.drawImage(pic, ...rect);
		else if (p.failed) drawMessage(ctx, "the browser cannot play this file (it is saved)", box);
		else drawMessage(ctx, p.pending() ? "loading…" : "run the workflow to see the video", box);
	}
}

app.registerExtension({
	name: "BCVideoNodes.SaveVideo",
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE) return;
		const table = nodeData.input?.required?.codec?.[1]?.bcv_codecs;

		installPlayer(nodeType, (node) => new SavedVideoWidget(node), [320, 480]);

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const r = onNodeCreated?.apply(this, args);
			const codec = widget(this, "codec");
			if (codec) {
				const callback = codec.callback;
				codec.callback = (...a) => {
					const out = callback?.apply(codec, a);
					guard(() => applyCodec(this, table, true));
					return out;
				};
			}
			guard(() => applyCodec(this, table, false));
			return r;
		};

		// A loaded workflow: narrow the widgets to its codec, keeping its values.
		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function (...args) {
			const r = onConfigure?.apply(this, args);
			guard(() => applyCodec(this, table, false));
			return r;
		};
	},
});
