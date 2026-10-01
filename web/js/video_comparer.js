import { app } from "../../../scripts/app.js";
import { PlayerWidget, drawMessage, drawTag, fitRect, installPlayer, storedOutput, viewUrl } from "./player.js";

// Video Comparer. A fills the node; while the pointer is over the picture, B is painted from the
// left edge up to the pointer, with a divider line and A/B tags. Leave it and A is shown alone.
//
// Both clips come in ONE file that holds A and B side by side (nodes/video_output.py writes them
// into a single video), each half painted into its own place: one decoder, one clock, nothing to
// keep in sync. Clips of different length were cut to the shorter one and a note says so.
//
// What is shown is the node's last output (the "bcv_video" entry). Nothing is saved into the
// workflow file: like Preview Image, the comparison lives with the run.

const NODE = "BCVVideoComparer";
const UI_KEY = "bcv_video";

class ComparerWidget extends PlayerWidget {
	constructor(node) {
		super(node, "comparer");
		this.seen = undefined; // the stored output the clip was built from
		this.sides = []; // which halves the file holds, left to right: ["A", "B"], ["A"] or ["B"]
		this.audio = false;
		this.note = "";
		this.hoverX = null; // local x while the pointer is over the picture
	}

	// Follow the stored output: a new run, a tab switch, a history click, a fresh workflow.
	sync() {
		const out = storedOutput(this.node);
		if (out === this.seen) return;
		this.seen = out;
		const info = out?.[UI_KEY]?.[0];
		this.sides = info?.sides ?? [];
		this.audio = !!info?.audio;
		const f = info?.frames ?? {};
		this.note = this.sides.length === 2 && f.A !== f.B && info.fps
			? `A ${(f.A / info.fps).toFixed(2)}s · B ${(f.B / info.fps).toFixed(2)}s — cut to the shorter clip`
			: "";
		this.player.load(info ? viewUrl(info) : null);
	}

	controls() {
		return { audio: this.audio, note: this.note };
	}

	hover(x) {
		this.hoverX = x;
		this.node.setDirtyCanvas(true, false);
	}

	// The source rect of one side inside the clip.
	half(side) {
		const [w, h] = this.player.size();
		const i = this.sides.indexOf(side);
		if (!w || !h || i < 0) return null;
		const sw = w / this.sides.length;
		return { sx: i * sw, sw, sh: h };
	}

	paint(ctx, [bx, by, bw, bh]) {
		const pic = this.player.picture();
		const a = pic && this.half("A");
		const b = pic && this.half("B");
		const rectA = a && fitRect(a.sw, a.sh, bx, by, bw, bh);
		const rectB = b && fitRect(b.sw, b.sh, bx, by, bw, bh);
		const draw = (src, rect) => ctx.drawImage(pic, src.sx, 0, src.sw, src.sh, ...rect);

		if (!rectA && !rectB) {
			const text = this.player.failed ? "not available (run the workflow again)" : this.player.pending() ? "loading…" : "run the workflow to compare";
			drawMessage(ctx, text, [bx, by, bw, bh]);
		} else if (!rectA || !rectB) {
			draw(a ?? b, rectA ?? rectB);
			drawTag(ctx, rectA ? "A" : "B", bx + bw - 8, by + 8, "right");
		} else {
			draw(a, rectA);
			if (this.hoverX != null) {
				const x = Math.min(Math.max(this.hoverX, bx), bx + bw);
				ctx.save();
				ctx.beginPath();
				ctx.rect(bx, by, x - bx, bh);
				ctx.clip();
				draw(b, rectB);
				ctx.restore();
				ctx.fillStyle = "rgba(255,255,255,.9)";
				ctx.fillRect(x - 1, by, 2, bh);
				drawTag(ctx, "B", bx + 8, by + 8, "left");
				drawTag(ctx, "A", bx + bw - 8, by + 8, "right");
			}
		}
	}
}

app.registerExtension({
	name: "BCVideoNodes.VideoComparer",
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name === NODE) installPlayer(nodeType, (node) => new ComparerWidget(node), [320, 300]);
	},
});
