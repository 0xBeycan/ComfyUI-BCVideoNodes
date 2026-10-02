import { app } from "../../../scripts/app.js";
import { guard } from "./player.js";

// Get Video Info: its outputs come in three groups, the audio of the loaded range and the settings
// Load Video loaded with (audio, model, resolution, orientation), the source's values (source_*)
// and the loaded frames' (loaded_*). The
// source and loaded groups each get a slot colour, and a thin line with the group's name above
// their first slot, so a source_ output and its loaded_ twin cannot be taken for each other. The
// names and types stay the node's own; a link keeps its type's colour.

const NODE = "BCVGetVideoInfo";
const GROUPS = [
	{ prefix: "source_", name: "source", color: "#e0a050" },
	{ prefix: "loaded_", name: "loaded", color: "#50b8e0" },
];

// The colours live on the slots, and a loaded workflow replaces the slots with its saved ones.
function colourSlots(node) {
	for (const output of node.outputs ?? []) {
		const group = GROUPS.find((g) => output.name.startsWith(g.prefix));
		if (!group) continue;
		output.color_on = group.color;
		output.color_off = group.color;
	}
}

// Halfway between the slot above a group and its first slot: a line across the node, and the
// group's name at the left of that first slot, where no input sits.
function drawGroups(node, ctx, lowQuality) {
	const outputs = node.outputs ?? [];
	const slotY = (i) => node.getOutputPos(i)[1] - node.pos[1];
	const w = node.size[0];
	ctx.save();
	ctx.lineWidth = 1;
	ctx.font = "10px sans-serif";
	ctx.textAlign = "left";
	ctx.textBaseline = "middle";
	for (const group of GROUPS) {
		const i = outputs.findIndex((o) => o.name.startsWith(group.prefix));
		if (i < 1) continue;
		const y = Math.round((slotY(i - 1) + slotY(i)) / 2) + 0.5;
		ctx.strokeStyle = group.color;
		ctx.beginPath();
		ctx.moveTo(6, y);
		ctx.lineTo(w - 6, y);
		ctx.stroke();
		if (lowQuality) continue;
		ctx.fillStyle = group.color;
		ctx.fillText(group.name, 10, slotY(i));
	}
	ctx.restore();
}

app.registerExtension({
	name: "BCVideoNodes.GetVideoInfo",
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE) return;

		for (const hook of ["onNodeCreated", "onConfigure"]) {
			const original = nodeType.prototype[hook];
			nodeType.prototype[hook] = function (...args) {
				const r = original?.apply(this, args);
				guard(() => colourSlots(this));
				return r;
			};
		}

		const onDrawForeground = nodeType.prototype.onDrawForeground;
		nodeType.prototype.onDrawForeground = function (ctx, canvas, ...rest) {
			const r = onDrawForeground?.apply(this, [ctx, canvas, ...rest]);
			if (!this.flags.collapsed) guard(() => drawGroups(this, ctx, canvas?.low_quality));
			return r;
		};
	},
});
