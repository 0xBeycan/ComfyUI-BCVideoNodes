import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// The toast of nodes/unused_outputs.py: the server sends EVENT when another pack's on_prompt
// handler runs after this pack's, which turns the RAM saving of unused outputs off for that run.
const EVENT = "bcvideonodes.unused_outputs";

app.registerExtension({
	name: "BCVideoNodes.UnusedOutputs",
	setup() {
		api.addEventListener(EVENT, ({ detail }) => {
			const toast = app.extensionManager?.toast;
			if (toast?.add) toast.add({ severity: "warn", summary: "BCVideoNodes", detail: detail?.message, life: 10000 });
			else console.warn(`BCVideoNodes: ${detail?.message}`);
		});
	},
});
