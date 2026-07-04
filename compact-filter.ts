/**
 * On compaction, keep a tracked path only if it's under pi's cwd and still exists
 * on disk -- drops temp files and anything since removed, no ignore rules to keep.
 */

import { access } from "node:fs/promises";
import { isAbsolute, relative, resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

async function keep(rawPath: string, cwd: string): Promise<boolean> {
	const abs = resolve(cwd, rawPath);
	const rel = relative(cwd, abs);
	if (rel.startsWith("..") || isAbsolute(rel)) return false; // outside cwd
	return access(abs).then(() => true, () => false); // exists on disk?
}

export default function (pi: ExtensionAPI) {
	pi.on("session_before_compact", async (event, ctx) => {
		let dropped = 0;
		for (const set of Object.values(event.preparation.fileOps)) {
			for (const p of [...set]) {
				if (!(await keep(p, ctx.cwd))) {
					set.delete(p);
					dropped++;
				}
			}
		}
		if (dropped) ctx.ui.notify(`Compaction: dropped ${dropped} path(s) outside cwd or missing on disk`, "info");
	});
}
