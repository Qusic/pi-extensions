// npm-description: Keep Pi compaction file lists scoped to the current summary window.
// npm-keywords: compaction, context, context-pruning, context-management, file-tracking, file-operations

/**
 * Pi carries file lists forward across compactions. Keep only paths from the
 * messages being summarized now, including the prefix of a split turn.
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI, FileOperations } from "@earendil-works/pi-coding-agent";

// File tracking reference: https://github.com/earendil-works/pi/blob/main/packages/coding-agent/src/core/compaction/utils.ts
function collectFileOps(message: AgentMessage, fileOps: FileOperations): void {
	if (message.role !== "assistant" && message.role !== "toolResult") return;
	const calls = message.role === "assistant"
		? message.content.filter((block) => block?.type === "toolCall")
		: message.nestedCalls?.calls ?? [];
	for (const call of calls) {
		const path = call.arguments?.path;
		if (typeof path !== "string" || !path) continue;
		if (call.name === "read") fileOps.read.add(path);
		else if (call.name === "write") fileOps.written.add(path);
		else if (call.name === "edit") fileOps.edited.add(path);
	}
}

export default function (pi: ExtensionAPI) {
	pi.on("session_before_compact", (event) => {
		const { fileOps, messagesToSummarize, turnPrefixMessages } = event.preparation;
		for (const set of Object.values(fileOps)) set.clear();

		for (const message of [...messagesToSummarize, ...turnPrefixMessages]) {
			collectFileOps(message, fileOps);
		}
	});
}
