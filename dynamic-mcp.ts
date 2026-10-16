// npm-description: Dynamically manage MCP servers for the current Pi session.
// npm-keywords: mcp, model-context-protocol, dynamic-mcp, session-scoped, stdio, streamable-http

import { isDeepStrictEqual } from "node:util";
import { StringEnum, Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI, McpServerConfig } from "@earendil-works/pi-coding-agent";

// Pi's McpServerConfig and validator: https://github.com/earendil-works/pi/blob/main/packages/coding-agent/src/core/mcp-servers.ts
// This model-facing subset is validated again by registerMcpServer.
const parameters = Type.Object({
	action: StringEnum(["add", "remove", "list"] as const),
	name: Type.Optional(Type.String({ description: "Server name for add or remove" })),
	config: Type.Optional(
		Type.Object({
			command: Type.Optional(Type.String({ description: "Stdio executable, not a shell command" })),
			args: Type.Optional(Type.Array(Type.String())),
			cwd: Type.Optional(Type.String()),
			env: Type.Optional(Type.Record(Type.String(), Type.String())),
			url: Type.Optional(Type.String({ description: "Streamable HTTP URL" })),
			headers: Type.Optional(Type.Record(Type.String(), Type.String())),
			description: Type.Optional(Type.String()),
			timeout: Type.Optional(Type.Number()),
		}),
	),
});

export default function (pi: ExtensionAPI) {
	const managed = new Map<string, McpServerConfig>();

	pi.registerTool({
		name: "mcp",
		label: "MCP",
		description:
			"Manage session MCP servers: add(name, config) with command/args (stdio) or url (HTTP), remove(name), or list(). list shows server configs managed by this tool, not connection status. env and headers support ${VAR}.",
		parameters,
		async execute(_id, { action, name, config }) {
			let text: string;
			if (action === "list") {
				text = JSON.stringify(Object.fromEntries(managed), null, 2);
			} else if (!name) {
				throw new Error("Server name is required");
			} else if (action === "remove") {
				if (!managed.has(name)) throw new Error(`Server "${name}" is not managed by this tool`);
				pi.unregisterMcpServer(name);
				managed.delete(name);
				text = `Removed "${name}"`;
			} else {
				if (!config || Boolean(config.command) === Boolean(config.url)) {
					throw new Error("Provide exactly one of command (stdio) or url (HTTP) in config");
				}
				const common = { description: config.description, timeout: config.timeout };
				const serverConfig: McpServerConfig = config.url
					? { ...common, url: config.url, headers: config.headers }
					: { ...common, command: config.command!, args: config.args, cwd: config.cwd, env: config.env };
				const duplicate = [...managed].find(([, existing]) => isDeepStrictEqual(existing, serverConfig));
				if (duplicate) throw new Error(`Server "${duplicate[0]}" is already registered with an identical config`);
				pi.registerMcpServer(name, serverConfig);
				managed.set(name, structuredClone(serverConfig));
				text = `Registered "${name}". Use codemode to find and call its tools.`;
			}
			return { content: [{ type: "text", text }], details: undefined };
		},
	});
}
