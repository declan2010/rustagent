//! MCP (Model Context Protocol) client and tool plumbing, matching
//! MinAgent's `src/mcp.mjs`: JSON-RPC over stdio or HTTP, hardened config
//! loading, bounded tool catalogs, and bounded result rendering.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, oneshot};

pub const MCP_PROTOCOL_VERSION: &str = "2025-11-25";
pub const MCP_GUIDANCE: &str = "Use relevant MCP tools.";
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &[
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    MCP_PROTOCOL_VERSION,
];
const REQUEST_TIMEOUT_MS: u64 = 120_000;
pub const MAX_MCP_TEXT_RESULT_CHARS: usize = 48_000;
const MAX_MCP_CONFIG_BYTES: u64 = 1024 * 1024;
pub const MAX_MCP_SERVERS: usize = 32;
pub const MAX_MCP_TOOLS: usize = 32;
pub const MAX_MCP_SCHEMA_BYTES: usize = 8 * 1024;
const MAX_MCP_TOTAL_TOOL_BYTES: usize = 64 * 1024;
pub const MAX_MCP_TOOL_DESCRIPTION_CHARS: usize = 1_500;
pub const MAX_MCP_GUIDANCE_CHARS: usize = 8 * 1024;
const MAX_STDIO_LINE_BYTES: usize = 16 * 1024 * 1024;
const MAX_HTTP_BODY_BYTES: usize = 16 * 1024 * 1024;
const MAX_INSTRUCTIONS_CHARS: usize = 12_000;

/// One advertised MCP tool: the server that owns it plus the remote name.
#[derive(Clone)]
pub struct ToolEntry {
    server: usize,
    pub server_name: String,
    pub remote_tool_name: String,
}

/// Connected servers, advertised tool definitions, and lookup wiring.
pub struct Connections {
    servers: Vec<Arc<Mutex<McpServer>>>,
    pub tool_definitions: Vec<Value>,
    tool_lookup: HashMap<String, ToolEntry>,
    pub server_guidance: Vec<(String, String)>,
    pub warnings: Vec<String>,
}

impl Connections {
    pub fn empty() -> Self {
        Self {
            servers: Vec::new(),
            tool_definitions: Vec::new(),
            tool_lookup: HashMap::new(),
            server_guidance: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn tool_count(&self) -> usize {
        self.tool_lookup.len()
    }

    pub fn lookup(&self, function_name: &str) -> Option<&ToolEntry> {
        self.tool_lookup.get(function_name)
    }

    pub fn has_tools_or_guidance(&self) -> bool {
        !self.tool_definitions.is_empty() || !self.server_guidance.is_empty()
    }

    /// Closes every server: stdio children get a graceful shutdown (stdin EOF,
    /// 1-second grace, then a process-tree kill); HTTP sessions get a DELETE.
    pub async fn close(&mut self) {
        for server in self.servers.drain(..) {
            let mut server = server.lock().await;
            let _ = server.close().await;
        }
    }

    /// Calls one MCP tool and renders its content through the bounded result
    /// formatter. Images are always omitted with a note: rustagent does not
    /// attach image parts to messages yet (MinAgent's attachments path).
    pub async fn execute(&self, function_name: &str, args: &Value) -> Result<(String, bool)> {
        let Some(entry) = self.tool_lookup.get(function_name) else {
            bail!("MCP tool is not available: {function_name}");
        };
        let server = self
            .servers
            .get(entry.server)
            .ok_or_else(|| anyhow::anyhow!("MCP tool is not available: {function_name}"))?;
        let mut server = server.lock().await;
        let result = server.call_tool(&entry.remote_tool_name, args).await?;
        let outcome =
            render_tool_result(&entry.server_name, &entry.remote_tool_name, &result, false);
        Ok((outcome.tool_text, outcome.is_error))
    }
}

/// A rendered MCP tool result: bounded text plus the server's error flag.
pub struct McpOutcome {
    pub tool_text: String,
    pub is_error: bool,
}

/// Renders a `tools/call` result like MinAgent's `executeMcpTool`: text and
/// resource parts join, unknown types get a note, images are omitted with the
/// reason, structured content is appended, and the whole text truncates at
/// 48,000 characters.
pub fn render_tool_result(
    server_name: &str,
    remote_tool_name: &str,
    result: &Value,
    image_enabled: bool,
) -> McpOutcome {
    let mut text_parts: Vec<String> = Vec::new();
    for item in result
        .get("content")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    text_parts.push(text.to_string());
                }
            }
            Some("resource") => {
                if let Some(text) = item.pointer("/resource/text").and_then(Value::as_str) {
                    text_parts.push(text.to_string());
                }
            }
            Some("resource_link") => {
                let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                let uri = item.get("uri").and_then(Value::as_str).unwrap_or("");
                text_parts.push(format!("Resource link: {name}{uri}\n{uri}"));
            }
            Some("image") => {
                let reason = if !image_enabled {
                    "image input is disabled for the model"
                } else {
                    "its format, encoding, or size is unsupported"
                };
                text_parts.push(format!("[An MCP image was omitted because {reason}.]"));
            }
            Some(kind) => {
                let kind: String = kind.chars().take(80).collect();
                text_parts.push(format!("[MCP returned content of type {kind}.]"));
            }
            None => {}
        }
    }
    if let Some(structured) = result
        .get("structuredContent")
        .filter(|value| value.is_object())
    {
        text_parts.push(format!("Structured result:\n{structured}"));
    }
    let mut tool_text = text_parts.join("\n\n").trim().to_string();
    if tool_text.is_empty() {
        tool_text = "The MCP tool returned no text content.".to_string();
    }
    let is_error = result
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if is_error {
        tool_text = format!("MCP tool reported an error.\n{tool_text}");
    }
    if tool_text.chars().count() > MAX_MCP_TEXT_RESULT_CHARS {
        let mut cut: String = tool_text.chars().take(MAX_MCP_TEXT_RESULT_CHARS).collect();
        cut.push_str("\n[Tool result truncated.]");
        tool_text = cut;
    }
    McpOutcome {
        tool_text: format!("[MCP {server_name}/{remote_tool_name}]\n{tool_text}"),
        is_error,
    }
}

/// Connects every configured MCP server, gathering advertised tools and
/// warnings. Mirrors MinAgent's `connectMcpServers` including all limits.
pub async fn connect_mcp_servers(config_path: &Path, default_cwd: &Path) -> Connections {
    let config = match read_config(config_path).await {
        Ok(config) => config,
        Err(error) => {
            if error.chain().any(|cause| {
                cause
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
            }) {
                // A missing config is the common case: stay silent.
                return Connections::empty();
            }
            let mut connections = Connections::empty();
            connections.warnings.push(format!(
                "Could not load MCP configuration {}: {error}",
                config_path.display()
            ));
            return connections;
        }
    };
    let Some(servers) = config.get("mcpServers").filter(|value| value.is_object()) else {
        let mut connections = Connections::empty();
        connections.warnings.push(format!(
            "MCP configuration must contain an object named \"mcpServers\": {}",
            config_path.display()
        ));
        return connections;
    };
    let entries: Vec<(String, Value)> = servers
        .as_object()
        .expect("checked above")
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    if entries.len() > MAX_MCP_SERVERS {
        let mut connections = Connections::empty();
        connections.warnings.push(format!(
            "MCP configuration contains more than {MAX_MCP_SERVERS} servers."
        ));
        return connections;
    }

    let mut connections = Connections::empty();
    let mut total_tool_definition_bytes = 0usize;
    let mut total_tool_budget_reached = false;

    for (server_index, (server_name, server_config)) in entries.iter().enumerate() {
        if total_tool_budget_reached {
            break;
        }
        if connections.tool_definitions.len() >= MAX_MCP_TOOLS {
            connections.warnings.push(format!(
                "Ignoring additional MCP tools after the {MAX_MCP_TOOLS}-tool limit."
            ));
            break;
        }
        let mut server = match create_server(server_config, default_cwd) {
            Ok(server) => server,
            Err(error) => {
                connections.warnings.push(format!(
                    "MCP server \"{server_name}\" could not be connected: {error}"
                ));
                continue;
            }
        };
        if let Err(error) = server.connect().await {
            let _ = server.close().await;
            connections.warnings.push(format!(
                "MCP server \"{server_name}\" could not be connected: {error}"
            ));
            continue;
        }
        let tool_list = match server.list_tools().await {
            Ok(list) => list,
            Err(error) => {
                let _ = server.close().await;
                connections.warnings.push(format!(
                    "MCP server \"{server_name}\" could not be connected: {error}"
                ));
                continue;
            }
        };
        if tool_list.truncated {
            connections.warnings.push(format!(
                "MCP server \"{server_name}\" returned more than {MAX_MCP_TOOLS} tools; only the first {MAX_MCP_TOOLS} were considered."
            ));
        }
        let instructions = server.instructions.clone();
        connections.servers.push(Arc::new(Mutex::new(server)));
        if !instructions.is_empty() {
            connections
                .server_guidance
                .push((server_name.clone(), instructions));
        }
        for (tool_index, remote_tool) in tool_list.tools.iter().enumerate() {
            if connections.tool_definitions.len() >= MAX_MCP_TOOLS {
                connections.warnings.push(format!(
                    "Ignoring additional MCP tools after the {MAX_MCP_TOOLS}-tool limit."
                ));
                total_tool_budget_reached = true;
                break;
            }
            let remote_name = remote_tool
                .get("name")
                .and_then(Value::as_str)
                .map(str::trim)
                .unwrap_or_default();
            if remote_name.is_empty() {
                connections.warnings.push(format!(
                    "Ignoring an MCP tool with no valid name from server \"{server_name}\"."
                ));
                continue;
            }
            let function_name =
                make_function_name(server_index, tool_index, server_name, remote_name);
            let parameters = match remote_tool.get("inputSchema") {
                Some(schema) if schema.is_object() => schema.clone(),
                _ => json!({ "type": "object", "properties": {} }),
            };
            if let Err(error) = validate_schema(&parameters) {
                connections.warnings.push(format!(
                    "Ignoring MCP tool \"{remote_name}\" from \"{server_name}\": {error}"
                ));
                continue;
            }
            let description = build_tool_description(remote_tool, remote_name, server_name);
            let definition = json!({
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": description,
                    "parameters": parameters,
                }
            });
            let definition_bytes = serde_json::to_string(&definition)
                .map(|text| text.len())
                .unwrap_or(usize::MAX);
            if total_tool_definition_bytes + definition_bytes > MAX_MCP_TOTAL_TOOL_BYTES {
                connections
                    .warnings
                    .push("Ignoring additional MCP tools after reaching the 64 KiB combined schema limit.".to_string());
                total_tool_budget_reached = true;
                break;
            }
            total_tool_definition_bytes += definition_bytes;
            connections.tool_definitions.push(definition);
            connections.tool_lookup.insert(
                function_name.clone(),
                ToolEntry {
                    server: connections.servers.len() - 1,
                    server_name: server_name.clone(),
                    remote_tool_name: remote_name.to_string(),
                },
            );
        }
    }
    connections
}

/// Reads the MCP config with the same hardening MinAgent applies: a regular,
/// unlinked file of at most 1 MiB holding a JSON object.
async fn read_config(config_path: &Path) -> Result<Value> {
    // `?` keeps the io::Error as the chain source so a missing file stays
    // recognizable downstream.
    let metadata = std::fs::symlink_metadata(config_path)?;
    if metadata.is_symlink() || !metadata.is_file() || linked(&metadata) {
        bail!("MCP configuration must be a regular, unlinked file.");
    }
    if metadata.len() > MAX_MCP_CONFIG_BYTES {
        bail!("MCP configuration exceeds {MAX_MCP_CONFIG_BYTES} bytes.");
    }
    let bytes = tokio::fs::read(config_path).await?;
    let config: Value = serde_json::from_slice(&bytes)?;
    if !config.is_object() {
        bail!("MCP configuration must be a JSON object.");
    }
    Ok(config)
}

#[cfg(unix)]
fn linked(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    metadata.nlink() > 1
}

#[cfg(windows)]
fn linked(_metadata: &std::fs::Metadata) -> bool {
    false
}

fn create_server(config: &Value, default_cwd: &Path) -> Result<McpServer> {
    if !config.is_object() {
        bail!("Server settings must be a JSON object.");
    }
    if let Some(url) = config.get("url").and_then(Value::as_str) {
        if !url.trim().is_empty() {
            return McpServer::http(config, url);
        }
    }
    let command = config
        .get("command")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if command.is_empty() {
        bail!("Set either a stdio server command or an HTTP server URL.");
    }
    let args = match config.get("args") {
        None => Vec::new(),
        Some(value) => value
            .as_array()
            .and_then(|items| {
                items
                    .iter()
                    .map(|item| item.as_str().map(str::to_string))
                    .collect::<Option<Vec<String>>>()
            })
            .ok_or_else(|| anyhow::anyhow!("Server args must be an array of strings."))?,
    };
    let env = match config.get("env") {
        None => Vec::new(),
        Some(value) => value
            .as_object()
            .and_then(|entries| {
                entries
                    .iter()
                    .map(|(key, value)| value.as_str().map(|text| (key.clone(), text.to_string())))
                    .collect::<Option<Vec<(String, String)>>>()
            })
            .ok_or_else(|| {
                anyhow::anyhow!("Server env must be an object containing string values.")
            })?,
    };
    let cwd = match config.get("cwd").and_then(Value::as_str) {
        None => default_cwd.to_path_buf(),
        Some(relative) => default_cwd.join(relative),
    };
    McpServer::stdio(command, &args, &env, &cwd)
}

/// Builds the advertised function name: `mcp_<server>_<tool>_<sanitized>`,
/// truncated to 64 characters.
pub fn make_function_name(
    server_index: usize,
    tool_index: usize,
    server_name: &str,
    tool_name: &str,
) -> String {
    let safe: String = format!("{server_name}_{tool_name}")
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character
            } else {
                '_'
            }
        })
        .collect();
    format!("mcp_{server_index}_{tool_index}_{safe}")
        .chars()
        .take(64)
        .collect()
}

fn validate_schema(parameters: &Value) -> Result<()> {
    if parameters.get("type").and_then(Value::as_str) != Some("object") {
        bail!("inputSchema must describe an object with object properties.");
    }
    if let Some(properties) = parameters.get("properties") {
        if !properties.is_object() {
            bail!("inputSchema must describe an object with object properties.");
        }
    }
    if let Some(required) = parameters.get("required") {
        let valid = required
            .as_array()
            .is_some_and(|items| items.iter().all(|item| item.is_string()));
        if !valid {
            bail!("inputSchema.required must be an array of strings.");
        }
    }
    let encoded = serde_json::to_string(parameters).unwrap_or_default();
    if encoded.len() > MAX_MCP_SCHEMA_BYTES {
        bail!("inputSchema exceeds {MAX_MCP_SCHEMA_BYTES} bytes.");
    }
    Ok(())
}

fn build_tool_description(remote_tool: &Value, remote_name: &str, server_name: &str) -> String {
    let title = remote_tool.get("title").and_then(Value::as_str);
    let body = remote_tool
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("");
    let body = if body.is_empty() {
        format!("Call the {remote_name} MCP tool.")
    } else {
        body.to_string()
    };
    let joined = match title {
        Some(title) => format!("{title}. {body}"),
        None => body,
    };
    let clean: String = joined
        .chars()
        .map(|character| {
            if character.is_control() || character == '\u{7f}' {
                ' '
            } else {
                character
            }
        })
        .take(MAX_MCP_TOOL_DESCRIPTION_CHARS)
        .collect();
    let server: String = server_name
        .chars()
        .map(|character| {
            if character.is_control() || character == '\u{7f}' {
                ' '
            } else {
                character
            }
        })
        .take(128)
        .collect();
    format!("{clean} (MCP server: {server}.)")
}

/// Renders server instructions into the system-prompt section: "MCP guidance:"
/// followed by each server's instructions, bounded to 8 KiB.
pub fn format_mcp_context(server_guidance: &[(String, String)]) -> String {
    if server_guidance.is_empty() {
        return String::new();
    }
    let mut context = String::from("MCP guidance:");
    for (server_name, instructions) in server_guidance {
        let safe_name: String = server_name
            .chars()
            .map(|character| {
                if character.is_control() || character == '\u{7f}' {
                    ' '
                } else {
                    character
                }
            })
            .take(128)
            .collect();
        let heading = format!("\n\n{}:\n", json!(safe_name));
        let remaining = MAX_MCP_GUIDANCE_CHARS
            .saturating_sub(context.chars().count() + heading.chars().count());
        if remaining <= 0 {
            break;
        }
        let safe_instructions: String = instructions
            .chars()
            .map(|character| {
                if character.is_control() && character != '\n' && character != '\t'
                    || character == '\u{7f}'
                {
                    ' '
                } else {
                    character
                }
            })
            .collect();
        if safe_instructions.chars().count() > remaining {
            let marker = "\n[truncated]";
            let content_limit = remaining.saturating_sub(marker.chars().count()).max(0);
            let mut cut: String = safe_instructions.chars().take(content_limit).collect();
            if remaining > marker.chars().count() {
                cut.push_str(marker);
            }
            context.push_str(&heading);
            context.push_str(&cut);
            break;
        }
        context.push_str(&heading);
        context.push_str(&safe_instructions);
    }
    context
}

// ---------------------------------------------------------------------------
// JSON-RPC client: shared request plumbing over stdio or HTTP transports.
// ---------------------------------------------------------------------------

enum Transport {
    Stdio(StdioTransport),
    Http(HttpTransport),
}

/// One connected MCP server.
pub struct McpServer {
    transport: Transport,
    next_id: u64,
    protocol_version: Option<String>,
    pub instructions: String,
}

/// The paged result of `tools/list`, bounded to [`MAX_MCP_TOOLS`].
pub struct ToolList {
    pub tools: Vec<Value>,
    pub truncated: bool,
}

impl McpServer {
    fn stdio(command: &str, args: &[String], env: &[(String, String)], cwd: &Path) -> Result<Self> {
        Ok(Self {
            transport: Transport::Stdio(StdioTransport::spawn(command, args, env, cwd)?),
            next_id: 1,
            protocol_version: None,
            instructions: String::new(),
        })
    }

    fn http(config: &Value, url: &str) -> Result<Self> {
        Ok(Self {
            transport: Transport::Http(HttpTransport::new(config, url)?),
            next_id: 1,
            protocol_version: None,
            instructions: String::new(),
        })
    }

    /// `initialize` handshake: exchange protocol version, keep the server's
    /// instructions (bounded), then send the `initialized` notification.
    pub async fn connect(&mut self) -> Result<()> {
        let result = self
            .request(
                "initialize",
                json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": { "name": "RustAgent", "version": env!("CARGO_PKG_VERSION") },
                }),
            )
            .await?;
        let version = result
            .get("protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !SUPPORTED_PROTOCOL_VERSIONS.contains(&version) {
            let shown = if version.is_empty() {
                "missing".to_string()
            } else {
                version.to_string()
            };
            bail!("Server selected unsupported MCP protocol version: {shown}");
        }
        self.protocol_version = Some(version.to_string());
        if let Transport::Http(transport) = &mut self.transport {
            transport.protocol_version = Some(version.to_string());
        }
        self.instructions = result
            .get("instructions")
            .and_then(Value::as_str)
            .map(|text| text.chars().take(MAX_INSTRUCTIONS_CHARS).collect())
            .unwrap_or_default();
        self.notify("notifications/initialized", None).await
    }

    /// Paged `tools/list` bounded to [`MAX_MCP_TOOLS`] with a 100-page cap.
    pub async fn list_tools(&mut self) -> Result<ToolList> {
        let mut tools: Vec<Value> = Vec::new();
        let mut cursor: Option<String> = None;
        for _page in 0..100 {
            let params = match &cursor {
                Some(cursor) => json!({ "cursor": cursor }),
                None => json!({}),
            };
            let result = self.request("tools/list", params).await?;
            let page_tools = result
                .get("tools")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let remaining = MAX_MCP_TOOLS - tools.len();
            let page_count = page_tools.len();
            let take = page_count.min(remaining);
            tools.extend(page_tools.into_iter().take(take));
            let next_cursor = result.get("nextCursor").and_then(Value::as_str);
            let truncated =
                page_count > remaining || (tools.len() >= MAX_MCP_TOOLS && next_cursor.is_some());
            if truncated {
                return Ok(ToolList {
                    tools,
                    truncated: true,
                });
            }
            match next_cursor {
                Some(next) => cursor = Some(next.to_string()),
                None => {
                    return Ok(ToolList {
                        tools,
                        truncated: false,
                    });
                }
            }
        }
        bail!("MCP tools/list exceeded the 100-page limit.");
    }

    pub async fn call_tool(&mut self, name: &str, arguments: &Value) -> Result<Value> {
        self.request(
            "tools/call",
            json!({ "name": name, "arguments": arguments }),
        )
        .await
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let message = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        match &mut self.transport {
            Transport::Stdio(transport) => transport.request(id, &message).await,
            Transport::Http(transport) => transport.request(id, &message).await,
        }
    }

    async fn notify(&mut self, method: &str, params: Option<Value>) -> Result<()> {
        let message = match params {
            Some(params) => json!({ "jsonrpc": "2.0", "method": method, "params": params }),
            None => json!({ "jsonrpc": "2.0", "method": method }),
        };
        match &mut self.transport {
            Transport::Stdio(transport) => transport.notify(&message).await,
            Transport::Http(transport) => transport.notify(&message).await,
        }
    }

    pub async fn close(&mut self) -> Result<()> {
        match &mut self.transport {
            Transport::Stdio(transport) => transport.close().await,
            Transport::Http(transport) => transport.close().await,
        }
    }

    /// The HTTP session id negotiated during `initialize`, if any (test hook).
    #[cfg(test)]
    pub fn session_id(&self) -> Option<String> {
        match &self.transport {
            Transport::Stdio(_) => None,
            Transport::Http(transport) => transport.session_id.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// stdio transport: newline-delimited JSON-RPC over the child's pipes.
// ---------------------------------------------------------------------------

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>;

struct StdioTransport {
    child: Child,
    stdin: tokio::process::ChildStdin,
    pending: Pending,
    failure: Arc<Mutex<Option<String>>>,
}

impl StdioTransport {
    fn spawn(command: &str, args: &[String], env: &[(String, String)], cwd: &Path) -> Result<Self> {
        let mut process = Command::new(command);
        process
            .args(args)
            .current_dir(cwd)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        for (key, value) in env {
            process.env(key, value);
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            process.as_std_mut().process_group(0);
        }
        let mut child = process
            .spawn()
            .map_err(|error| anyhow::anyhow!("Could not start the MCP server: {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("MCP server stdin is not available."))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("MCP server stdout is not available."))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow::anyhow!("MCP server stderr is not available."))?;

        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let failure = Arc::new(Mutex::new(None::<String>));
        let stderr_tail = Arc::new(Mutex::new(String::new()));

        // Response pump: one line per JSON-RPC message.
        {
            let pending = Arc::clone(&pending);
            let failure = Arc::clone(&failure);
            let stderr_tail = Arc::clone(&stderr_tail);
            let child_id = child.id();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stdout);
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line).await {
                        Ok(0) | Err(_) => {
                            let tail = stderr_tail.lock().await.trim().to_string();
                            let detail = if tail.is_empty() {
                                String::new()
                            } else {
                                format!(": {tail}")
                            };
                            fail(
                                &failure,
                                &pending,
                                format!("Server exited (unknown status){detail}"),
                            )
                            .await;
                            let _ = child_id;
                            break;
                        }
                        Ok(_) => {}
                    }
                    if line.len() > MAX_STDIO_LINE_BYTES {
                        fail(
                            &failure,
                            &pending,
                            "MCP server sent an oversized stdio message.".to_string(),
                        )
                        .await;
                        continue;
                    }
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let message: Value = match serde_json::from_str(trimmed) {
                        Ok(message) => message,
                        Err(_) => {
                            fail(
                                &failure,
                                &pending,
                                "MCP server wrote a non-JSON line to stdout.".to_string(),
                            )
                            .await;
                            continue;
                        }
                    };
                    let Some(id) = message.get("id") else {
                        continue;
                    };
                    let Some(id) = id
                        .as_u64()
                        .or_else(|| id.as_i64().map(|value| value as u64))
                    else {
                        continue;
                    };
                    let mut pending = pending.lock().await;
                    if let Some(sender) = pending.remove(&id) {
                        if let Some(error) = message.get("error") {
                            let text = error
                                .get("message")
                                .and_then(Value::as_str)
                                .unwrap_or("MCP request failed.");
                            let _ = sender.send(Err(text.to_string()));
                        } else {
                            let _ = sender
                                .send(Ok(message.get("result").cloned().unwrap_or(Value::Null)));
                        }
                    }
                }
            });
        }
        // stderr pump: keep the last 8,000 characters for failure messages.
        {
            let stderr_tail = Arc::clone(&stderr_tail);
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut chunk = String::new();
                loop {
                    chunk.clear();
                    match reader.read_line(&mut chunk).await {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {}
                    }
                    let mut tail = stderr_tail.lock().await;
                    tail.push_str(&chunk);
                    let excess = tail.len().saturating_sub(8_000);
                    if excess > 0 {
                        *tail = tail.split_off(excess);
                    }
                }
            });
        }

        Ok(Self {
            child,
            stdin,
            pending,
            failure,
        })
    }

    async fn request(&mut self, id: u64, message: &Value) -> Result<Value> {
        if let Some(failure) = self.failure_message().await {
            bail!("{failure}");
        }
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().await.insert(id, sender);
        let line = format!("{}\n", serde_json::to_string(message)?);
        if let Err(error) = self.stdin.write_all(line.as_bytes()).await {
            self.pending.lock().await.remove(&id);
            bail!("MCP server is not available: {error}");
        }
        let timeout = tokio::time::timeout(Duration::from_millis(REQUEST_TIMEOUT_MS), receiver);
        match timeout.await {
            Ok(Ok(Ok(value))) => Ok(value),
            Ok(Ok(Err(text))) => bail!("{text}"),
            Ok(Err(_)) => bail!("MCP server dropped the request."),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                bail!("MCP request timed out");
            }
        }
    }

    async fn notify(&mut self, message: &Value) -> Result<()> {
        let line = format!("{}\n", serde_json::to_string(message)?);
        self.stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|error| anyhow::anyhow!("MCP server is not available: {error}"))
    }

    async fn failure_message(&self) -> Option<String> {
        self.failure.lock().await.clone()
    }

    /// Graceful shutdown: stdin EOF, a 1-second grace period, then a
    /// process-tree kill (matching MinAgent's terminateProcessTree).
    async fn close(&mut self) -> Result<()> {
        let _ = self.stdin.shutdown().await;
        let wait = self.child.wait();
        match tokio::time::timeout(Duration::from_secs(1), wait).await {
            Ok(_) => {}
            Err(_) => {
                #[cfg(unix)]
                {
                    if let Some(pid) = self.child.id() {
                        unsafe {
                            libc::kill(-(pid as i32), libc::SIGKILL);
                        }
                    }
                }
                let _ = self.child.kill().await;
            }
        }
        fail(
            &self.failure,
            &self.pending,
            "MCP server closed.".to_string(),
        )
        .await;
        Ok(())
    }
}

/// Marks the transport failed and rejects every pending request, matching
/// MinAgent's `fail`.
async fn fail(failure: &Arc<Mutex<Option<String>>>, pending: &Pending, message: String) {
    let mut failure = failure.lock().await;
    if failure.is_none() {
        *failure = Some(message.clone());
    }
    let mut pending = pending.lock().await;
    for (_, sender) in pending.drain() {
        let _ = sender.send(Err(failure.clone().unwrap_or_default()));
    }
}

// ---------------------------------------------------------------------------
// HTTP transport: JSON POST responses or SSE event streams.
// ---------------------------------------------------------------------------

struct HttpTransport {
    http: reqwest::Client,
    url: String,
    headers: Vec<(String, String)>,
    session_id: Option<String>,
    protocol_version: Option<String>,
}

impl HttpTransport {
    fn new(config: &Value, url: &str) -> Result<Self> {
        let parsed =
            reqwest::Url::parse(url).map_err(|_| anyhow::anyhow!("Server URL is invalid."))?;
        if parsed.scheme() != "http" && parsed.scheme() != "https" {
            bail!("MCP server URLs must use HTTP or HTTPS.");
        }
        let headers = match config.get("headers") {
            None => Vec::new(),
            Some(value) => value
                .as_object()
                .and_then(|entries| {
                    entries
                        .iter()
                        .map(|(key, value)| {
                            value.as_str().map(|text| (key.clone(), text.to_string()))
                        })
                        .collect::<Option<Vec<(String, String)>>>()
                })
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "HTTP server headers must be an object containing string values."
                    )
                })?,
        };
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_millis(REQUEST_TIMEOUT_MS))
            .build()?;
        Ok(Self {
            http,
            url: url.trim().to_string(),
            headers,
            session_id: None,
            protocol_version: None,
        })
    }

    async fn request(&mut self, id: u64, message: &Value) -> Result<Value> {
        self.post(message, Some(id)).await
    }

    async fn notify(&mut self, message: &Value) -> Result<()> {
        self.post(message, None).await?;
        Ok(())
    }

    async fn post(&mut self, message: &Value, id: Option<u64>) -> Result<Value> {
        let mut request = self
            .http
            .post(&self.url)
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json")
            .json(message);
        for (key, value) in &self.headers {
            request = request.header(key, value);
        }
        if let Some(session) = &self.session_id {
            request = request.header("Mcp-Session-Id", session);
        }
        if let Some(version) = &self.protocol_version {
            request = request.header("MCP-Protocol-Version", version);
        }
        let response = request
            .send()
            .await
            .map_err(|error| anyhow::anyhow!("MCP HTTP request failed: {error}"))?;
        if let Some(session) = response
            .headers()
            .get("mcp-session-id")
            .and_then(|value| value.to_str().ok())
        {
            self.session_id = Some(session.to_string());
        }
        let status = response.status();
        if !status.is_success() {
            let body = read_limited_text(response).await?;
            let mut detail: String = body.chars().take(1000).collect();
            if let Ok(parsed) = serde_json::from_str::<Value>(&body) {
                if let Some(message) = parsed.pointer("/error/message").and_then(Value::as_str) {
                    detail = message.to_string();
                }
            }
            bail!("MCP HTTP {status}: {detail}");
        }
        let Some(id) = id else {
            return Ok(Value::Null);
        };
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_lowercase();
        let result = if content_type.contains("text/event-stream") {
            read_sse_response(response, id).await?
        } else {
            let body = read_limited_text(response).await?;
            if body.trim().is_empty() {
                bail!("MCP server returned an empty response.");
            }
            serde_json::from_str(&body)?
        };
        let response_id = result.get("id").and_then(Value::as_u64).or_else(|| {
            result
                .get("id")
                .and_then(Value::as_i64)
                .map(|value| value as u64)
        });
        if response_id != Some(id) {
            bail!("MCP server returned the wrong response ID.");
        }
        if let Some(error) = result.get("error") {
            bail!(
                "{}",
                error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("MCP request failed.")
            );
        }
        Ok(result.get("result").cloned().unwrap_or(Value::Null))
    }

    async fn close(&mut self) -> Result<()> {
        let Some(session) = &self.session_id else {
            return Ok(());
        };
        let mut request = self
            .http
            .delete(&self.url)
            .header("Mcp-Session-Id", session);
        for (key, value) in &self.headers {
            request = request.header(key, value);
        }
        if let Some(version) = &self.protocol_version {
            request = request.header("MCP-Protocol-Version", version);
        }
        let _ = tokio::time::timeout(Duration::from_secs(3), request.send()).await;
        Ok(())
    }
}

/// Reads the body with a 16 MiB cap, mirroring readLimitedResponseText.
async fn read_limited_text(response: reqwest::Response) -> Result<String> {
    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if buffer.len() + chunk.len() > MAX_HTTP_BODY_BYTES {
            bail!("MCP HTTP response exceeds {MAX_HTTP_BODY_BYTES} bytes.");
        }
        buffer.extend_from_slice(&chunk);
    }
    Ok(String::from_utf8_lossy(&buffer).to_string())
}

/// Reads an SSE response until the frame carrying `id` arrives.
async fn read_sse_response(response: reqwest::Response, id: u64) -> Result<Value> {
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut total_bytes = 0usize;
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        total_bytes += chunk.len();
        if total_bytes > MAX_HTTP_BODY_BYTES {
            bail!("MCP HTTP response exceeds {MAX_HTTP_BODY_BYTES} bytes.");
        }
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(at) = find_frame_boundary(&buffer) {
            let frame: String = buffer.drain(..at.end).collect();
            let data: String = frame
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|line| line.trim_start())
                .collect::<Vec<&str>>()
                .join("\n");
            if data.trim().is_empty() {
                continue;
            }
            let message: Value = serde_json::from_str(&data)
                .map_err(|_| anyhow::anyhow!("MCP server sent an invalid JSON-RPC event."))?;
            let message_id = message.get("id").and_then(Value::as_u64).or_else(|| {
                message
                    .get("id")
                    .and_then(Value::as_i64)
                    .map(|value| value as u64)
            });
            if message_id == Some(id) {
                return Ok(message);
            }
        }
    }
    bail!("MCP server closed the event stream before returning the requested response.");
}

fn find_frame_boundary(buffer: &str) -> Option<std::ops::Range<usize>> {
    let plain = buffer.find("\n\n").map(|at| at..at + 2);
    let carriage = buffer.find("\r\n\r\n").map(|at| at..at + 4);
    match (plain, carriage) {
        (Some(plain), Some(carriage)) => {
            if plain.start < carriage.start {
                Some(plain)
            } else {
                Some(carriage)
            }
        }
        (Some(plain), None) => Some(plain),
        (None, Some(carriage)) => Some(carriage),
        (None, None) => None,
    }
}

/// The MCP config path: `~/.config/rustagent/mcp.json`, keeping rustagent's
/// user-level config convention (MinAgent keeps `mcp.json` next to the app).
pub fn config_path() -> Option<PathBuf> {
    let config_home = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(config_home.join("rustagent").join("mcp.json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn workspace() -> TempDir {
        TempDir::new().expect("temp dir")
    }

    #[test]
    fn function_names_are_sanitized_and_bounded() {
        assert_eq!(
            make_function_name(0, 1, "fs", "read file"),
            "mcp_0_1_fs_read_file"
        );
        assert_eq!(
            make_function_name(3, 12, "weird/server", "tool.name"),
            "mcp_3_12_weird_server_tool_name"
        );
        let long = make_function_name(9, 9, &"s".repeat(80), &"t".repeat(80));
        assert_eq!(long.chars().count(), 64);
        assert!(long.starts_with("mcp_9_9_"));
    }

    #[test]
    fn guidance_context_is_bounded() {
        assert_eq!(format_mcp_context(&[]), "");
        let guidance = vec![("server-one".to_string(), "Be concise.".repeat(2_000))];
        let context = format_mcp_context(&guidance);
        assert!(context.starts_with("MCP guidance:"));
        assert!(context.contains("\"server-one\":"));
        assert!(context.contains("[truncated]"), "long guidance truncates");
        assert!(context.chars().count() <= MAX_MCP_GUIDANCE_CHARS);
    }

    #[test]
    fn tool_descriptions_carry_title_and_server() {
        let tool = json!({
            "name": "echo",
            "title": "Echo",
            "description": "Repeats input"
        });
        let description = build_tool_description(&tool, "echo", "demo");
        assert!(description.starts_with("Echo. Repeats input"));
        assert!(description.ends_with("(MCP server: demo.)"));
        let tool = json!({ "name": "echo" });
        let description = build_tool_description(&tool, "echo", "demo");
        assert!(description.starts_with("Call the echo MCP tool."));
    }

    #[test]
    fn schemas_are_validated_like_minagent() {
        assert!(validate_schema(&json!({ "type": "object" })).is_ok());
        assert!(
            validate_schema(&json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }))
            .is_ok()
        );
        assert!(validate_schema(&json!({ "type": "string" })).is_err());
        assert!(
            validate_schema(&json!({
                "type": "object",
                "properties": []
            }))
            .is_err()
        );
        assert!(
            validate_schema(&json!({
                "type": "object",
                "required": [7]
            }))
            .is_err()
        );
        let huge =
            json!({ "type": "object", "properties": { "x": { "blob": "y".repeat(9 * 1024) } } });
        assert!(validate_schema(&huge).is_err());
    }

    #[test]
    fn results_render_text_resources_and_notes() {
        let result = render_tool_result(
            "srv",
            "echo",
            &json!({
                "content": [
                    { "type": "text", "text": "Hello" },
                    { "type": "resource", "resource": { "text": "file body" } },
                    { "type": "resource_link", "name": "doc", "uri": "file:///doc" },
                    { "type": "image", "data": "AAAA", "mimeType": "image/png" },
                    { "type": "audio", "blob": "zzz" },
                ],
                "structuredContent": { "rows": 2 }
            }),
            false,
        );
        assert!(result.tool_text.starts_with("[MCP srv/echo]\n"));
        assert!(result.tool_text.contains("Hello\n\nfile body"));
        assert!(result.tool_text.contains("Resource link: docfile:///doc"));
        assert!(
            result.tool_text.contains(
                "[An MCP image was omitted because image input is disabled for the model.]"
            )
        );
        assert!(
            result
                .tool_text
                .contains("[MCP returned content of type audio.]")
        );
        assert!(result.tool_text.contains("Structured result:\n{"));
        assert!(!result.is_error);
    }

    #[test]
    fn result_errors_and_truncation_apply() {
        let error_result = render_tool_result(
            "srv",
            "boom",
            &json!({
                "content": [{ "type": "text", "text": "failed" }],
                "isError": true
            }),
            false,
        );
        assert!(error_result.is_error);
        assert!(
            error_result
                .tool_text
                .contains("MCP tool reported an error.\nfailed")
        );

        let empty = render_tool_result("srv", "quiet", &json!({ "content": [] }), false);
        assert!(
            empty
                .tool_text
                .contains("The MCP tool returned no text content.")
        );

        let big = "x".repeat(MAX_MCP_TEXT_RESULT_CHARS + 500);
        let big_result = render_tool_result(
            "srv",
            "big",
            &json!({
                "content": [{ "type": "text", "text": big }]
            }),
            false,
        );
        assert!(big_result.tool_text.contains("[Tool result truncated.]"));
    }

    #[tokio::test]
    async fn missing_config_is_silent_and_bad_config_warns() {
        let root = workspace();
        let missing = root.path().join("missing.json");
        let connections = connect_mcp_servers(&missing, root.path()).await;
        assert!(connections.warnings.is_empty());
        assert_eq!(connections.tool_count(), 0);

        let bad = root.path().join("bad.json");
        tokio::fs::write(&bad, "{ \"other\": {} }")
            .await
            .expect("write");
        let connections = connect_mcp_servers(&bad, root.path()).await;
        assert_eq!(connections.warnings.len(), 1);
        assert!(connections.warnings[0].contains("mcpServers"));

        let too_many = root.path().join("many.json");
        let mut servers = serde_json::Map::new();
        for index in 0..40 {
            servers.insert(
                format!("s{index}"),
                json!({ "url": "http://127.0.0.1:1/mcp" }),
            );
        }
        tokio::fs::write(&too_many, json!({ "mcpServers": servers }).to_string())
            .await
            .expect("write");
        let connections = connect_mcp_servers(&too_many, root.path()).await;
        assert!(
            connections
                .warnings
                .iter()
                .any(|warning| warning.contains("more than 32 servers")),
            "{:?}",
            connections.warnings
        );
    }

    #[test]
    fn server_settings_are_validated() {
        let root = workspace();
        assert!(
            create_server(&json!({ "command": "" }), root.path())
                .err()
                .expect("empty command fails")
                .to_string()
                .contains("Set either a stdio server command")
        );
        assert!(
            create_server(&json!({ "command": "x", "args": [1] }), root.path())
                .err()
                .expect("bad args fail")
                .to_string()
                .contains("args must be an array of strings")
        );
        assert!(
            create_server(&json!({ "command": "x", "env": { "A": 1 } }), root.path())
                .err()
                .expect("bad env fails")
                .to_string()
                .contains("env must be an object")
        );
        assert!(
            create_server(&json!({ "url": "ftp://example.invalid/mcp" }), root.path())
                .err()
                .expect("bad url fails")
                .to_string()
                .contains("HTTP or HTTPS")
        );
        assert!(
            create_server(
                &json!({ "url": "http://127.0.0.1:9/mcp", "headers": { "X": 3 } }),
                root.path()
            )
            .err()
            .expect("bad headers fail")
            .to_string()
            .contains("headers must be an object")
        );
    }

    /// End-to-end stdio handshake against a tiny python3 MCP server: the
    /// connection lists tools, executes one, and closes cleanly. Skips when
    /// python3 is not available.
    #[tokio::test]
    async fn stdio_server_end_to_end() {
        if std::process::Command::new("python3")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_err()
        {
            return;
        }
        let root = workspace();
        let script = r#"
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    message = json.loads(line)
    if "id" not in message:
        continue
    method = message.get("method")
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "instructions": "Use the demo tools politely.",
        }
    elif method == "tools/list":
        result = {
            "tools": [
                { "name": "echo", "description": "Echo text", "inputSchema": { "type": "object", "properties": { "text": { "type": "string" } }, "required": ["text"] } },
                { "name": "ping", "inputSchema": { "type": "object" } },
            ]
        }
    elif method == "tools/call":
        text = message["params"]["arguments"].get("text", "")
        result = { "content": [{ "type": "text", "text": "echo:" + text }] }
    else:
        result = {}
    sys.stdout.write(json.dumps({ "jsonrpc": "2.0", "id": message["id"], "result": result }) + "\n")
    sys.stdout.flush()
"#;
        let script_path = root.path().join("server.py");
        tokio::fs::write(&script_path, script)
            .await
            .expect("write script");
        let mut server = McpServer::stdio(
            "python3",
            &[script_path.to_string_lossy().to_string()],
            &[],
            root.path(),
        )
        .expect("spawn");
        server.connect().await.expect("connect");
        assert_eq!(server.instructions, "Use the demo tools politely.");
        let list = server.list_tools().await.expect("list tools");
        assert_eq!(list.tools.len(), 2);
        assert!(!list.truncated);
        let called = server
            .call_tool("echo", &json!({ "text": "hi" }))
            .await
            .expect("call");
        let outcome = render_tool_result("demo", "echo", &called, false);
        assert!(outcome.tool_text.contains("echo:hi"));
        server.close().await.expect("close");
    }

    /// End-to-end HTTP handshake against a mock server: session headers flow
    /// both ways over one keep-alive connection and close issues the DELETE.
    #[tokio::test]
    async fn http_server_end_to_end() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let server_task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut deleted = false;
            loop {
                let mut buffer = vec![0u8; 64 * 1024];
                let Ok(count) = socket.read(&mut buffer).await else {
                    break;
                };
                if count == 0 {
                    break;
                }
                let request = String::from_utf8_lossy(&buffer[..count]).to_string();
                let Some((head, body)) = request.split_once("\r\n\r\n") else {
                    continue;
                };
                let start = head.lines().next().expect("start line");
                if start.starts_with("DELETE") {
                    deleted = true;
                    socket
                        .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")
                        .await
                        .expect("delete response");
                    break;
                }
                let parsed: Value = serde_json::from_str(body).expect("body json");
                // Notifications carry no id: acknowledge and keep waiting.
                let Some(id) = parsed.get("id").and_then(Value::as_u64) else {
                    socket
                        .write_all(b"HTTP/1.1 202 Accepted\r\ncontent-length: 0\r\n\r\n")
                        .await
                        .expect("notify response");
                    continue;
                };
                let (result, session_header) = if body.contains("initialize") {
                    (
                        json!({ "protocolVersion": "2025-06-18", "instructions": "Be brief." }),
                        "mcp-session-id: session-1\r\n",
                    )
                } else if body.contains("tools/list") {
                    (
                        json!({ "tools": [
                            { "name": "hello", "description": "Say hi", "inputSchema": { "type": "object" } }
                        ] }),
                        "",
                    )
                } else if body.contains("tools/call") {
                    (
                        json!({ "content": [{ "type": "text", "text": "hi there" }] }),
                        "",
                    )
                } else {
                    (json!({}), "")
                };
                let payload = json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n{session_header}content-length: {}\r\n\r\n{}",
                    payload.len(),
                    payload
                );
                socket.write_all(response.as_bytes()).await.expect("write");
            }
            assert!(deleted, "close must issue the session DELETE");
        });

        let transport = HttpTransport::new(
            &json!({ "headers": { "Authorization": "Bearer demo" } }),
            &format!("http://{address}/mcp"),
        )
        .expect("client");
        let mut server = McpServer {
            transport: Transport::Http(transport),
            next_id: 1,
            protocol_version: None,
            instructions: String::new(),
        };
        server.connect().await.expect("connect");
        assert_eq!(server.instructions, "Be brief.");
        assert_eq!(server.session_id().as_deref(), Some("session-1"));
        let list = server.list_tools().await.expect("list");
        assert_eq!(list.tools.len(), 1);
        let called = server.call_tool("hello", &json!({})).await.expect("call");
        let outcome = render_tool_result("http-demo", "hello", &called, false);
        assert!(outcome.tool_text.contains("hi there"));
        server.close().await.expect("close");
        server_task.await.expect("server task");
    }
}
