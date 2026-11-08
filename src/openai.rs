use anyhow::{Context, Result, anyhow, bail};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Duration;

use crate::config::Config;
use crate::types::{Completion, FunctionCall, Message, ToolCall, Usage};

const MAX_TOOL_ARGUMENT_CHARS: usize = 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Backoff schedule for recoverable endpoint errors (connect drops, 429, 5xx,
/// non-streaming shape, and streams that end before any delta is emitted).
/// Mirrors MinAgent's `ENDPOINT_RECOVERY_DELAYS_MS`.
const ENDPOINT_RECOVERY_DELAYS_MS: [u64; 3] = [5_000, 15_000, 30_000];

/// Classifies an endpoint error as retryable. A stream that ended or errored is
/// only retryable when no delta was emitted yet, so partial output is never
/// duplicated. Shape/connect/429/5xx errors are always retryable.
fn is_recoverable_endpoint_error(error: &anyhow::Error, emitted_deltas: usize) -> bool {
    let message = error.to_string();
    if message.contains("[ENDPOINT_SHAPE]") {
        return true;
    }
    if message.contains("[STREAM_ENDED]") || message.contains("[STREAM_ERROR]") {
        return emitted_deltas == 0;
    }
    if message.contains("Could not connect to the OpenAI-compatible endpoint") {
        return true;
    }
    if let Some(rest) = message.strip_prefix("Endpoint returned HTTP ") {
        if let Some(code_text) = rest.split_whitespace().next() {
            let code_text = code_text.trim_end_matches(':');
            if let Ok(code) = code_text.parse::<u16>() {
                if code == 429 || code >= 500 {
                    return true;
                }
            }
        }
    }
    false
}

/// Unifies tool_call ids across the flow: synthesized ids (call_0, call_1 by position)
/// repeat between rounds and cloud upstreams rename ids per request. On collisions the
/// upstream attributes tool results to the wrong round and the model enters infinite
/// verification. Only repeated ids are renamed, as a pair (call + its result), with a
/// deterministic suffix; ids that are already unique stay untouched.
pub fn unique_tool_call_ids(messages: &[Message]) -> Vec<Message> {
    let mut used: HashSet<String> = HashSet::new();
    let mut duplicate = 0usize;
    let mut caller_map: HashMap<String, String> = HashMap::new();
    let mut result = Vec::with_capacity(messages.len());

    for message in messages {
        let mut mapped = message.clone();
        if let Some(calls) = &message.tool_calls {
            if !calls.is_empty() {
                caller_map.clear();
                let mut mapped_calls = Vec::with_capacity(calls.len());
                for call in calls {
                    let mut final_id = call.id.clone();
                    if !call.id.is_empty() {
                        if used.contains(&call.id) {
                            loop {
                                duplicate += 1;
                                let candidate = format!("{}__d{duplicate}", call.id);
                                if !used.contains(&candidate) {
                                    final_id = candidate;
                                    break;
                                }
                            }
                        }
                        used.insert(final_id.clone());
                        caller_map.insert(call.id.clone(), final_id.clone());
                    }
                    let mut mapped_call = call.clone();
                    mapped_call.id = final_id;
                    mapped_calls.push(mapped_call);
                }
                mapped.tool_calls = Some(mapped_calls);
            }
        }
        if mapped.role == "tool" {
            if let Some(id) = &mapped.tool_call_id {
                if let Some(mapped_id) = caller_map.get(id) {
                    mapped.tool_call_id = Some(mapped_id.clone());
                }
            }
        }
        result.push(mapped);
    }
    result
}

/// True when any non-empty tool-call id repeats, i.e. when [`unique_tool_call_ids`]
/// would actually rewrite something. Lets requests skip cloning the whole
/// conversation in the common case.
fn has_duplicate_tool_call_ids(messages: &[Message]) -> bool {
    let mut used: HashSet<&str> = HashSet::new();
    for message in messages {
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                if !call.id.is_empty() && !used.insert(call.id.as_str()) {
                    return true;
                }
            }
        }
    }
    false
}

/// Adapts OpenAI messages to Ollama's native format: non-string content serialized and
/// arguments as a JSON OBJECT (a raw string triggers HTTP 400 "can't find closing '}'").
/// id/type are preserved: the cloud bridge links the tool result by id.
pub fn to_ollama_messages(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .map(|message| {
            let mut mapped = serde_json::to_value(message).unwrap_or(Value::Null);
            if message.role == "developer" {
                mapped["role"] = json!("system");
            }
            if let Some(content) = &message.content {
                if !content.is_string() && !content.is_array() && !content.is_null() {
                    mapped["content"] = Value::String(content.to_string());
                }
            }
            if let Some(calls) = &message.tool_calls {
                let mapped_calls: Vec<Value> = calls
                    .iter()
                    .map(|call| {
                        let parsed = serde_json::from_str::<Value>(&call.function.arguments)
                            .ok()
                            .filter(Value::is_object);
                        let mut function = json!({ "name": call.function.name });
                        if let Some(parsed) = parsed {
                            function["arguments"] = parsed;
                        }
                        json!({ "id": call.id, "type": call.kind, "function": function })
                    })
                    .collect();
                mapped["tool_calls"] = Value::Array(mapped_calls);
            }
            mapped
        })
        .collect()
}

/// Turns an .../v1/chat/completions endpoint into .../api/chat on the same host.
/// Strips the `/v1/chat/completions` bits of an OpenAI-compatible endpoint and
/// returns Ollama's root URL (used by the `/api/show` capability lookups).
/// Returns an empty string when the endpoint has no parseable origin.
pub fn ollama_base_url(endpoint: &str) -> String {
    match ollama_chat_url(endpoint) {
        Some(chat) => chat
            .trim_end_matches("/api/chat")
            .trim_end_matches('/')
            .to_string(),
        None => String::new(),
    }
}

/// Minimal percent-encoding for the `/api/show?model=` fallback query.
fn urlencode(model: &str) -> String {
    let mut encoded = String::new();
    for byte in model.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b':' => {
                encoded.push(byte as char);
            }
            _ => encoded.push_str(&format!("%{:02X}", byte)),
        }
    }
    encoded
}

pub fn ollama_chat_url(endpoint: &str) -> Option<String> {
    let (scheme, rest) = endpoint.split_once("://")?;
    let (authority, path) = match rest.find('/') {
        Some(index) => (&rest[..index], &rest[index..]),
        None => (rest, ""),
    };
    if authority.is_empty() {
        return None;
    }
    let path = path.trim_end_matches('/');
    let path = path.strip_suffix("/chat/completions").unwrap_or(path);
    let path = path.strip_suffix("/v1").unwrap_or(path);
    Some(format!("{scheme}://{authority}{path}/api/chat"))
}

fn models_endpoint(endpoint: &str) -> Option<String> {
    let base = endpoint.trim_end_matches('/');
    let base = base.strip_suffix("/chat/completions")?;
    Some(format!("{base}/models"))
}

fn parse_usage(value: &Value) -> Usage {
    Usage {
        prompt_tokens: value
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        completion_tokens: value
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    }
}

fn truncate(value: &str, max: usize) -> String {
    if value.len() <= max {
        return value.to_string();
    }
    let mut end = max;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &value[..end])
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Returns the byte length of the first complete frame and the number of bytes consumed.
fn next_sse_frame(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = find_subslice(buffer, b"\n\n").map(|index| (index, index + 2));
    let crlf = find_subslice(buffer, b"\r\n\r\n").map(|index| (index, index + 4));
    match (lf, crlf) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

pub struct StreamParser<'a> {
    on_text: &'a mut dyn FnMut(&str),
    on_reasoning: &'a mut dyn FnMut(&str),
    on_tool_call: &'a mut dyn FnMut(&str),
    native: bool,
    content_type: String,
    tool_calls: BTreeMap<i64, ToolCall>,
    content: String,
    // gpt-oss-style models (e.g. north-mini-code) emit tool calls and reasoning
    // as literal text tags (`<|START_ACTION|>...<|END_ACTION|>`,
    // `<|START_THINKING|>...<|END_THINKING|>`) inside `content` instead of via
    // the structured `tool_calls`/`thinking` fields. These fields buffer and
    // re-route those tags so they never reach the chat as plain text.
    content_buffer: String,
    in_thinking: bool,
    in_action: bool,
    action_buf: String,
    usage: Option<Usage>,
    finish_reason: Option<String>,
    finished: bool,
    saw_data_line: bool,
    bytes_read: usize,
    max_response_bytes: usize,
}

impl<'a> StreamParser<'a> {
    pub fn new(
        on_text: &'a mut dyn FnMut(&str),
        on_reasoning: &'a mut dyn FnMut(&str),
        on_tool_call: &'a mut dyn FnMut(&str),
        native: bool,
        content_type: String,
    ) -> Self {
        Self {
            on_text,
            on_reasoning,
            on_tool_call,
            native,
            content_type,
            tool_calls: BTreeMap::new(),
            content: String::new(),
            content_buffer: String::new(),
            in_thinking: false,
            in_action: false,
            action_buf: String::new(),
            usage: None,
            finish_reason: None,
            finished: false,
            saw_data_line: false,
            bytes_read: 0,
            max_response_bytes: MAX_RESPONSE_BYTES,
        }
    }

    pub fn finished(&self) -> bool {
        self.finished
    }

    /// The gpt-oss tag markers, kept in one place so the scanner and the
    /// partial-prefix check stay in sync.
    const START_THINKING: &'static str = "<|START_THINKING|>";
    const END_THINKING: &'static str = "<|END_THINKING|>";
    const START_ACTION: &'static str = "<|START_ACTION|>";
    const END_ACTION: &'static str = "<|END_ACTION|>";

    /// Routes an incoming `content` delta through the gpt-oss tag filter. Plain
    /// text is streamed to `on_text`, reasoning tags to `on_reasoning`, and
    /// action tags are parsed into structured tool calls. A small tail that
    /// could be the start of a marker is held back until the next chunk.
    fn process_content(&mut self, text: &str) {
        self.content_buffer.push_str(text);
        self.drain_content_buffer();
    }

    fn drain_content_buffer(&mut self) {
        let markers: [&'static str; 4] = [
            Self::START_THINKING,
            Self::END_THINKING,
            Self::START_ACTION,
            Self::END_ACTION,
        ];
        loop {
            if self.in_action {
                // Accumulate action JSON until the closing tag. Hold back a
                // tail that might be the start of <|END_ACTION|>.
                if let Some(pos) = self.content_buffer.find(Self::END_ACTION) {
                    self.action_buf.push_str(&self.content_buffer[..pos]);
                    self.content_buffer =
                        self.content_buffer[pos + Self::END_ACTION.len()..].to_string();
                    self.in_action = false;
                    self.finalize_action();
                } else {
                    let safe = self
                        .content_buffer
                        .len()
                        .saturating_sub(Self::END_ACTION.len() - 1);
                    let safe = self.safe_cut(&self.content_buffer, safe);
                    self.action_buf.push_str(&self.content_buffer[..safe]);
                    self.content_buffer = self.content_buffer[safe..].to_string();
                    break;
                }
                continue;
            }

            if self.in_thinking {
                if let Some(pos) = self.content_buffer.find(Self::END_THINKING) {
                    let head = &self.content_buffer[..pos];
                    if !head.is_empty() {
                        (self.on_reasoning)(head);
                    }
                    self.content_buffer =
                        self.content_buffer[pos + Self::END_THINKING.len()..].to_string();
                    self.in_thinking = false;
                } else {
                    let safe = self
                        .content_buffer
                        .len()
                        .saturating_sub(Self::END_THINKING.len() - 1);
                    let safe = self.safe_cut(&self.content_buffer, safe);
                    if safe > 0 {
                        (self.on_reasoning)(&self.content_buffer[..safe]);
                    }
                    self.content_buffer = self.content_buffer[safe..].to_string();
                    break;
                }
                continue;
            }

            // Not inside a tag: find the earliest marker.
            let earliest = markers
                .iter()
                .filter_map(|marker| self.content_buffer.find(marker).map(|pos| (pos, *marker)))
                .min_by_key(|(pos, _)| *pos);

            match earliest {
                Some((pos, marker)) => {
                    let head = &self.content_buffer[..pos];
                    if !head.is_empty() {
                        self.content.push_str(head);
                        (self.on_text)(head);
                    }
                    self.content_buffer = self.content_buffer[pos + marker.len()..].to_string();
                    match marker {
                        Self::START_THINKING => self.in_thinking = true,
                        Self::START_ACTION => {
                            self.in_action = true;
                            self.action_buf.clear();
                        }
                        Self::END_THINKING | Self::END_ACTION => {
                            // A closing tag without an opening tag: drop it.
                        }
                        _ => {}
                    }
                }
                None => {
                    // No complete marker yet. A marker can only start at `<|`,
                    // so flush everything except a tail beginning with `<|` that
                    // is still a prefix of some marker. This keeps plain text
                    // streaming without the full marker-length delay.
                    let mut hold = self.content_buffer.len();
                    for (pos, _) in self.content_buffer.match_indices("<|") {
                        let tail = &self.content_buffer[pos..];
                        let forming = [
                            Self::START_THINKING,
                            Self::END_THINKING,
                            Self::START_ACTION,
                            Self::END_ACTION,
                        ]
                        .iter()
                        .any(|marker| marker.starts_with(tail) && tail.len() < marker.len());
                        if forming {
                            hold = pos;
                            break;
                        }
                    }
                    let limit = self.safe_cut(&self.content_buffer, hold);
                    if limit > 0 {
                        let head = &self.content_buffer[..limit];
                        if !head.is_empty() {
                            self.content.push_str(head);
                            (self.on_text)(head);
                        }
                    }
                    self.content_buffer = self.content_buffer[limit..].to_string();
                    break;
                }
            }
        }
    }

    /// Returns the largest cut point `<= limit` that lands on a UTF-8 boundary.
    fn safe_cut(&self, text: &str, limit: usize) -> usize {
        if limit >= text.len() {
            return text.len();
        }
        let mut cut = limit;
        while cut > 0 && !text.is_char_boundary(cut) {
            cut -= 1;
        }
        cut
    }

    /// Parses the buffered action JSON into structured tool calls. The gpt-oss
    /// format is an array of `{tool_call_id, tool_name, parameters}` objects.
    fn finalize_action(&mut self) {
        let trimmed = self.action_buf.trim();
        if trimmed.is_empty() {
            return;
        }
        // Some models wrap the array in whitespace/newlines; trim to the JSON.
        let start = match trimmed.find('[') {
            Some(index) => index,
            None => return,
        };
        let end = match trimmed.rfind(']') {
            Some(index) => index,
            None => return,
        };
        let payload = &trimmed[start..=end];
        let Ok(actions) = serde_json::from_str::<Vec<Value>>(payload) else {
            // If it is not a valid array, surface it as text so nothing is lost.
            let text = self.action_buf.clone();
            self.content.push_str(&text);
            (self.on_text)(&text);
            return;
        };
        for (position, action) in actions.iter().enumerate() {
            let index = position as i64;
            let name = action
                .get("tool_name")
                .and_then(Value::as_str)
                .or_else(|| {
                    action
                        .get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(Value::as_str)
                })
                .unwrap_or("");
            if name.is_empty() {
                continue;
            }
            let parameters = action
                .get("parameters")
                .or_else(|| action.get("arguments"))
                .or_else(|| {
                    action
                        .get("function")
                        .and_then(|function| function.get("arguments"))
                });
            let arguments = match parameters {
                Some(Value::String(text)) => text.clone(),
                Some(value) => value.to_string(),
                None => String::new(),
            };
            let id = action
                .get("tool_call_id")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(|text| text.to_string())
                .unwrap_or_else(|| format!("call_{position}"));
            let entry = ToolCall {
                id,
                kind: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments,
                },
            };
            self.tool_calls.insert(index, entry);
            (self.on_tool_call)(name);
        }
    }

    /// Flushes any buffered content when the stream ends.
    pub fn flush_content(&mut self) {
        if self.in_action {
            // No closing tag arrived; absorb whatever is still buffered and try
            // to parse it as the action payload.
            self.action_buf.push_str(&self.content_buffer);
            self.content_buffer.clear();
            self.in_action = false;
            self.finalize_action();
            self.action_buf.clear();
        }
        if self.in_thinking {
            let head = std::mem::take(&mut self.content_buffer);
            if !head.is_empty() {
                (self.on_reasoning)(&head);
            }
            self.in_thinking = false;
        }
        if !self.content_buffer.is_empty() {
            let head = std::mem::take(&mut self.content_buffer);
            if !head.is_empty() {
                self.content.push_str(&head);
                (self.on_text)(&head);
            }
        }
    }

    pub fn add_bytes(&mut self, count: usize) -> Result<()> {
        self.bytes_read += count;
        if self.bytes_read > self.max_response_bytes {
            bail!("Endpoint streaming response exceeds the {MAX_RESPONSE_BYTES} byte limit.");
        }
        Ok(())
    }

    pub fn feed_sse_frame(&mut self, frame: &str) -> Result<()> {
        let data = frame
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(|line| line.trim_start())
            .collect::<Vec<_>>()
            .join("\n");
        let data = data.trim();
        if data.is_empty() {
            return Ok(());
        }
        if data == "[DONE]" {
            self.finished = true;
            return Ok(());
        }
        let event: Value = serde_json::from_str(data)
            .map_err(|_| anyhow!("Endpoint sent an invalid streaming event."))?;
        if let Some(error) = event.get("error").filter(|value| !value.is_null()) {
            let detail = match error {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            bail!("Endpoint streaming error: {}", truncate(&detail, 1000));
        }
        if let Some(usage) = event.get("usage").filter(|value| !value.is_null()) {
            self.usage = Some(parse_usage(usage));
        }
        let Some(choice) = event.get("choices").and_then(|choices| choices.get(0)) else {
            return Ok(());
        };
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            if !reason.is_empty() {
                self.finish_reason = Some(reason.to_string());
            }
        }
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        let reasoning = delta
            .get("reasoning_summary")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
            .or_else(|| {
                delta
                    .get("reasoning_content")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
            });
        if let Some(reasoning) = reasoning {
            (self.on_reasoning)(reasoning);
        }
        if let Some(content) = delta.get("content") {
            if let Some(text) = content.as_str() {
                if !text.is_empty() {
                    self.process_content(text);
                }
            } else if let Some(parts) = content.as_array() {
                for part in parts {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        if !text.is_empty() {
                            self.process_content(text);
                        }
                    }
                }
            }
        }
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for delta_call in calls {
                let index = delta_call
                    .get("index")
                    .and_then(Value::as_i64)
                    .unwrap_or(self.tool_calls.len() as i64);
                let entry = self.tool_calls.entry(index).or_insert_with(|| ToolCall {
                    id: String::new(),
                    kind: "function".to_string(),
                    function: FunctionCall::default(),
                });
                if let Some(id) = delta_call.get("id").and_then(Value::as_str) {
                    if !id.is_empty() {
                        entry.id = id.to_string();
                    }
                }
                if let Some(kind) = delta_call.get("type").and_then(Value::as_str) {
                    if !kind.is_empty() {
                        entry.kind = kind.to_string();
                    }
                }
                if let Some(name) = delta_call
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                {
                    if !name.is_empty() {
                        entry.function.name.push_str(name);
                        (self.on_tool_call)(&entry.function.name);
                    }
                }
                if let Some(arguments) = delta_call
                    .get("function")
                    .and_then(|function| function.get("arguments"))
                    .and_then(Value::as_str)
                {
                    if !arguments.is_empty() {
                        entry.function.arguments.push_str(arguments);
                        if entry.function.arguments.len() > MAX_TOOL_ARGUMENT_CHARS {
                            bail!("Endpoint returned tool arguments larger than the 1 MiB limit.");
                        }
                        (self.on_tool_call)(&entry.function.name);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn feed_ndjson_line(&mut self, line: &str) -> Result<()> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(());
        }
        let event: Value = serde_json::from_str(trimmed)
            .map_err(|_| anyhow!("Endpoint sent an invalid streaming event."))?;
        if let Some(error) = event.get("error").filter(|value| !value.is_null()) {
            let detail = match error {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            // An error as the FIRST line labeled application/json means the whole response is
            // the error (cloud bridge / dead runner): endpoint shape, not stream corruption.
            let code = if !self.saw_data_line && self.content_type.contains("application/json") {
                "ENDPOINT_SHAPE"
            } else {
                "STREAM_ERROR"
            };
            bail!(
                "[{code}] Endpoint streaming error: {}",
                truncate(&detail, 1000)
            );
        }
        self.saw_data_line = true;

        if let Some(message) = event.get("message").filter(|value| value.is_object()) {
            if let Some(thinking) = message.get("thinking").and_then(Value::as_str) {
                if !thinking.is_empty() {
                    (self.on_reasoning)(thinking);
                }
            }
            if let Some(text) = message.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    self.process_content(text);
                }
            }
            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                for (position, incoming) in calls.iter().enumerate() {
                    let index = incoming
                        .get("index")
                        .and_then(Value::as_i64)
                        .unwrap_or(position as i64);
                    let name = incoming
                        .get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let arguments = match incoming
                        .get("function")
                        .and_then(|function| function.get("arguments"))
                    {
                        Some(Value::String(text)) => text.clone(),
                        Some(Value::Null) | None => String::new(),
                        Some(other) => other.to_string(),
                    };
                    let complete_incoming = !arguments.is_empty()
                        && serde_json::from_str::<Value>(&arguments)
                            .ok()
                            .map(|value| value.is_object())
                            .unwrap_or(false);
                    let existing = self.tool_calls.get(&index);
                    let compatible_name = match existing {
                        None => true,
                        Some(call) => {
                            name.is_empty()
                                || call.function.name.is_empty()
                                || name.starts_with(&call.function.name)
                                || call.function.name.starts_with(name)
                        }
                    };
                    let replace_complete =
                        existing.is_some() && complete_incoming && compatible_name;
                    let replace_repeated = !replace_complete
                        && existing
                            .map(|call| {
                                name.starts_with(&call.function.name)
                                    && arguments.starts_with(&call.function.arguments)
                            })
                            .unwrap_or(false);
                    if replace_complete {
                        // The incoming args form a complete object: replace, never concatenate.
                        let entry = self.tool_calls.get_mut(&index).expect("entry exists");
                        entry.function.arguments = arguments;
                        if !name.is_empty()
                            && (entry.function.name.is_empty()
                                || name.starts_with(&entry.function.name))
                        {
                            entry.function.name = name.to_string();
                        }
                    } else if replace_repeated {
                        // The chunk repeats the full call so far (accumulated): replace.
                        let entry = self.tool_calls.get_mut(&index).expect("entry exists");
                        entry.function.name = name.to_string();
                        entry.function.arguments = arguments;
                    } else {
                        let entry = self.tool_calls.entry(index).or_insert_with(|| ToolCall {
                            id: format!("call_{index}"),
                            kind: "function".to_string(),
                            function: FunctionCall::default(),
                        });
                        entry.function.name.push_str(name);
                        entry.function.arguments.push_str(&arguments);
                        if entry.function.arguments.len() > MAX_TOOL_ARGUMENT_CHARS {
                            bail!("Endpoint returned tool arguments larger than the 1 MiB limit.");
                        }
                    }
                    let current_name = self
                        .tool_calls
                        .get(&index)
                        .map(|call| call.function.name.clone())
                        .unwrap_or_default();
                    (self.on_tool_call)(&current_name);
                }
            }
        }

        if event.get("done").and_then(Value::as_bool).unwrap_or(false) {
            self.finished = true;
            if let Some(reason) = event.get("done_reason").and_then(Value::as_str) {
                if !reason.is_empty() {
                    self.finish_reason =
                        Some(if reason == "abort" { "aborted" } else { reason }.to_string());
                }
            }
            let prompt_tokens = event.get("prompt_eval_count").and_then(Value::as_u64);
            let completion_tokens = event.get("eval_count").and_then(Value::as_u64);
            if prompt_tokens.is_some() || completion_tokens.is_some() {
                self.usage = Some(Usage {
                    prompt_tokens: prompt_tokens.unwrap_or(0),
                    completion_tokens: completion_tokens.unwrap_or(0),
                });
            }
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<Completion> {
        self.flush_content();
        if !self.finished && self.finish_reason.is_none() {
            bail!("[STREAM_ENDED] Endpoint stream ended before a complete response was received.");
        }
        let content = if self.content.is_empty() {
            Some(Value::Null)
        } else {
            Some(Value::String(self.content))
        };
        let mut message = Message::assistant(content, None);
        if !self.tool_calls.is_empty() {
            message.tool_calls = Some(self.tool_calls.into_values().collect());
        }
        if self.finish_reason.as_deref() == Some("length") {
            bail!(
                "Endpoint stopped at its output token limit; the partial response was discarded."
            );
        }
        if self.finish_reason.as_deref() == Some("content_filter") {
            bail!("Endpoint stopped the response because of its content filter.");
        }
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                if !self.native && call.id.is_empty() {
                    bail!("[INVALID_TOOL_CALL] Endpoint returned an incomplete tool call.");
                }
                if call.function.name.is_empty() {
                    bail!("[INVALID_TOOL_CALL] Endpoint returned an incomplete tool call.");
                }
                let valid = serde_json::from_str::<Value>(&call.function.arguments)
                    .ok()
                    .map(|value| value.is_object())
                    .unwrap_or(false);
                if !valid {
                    bail!(
                        "[INVALID_TOOL_CALL] Endpoint returned invalid arguments for tool {}.",
                        call.function.name
                    );
                }
            }
        }
        Ok(Completion {
            message,
            usage: self.usage,
            finish_reason: self.finish_reason,
        })
    }
}

async fn read_prefix(response: reqwest::Response, max_bytes: usize) -> String {
    let mut collected: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let Ok(chunk) = chunk else { break };
        let remaining = max_bytes.saturating_sub(collected.len());
        if remaining == 0 {
            break;
        }
        let take = remaining.min(chunk.len());
        collected.extend_from_slice(&chunk[..take]);
        if collected.len() >= max_bytes {
            break;
        }
    }
    let mut text = String::from_utf8_lossy(&collected).to_string();
    if collected.len() >= max_bytes {
        text.push_str(" [response excerpt truncated]");
    }
    text
}

pub struct Client {
    pub endpoint: String,
    pub api_key: Option<String>,
    pub model: String,
    pub ollama_think: Option<String>,
    /// Context window used to set Ollama's `num_ctx` on the native `/api/chat`
    /// path so long conversations do not get truncated to Ollama's small default.
    pub context_window: u64,
    /// Models that rejected the `think` option with HTTP 400 ("does not
    /// support thinking"): subsequent requests for them omit `think`.
    thinking_disabled: std::sync::Mutex<std::collections::HashSet<String>>,
    /// Ollama `/api/show` snapshots per model: thinking capability, declared
    /// reasoning-effort levels, native-gguf marker. Populated lazily.
    think_cache: std::sync::Mutex<std::collections::HashMap<String, ThinkInfo>>,
    /// Models whose `/api/show` lookup failed once: lookup is not retried, so
    /// the native `think` logic and the old HTTP 400 recovery remain in charge.
    show_failed: std::sync::Mutex<std::collections::HashSet<String>>,
    /// (model, level) pairs whose "Ollama may ignore this level" warning has
    /// already been printed once.
    level_warned: std::sync::Mutex<std::collections::HashSet<(String, String)>>,
    http: reqwest::Client,
}

impl Client {
    pub fn new(config: &Config) -> Result<Self> {
        let mut builder = reqwest::Client::builder();
        if config.timeout_ms > 0 {
            builder = builder.timeout(Duration::from_millis(config.timeout_ms));
        }
        let http = builder.build().context("could not build the HTTP client")?;
        Ok(Self {
            endpoint: config.endpoint.clone(),
            api_key: config.api_key.clone(),
            model: config.model.clone(),
            ollama_think: config.ollama_think.clone(),
            context_window: config.context_window,
            thinking_disabled: std::sync::Mutex::new(std::collections::HashSet::new()),
            think_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            show_failed: std::sync::Mutex::new(std::collections::HashSet::new()),
            level_warned: std::sync::Mutex::new(std::collections::HashSet::new()),
            http,
        })
    }

    pub async fn complete(
        &self,
        messages: &[Message],
        tools: Option<&[Value]>,
        on_text: &mut dyn FnMut(&str),
        on_reasoning: &mut dyn FnMut(&str),
        on_tool_call: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        let native_target = self
            .ollama_think
            .as_ref()
            .and_then(|_| ollama_chat_url(&self.endpoint));
        let unique_storage;
        let unique: &[Message] = if has_duplicate_tool_call_ids(messages) {
            unique_storage = unique_tool_call_ids(messages);
            unique_storage.as_slice()
        } else {
            messages
        };
        let body_messages = if native_target.is_some() {
            Value::Array(to_ollama_messages(unique))
        } else {
            serde_json::to_value(unique)?
        };
        let mut body = json!({
            "model": self.model,
            "messages": body_messages,
            "stream": true
        });
        if let Some(think) = &self.ollama_think {
            // A previous attempt provably showed this model cannot think;
            // do not repeat the rejected option on every request.
            let disabled = self
                .thinking_disabled
                .lock()
                .ok()
                .map(|set| set.contains(&self.model))
                .unwrap_or(false);
            if !disabled {
                // Ask Ollama once (cached) whether this model thinks at all: a
                // model without the `thinking` capability rejects `think` with
                // HTTP 400, so never send the option in the first place.
                if self.model_lacks_thinking().await {
                    if let Ok(mut set) = self.thinking_disabled.lock() {
                        set.insert(self.model.clone());
                    }
                    crate::ui::print_info(&crate::ui::warning(&format!(
                        "· {} does not support thinking; continuing without it.",
                        self.model
                    )));
                } else {
                    body["think"] = match think.as_str() {
                        "false" => json!(false),
                        "true" => json!(true),
                        other => json!(other),
                    };
                    if !matches!(think.as_str(), "false" | "true") {
                        self.warn_unmapped_level(think).await;
                    }
                }
            }
        }
        if let Some(tools) = tools {
            body["tools"] = json!(tools);
            body["tool_choice"] = json!("auto");
        }
        // On the native /api/chat path, tell Ollama the real context window so
        // long conversations are not truncated to its small default num_ctx.
        if native_target.is_some() {
            body["options"] = json!({ "num_ctx": self.context_window });
        }

        let target = native_target
            .as_deref()
            .unwrap_or(&self.endpoint)
            .to_string();
        let native = native_target.is_some();

        // Shared delta counter so all three callbacks can increment it without
        // multiple mutable borrows. A retry is only safe when this stays 0.
        let emitted_deltas = std::cell::Cell::new(0usize);
        let mut attempt = 0usize;
        loop {
            let mut on_text_w = |chunk: &str| {
                emitted_deltas.set(emitted_deltas.get() + 1);
                on_text(chunk);
            };
            let mut on_reasoning_w = |chunk: &str| {
                emitted_deltas.set(emitted_deltas.get() + 1);
                on_reasoning(chunk);
            };
            let mut on_tool_call_w = |name: &str| {
                emitted_deltas.set(emitted_deltas.get() + 1);
                on_tool_call(name);
            };
            match self
                .stream_once(
                    &target,
                    native,
                    &body,
                    &mut on_text_w,
                    &mut on_reasoning_w,
                    &mut on_tool_call_w,
                )
                .await
            {
                Ok(completion) => return Ok(completion),
                Err(error) => {
                    let message = error.to_string();
                    // Ollama rejects the whole request when `think` is sent to
                    // a model that cannot think. Retry immediately without it
                    // and remember the model: this fires at most once because
                    // the `think` field is gone on the next pass.
                    if native
                        && message.contains("does not support thinking")
                        && body.get("think").is_some()
                    {
                        if let Some(object) = body.as_object_mut() {
                            object.remove("think");
                        }
                        if let Ok(mut set) = self.thinking_disabled.lock() {
                            set.insert(self.model.clone());
                        }
                        crate::ui::print_info(&crate::ui::warning(&format!(
                            "· {} does not support thinking; continuing without it.",
                            self.model
                        )));
                        continue;
                    }
                    if !is_recoverable_endpoint_error(&error, emitted_deltas.get())
                        || attempt >= ENDPOINT_RECOVERY_DELAYS_MS.len()
                    {
                        return Err(error);
                    }
                    let delay = ENDPOINT_RECOVERY_DELAYS_MS[attempt];
                    let kind = if error.to_string().contains("[ENDPOINT_SHAPE]") {
                        "Non-streaming response from"
                    } else {
                        "Unstable endpoint:"
                    };
                    crate::ui::print_info(&crate::ui::warning(&format!(
                        "· {kind} {target} ({}); retry {}/{} in {}s…",
                        truncate(&error.to_string(), 70),
                        attempt + 1,
                        ENDPOINT_RECOVERY_DELAYS_MS.len(),
                        delay / 1000,
                    )));
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                    attempt += 1;
                }
            }
        }
    }

    /// One streaming attempt: sends the request, classifies the response shape,
    /// and drives the [`StreamParser`] to completion.
    async fn stream_once(
        &self,
        target: &str,
        native: bool,
        body: &Value,
        on_text: &mut dyn FnMut(&str),
        on_reasoning: &mut dyn FnMut(&str),
        on_tool_call: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        let mut request = self.http.post(target).json(body);
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }
        let response = request.send().await.map_err(|error| {
            anyhow!("Could not connect to the OpenAI-compatible endpoint: {error}")
        })?;
        if !response.status().is_success() {
            let status = response.status();
            let body_text = read_prefix(response, 8 * 1024).await;
            bail!("Endpoint returned HTTP {status}: {body_text}");
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_lowercase();
        // Native Ollama emits NDJSON; the cloud bridge labels its stream application/json.
        // Only SSE is excluded on this path.
        let use_ndjson = native && !content_type.contains("text/event-stream");
        if !use_ndjson && !content_type.contains("text/event-stream") {
            let body_text = read_prefix(response, 2 * 1024).await;
            let expected = if native {
                "streaming NDJSON"
            } else {
                "streaming"
            };
            bail!(
                "[ENDPOINT_SHAPE] The endpoint did not return a {expected} response (Content-Type: {}). Response: {body_text}",
                if content_type.is_empty() {
                    "unknown"
                } else {
                    &content_type
                }
            );
        }

        let mut parser =
            StreamParser::new(on_text, on_reasoning, on_tool_call, native, content_type);
        let mut stream = response.bytes_stream();
        let mut buffer: Vec<u8> = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("could not read the streaming response")?;
            parser.add_bytes(chunk.len())?;
            buffer.extend_from_slice(&chunk);
            if use_ndjson {
                while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
                    let line: Vec<u8> = buffer.drain(..=position).collect();
                    let text = String::from_utf8_lossy(&line[..line.len() - 1]);
                    parser.feed_ndjson_line(&text)?;
                    if parser.finished() {
                        break;
                    }
                }
            } else {
                while let Some((frame_len, consumed)) = next_sse_frame(&buffer) {
                    let text = String::from_utf8_lossy(&buffer[..frame_len]).to_string();
                    buffer.drain(..consumed);
                    parser.feed_sse_frame(&text)?;
                    if parser.finished() {
                        break;
                    }
                }
            }
            if parser.finished() {
                break;
            }
        }
        if !parser.finished() && !buffer.is_empty() {
            let rest = String::from_utf8_lossy(&buffer).to_string();
            if !rest.trim().is_empty() {
                if use_ndjson {
                    parser.feed_ndjson_line(&rest)?;
                } else {
                    parser.feed_sse_frame(&rest)?;
                }
            }
        }
        parser.finish()
    }

    pub async fn list_models(&self) -> Result<Vec<ModelEntry>> {
        let url = models_endpoint(&self.endpoint)
            .ok_or_else(|| anyhow!("Cannot derive the model catalog URL from this endpoint."))?;
        let mut request = self
            .http
            .get(url)
            .timeout(Duration::from_secs(15))
            .header(reqwest::header::ACCEPT, "application/json");
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }
        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let suffix = if status == reqwest::StatusCode::NOT_FOUND
                || status == reqwest::StatusCode::METHOD_NOT_ALLOWED
            {
                " (this server does not expose a model catalog)"
            } else {
                ""
            };
            bail!("Could not list models: HTTP {status}{suffix}.");
        }
        let payload: Value = response.json().await?;
        let data = payload
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("The model catalog must contain a data array."))?;
        let mut seen = HashSet::new();
        let mut models = Vec::new();
        for entry in data {
            let Some(id) = entry.get("id").and_then(Value::as_str) else {
                continue;
            };
            let id = id.trim();
            if id.is_empty() || id.chars().any(|character| character.is_control()) {
                continue;
            }
            if !seen.insert(id.to_string()) {
                continue;
            }
            let context_window = entry
                .get("context_window")
                .and_then(Value::as_u64)
                .or_else(|| entry.get("context_length").and_then(Value::as_u64));
            let input_modalities = entry
                .get("input_modalities")
                .and_then(Value::as_array)
                .or_else(|| {
                    entry
                        .get("architecture")
                        .and_then(|architecture| architecture.get("input_modalities"))
                        .and_then(Value::as_array)
                })
                .map(|array| {
                    array
                        .iter()
                        .filter_map(|item| item.as_str().map(|text| text.to_string()))
                        .collect::<Vec<_>>()
                });
            let supports_tools = entry
                .get("capabilities")
                .and_then(|capabilities| capabilities.get("tools"))
                .and_then(Value::as_bool);
            models.push(ModelEntry {
                id: id.to_string(),
                context_window,
                input_modalities,
                supports_tools,
            });
        }
        if models.is_empty() {
            bail!("The API returned no available model identifiers.");
        }
        models.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(models)
    }

    /// Returns the cached `ThinkInfo` for a model, ignoring `/api/show`
    /// failures (synchronous, used by the Ctrl+T cycle and the selector).
    pub fn cached_think_info(&self, model: &str) -> Option<ThinkInfo> {
        self.think_cache
            .lock()
            .ok()
            .and_then(|cache| cache.get(model).cloned())
    }

    /// Ollama's `/api/show` snapshot for a model, fetched once and cached.
    /// Failures are remembered (no retry storm); the caller treats `None` as
    /// "unknown, behave as before".
    pub async fn think_info_for(&self, model: &str) -> Option<ThinkInfo> {
        if let Some(info) = self.cached_think_info(model) {
            return Some(info);
        }
        if self
            .show_failed
            .lock()
            .ok()
            .map(|failed| failed.contains(model))
            .unwrap_or(false)
        {
            return None;
        }
        let base = ollama_base_url(&self.endpoint);
        if base.is_empty() {
            return None;
        }
        let payload = match self.show_request(&format!("{base}/api/show"), model).await {
            Some(payload) => Some(payload),
            None => {
                self.show_request(
                    &format!("{base}/api/show?model={}", urlencode(model)),
                    model,
                )
                .await
            }
        };
        let Some(payload) = payload else {
            if let Ok(mut failed) = self.show_failed.lock() {
                failed.insert(model.to_string());
            }
            return None;
        };
        let has_thinking = payload
            .get("capabilities")
            .and_then(Value::as_array)
            .map(|capabilities| {
                capabilities
                    .iter()
                    .filter_map(Value::as_str)
                    .any(|capability| capability == "thinking")
            })
            .unwrap_or(false);
        let vision = payload
            .get("capabilities")
            .and_then(Value::as_array)
            .map(|capabilities| {
                capabilities
                    .iter()
                    .filter_map(Value::as_str)
                    .any(|capability| capability == "vision")
            })
            .unwrap_or(false);
        let audio = payload
            .get("capabilities")
            .and_then(Value::as_array)
            .map(|capabilities| {
                capabilities
                    .iter()
                    .filter_map(Value::as_str)
                    .any(|capability| capability == "audio")
            })
            .unwrap_or(false);
        let template = payload
            .get("template")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let (levels, default_level) = template_think_levels(template);
        let gguf_arch = payload
            .get("model_info")
            .and_then(Value::as_object)
            .and_then(|entries| {
                entries
                    .iter()
                    .find(|(key, _)| {
                        key.as_str() == "general.architecture" || key.ends_with(".architecture")
                    })
                    .and_then(|(_, value)| value.as_str())
                    .map(str::to_string)
            });
        // The declared context length is an architecture-prefixed metadata
        // key (qwen35.context_length, llama.context_length, ...).
        let context_length = payload
            .get("model_info")
            .and_then(Value::as_object)
            .and_then(|entries| {
                entries
                    .iter()
                    .find(|(key, _)| {
                        let key = key.as_str();
                        key == "context_length" || key.ends_with(".context_length")
                    })
                    .and_then(|(_, value)| value.as_u64())
            });
        let info = ThinkInfo {
            has_thinking,
            vision,
            audio,
            levels,
            default_level,
            gguf_arch,
            context_length,
        };
        if let Ok(mut cache) = self.think_cache.lock() {
            cache.insert(model.to_string(), info.clone());
        }
        Some(info)
    }

    async fn show_request(&self, url: &str, model: &str) -> Option<Value> {
        let mut request = self
            .http
            .post(url)
            .timeout(Duration::from_secs(3))
            .json(&json!({ "model": model }));
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }
        let response = request.send().await.ok()?;
        if !response.status().is_success() {
            return None;
        }
        response.json::<Value>().await.ok()
    }

    /// Whether the model provably cannot think (`thinking` absent from its
    /// capabilities). Fetches the snapshot on the first call.
    async fn model_lacks_thinking(&self) -> bool {
        matches!(
            self.think_info_for(&self.model).await,
            Some(info) if !info.has_thinking
        )
    }

    /// One-line warning when a template-level is selected for a native GGUF
    /// import: current Ollama builds may not map it, letting the template
    /// default (usually the extra-deep one) apply instead. Printed once per
    /// model and level.
    async fn warn_unmapped_level(&self, level: &str) {
        self.think_info_for(&self.model).await;
        let Some(info) = self.cached_think_info(&self.model) else {
            return;
        };
        if !info.levels.iter().any(|declared| declared == level) {
            return;
        }
        if info.gguf_arch.is_none() {
            return;
        }
        let already = self
            .level_warned
            .lock()
            .ok()
            .map(|mut warned| !warned.insert((self.model.clone(), level.to_string())))
            .unwrap_or(true);
        if already {
            return;
        }
        let fallback = info
            .default_level
            .clone()
            .unwrap_or_else(|| "the template's default".to_string());
        crate::ui::print_info(&crate::ui::warning(&format!(
            "· Ollama may ignore think levels for this native model; its template default '{}' applies if unmapped.",
            fallback
        )));
    }

    /// Short per-model annotations for the `/model` selector panel: models
    /// known to lack the `thinking` capability are marked "(no think)".
    pub async fn think_notes(&self, ids: &[String]) -> Vec<String> {
        let futures = ids.iter().map(|model| {
            let model = model.clone();
            async move {
                self.think_info_for(&model)
                    .await
                    .map(|info| {
                        if info.has_thinking {
                            String::new()
                        } else {
                            " (no think)".to_string()
                        }
                    })
                    .unwrap_or_default()
            }
        });
        futures_util::future::join_all(futures).await
    }

    /// Pulls a model through Ollama's native `/api/pull` (NDJSON streaming),
    /// forwarding each `status` field to `on_status`. Fails when the endpoint
    /// is not parseable as an Ollama origin or when the stream reports an
    /// `error` field.
    pub async fn pull_model(&self, name: &str, on_status: &mut dyn FnMut(&str)) -> Result<()> {
        let base = ollama_base_url(&self.endpoint);
        anyhow::ensure!(
            !base.is_empty(),
            "this endpoint does not expose Ollama's /api/pull"
        );
        let mut request = self
            .http
            .post(format!("{base}/api/pull"))
            .json(&json!({ "name": name, "stream": true }));
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }
        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let detail = response.text().await.unwrap_or_default();
            bail!(format!("pull failed: HTTP {status} {detail}"));
        }
        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();
        use futures_util::StreamExt;
        'streaming: while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            buffer.extend_from_slice(&chunk);
            while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
                let line: Vec<u8> = buffer.drain(..=position).collect();
                let line = String::from_utf8_lossy(&line[..line.len() - 1]);
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let payload: Value = match serde_json::from_str(line) {
                    Ok(payload) => payload,
                    Err(_) => continue,
                };
                if let Some(error) = payload.get("error").and_then(Value::as_str) {
                    bail!("{error}");
                }
                let Some(status) = payload.get("status").and_then(Value::as_str) else {
                    continue;
                };
                // Ollama streams per-layer progress as separate completed/total
                // fields; render a percent when the raw status lacks one so the
                // caller can show sparse milestone lines.
                let text = match (
                    payload.get("completed").and_then(Value::as_u64),
                    payload.get("total").and_then(Value::as_u64),
                ) {
                    (Some(completed), Some(total)) if total > 0 && !status.contains('%') => {
                        let percent = (completed.min(total) * 100 / total) as usize;
                        format!("{status} {percent}%")
                    }
                    _ => status.to_string(),
                };
                on_status(&text);
                if status == "success" {
                    break 'streaming;
                }
            }
        }
        Ok(())
    }
}

/// A catalog entry from `GET /models`. Optional fields are populated when the
/// endpoint advertises them (Ollama's OpenAI-compatible catalog typically only
/// supplies `id`), and fall back to the configured defaults otherwise.
#[derive(Clone, Debug)]
pub struct ModelEntry {
    pub id: String,
    pub context_window: Option<u64>,
    pub input_modalities: Option<Vec<String>>,
    pub supports_tools: Option<bool>,
}

/// What Ollama's `/api/show` says about a model's reasoning support.
///
/// `has_thinking` comes from the `thinking` capability; `levels` and
/// `default_level` come from the chat template, which for effort-aware models
/// (e.g. qwen3.8) declares `reasoning_effort|default('xhigh')` and the accepted
/// level set. `gguf_arch` is set for native GGUF imports, whose think levels
/// the current Ollama builds may silently ignore (the template default wins).
#[derive(Clone, Debug, PartialEq)]
pub struct ThinkInfo {
    pub has_thinking: bool,
    /// Whether the model declares the `vision` capability (it accepts
    /// images besides text).
    pub vision: bool,
    /// Whether the model declares the `audio` capability (it accepts audio
    /// input besides text).
    pub audio: bool,
    pub levels: Vec<String>,
    pub default_level: Option<String>,
    pub gguf_arch: Option<String>,
    /// The model's declared maximum context (GGUF metadata, e.g.
    /// `qwen35.context_length`), when the snapshot exposes it.
    pub context_length: Option<u64>,
}

impl ThinkInfo {
    /// The valid Ctrl+T states for this model: off plus whatever the template
    /// declares (falling back to plain on/off when no levels are declared).
    pub fn cycle(&self) -> Vec<String> {
        if !self.has_thinking {
            return vec!["false".to_string()];
        }
        let mut states = vec!["false".to_string()];
        if self.levels.is_empty() {
            states.push("true".to_string());
        }
        for level in &self.levels {
            if !states.contains(level) {
                states.push(level.clone());
            }
        }
        states
    }
}

/// Extracts the declared reasoning-effort levels (and their default) from a
/// chat template: the `reasoning_effort|default('xhigh')` set clause plus the
/// `not in ('xhigh', 'medium', 'low')` membership test. Judgment calls are
/// conservative: only short quoted identifiers next to `reasoning_effort`
/// count, so unrelated `in (...)` clauses are never picked up.
fn template_think_levels(template: &str) -> (Vec<String>, Option<String>) {
    let mut levels: Vec<String> = Vec::new();
    let mut default_level: Option<String> = None;
    let mut search = 0usize;
    while let Some(found) = template[search..].find("reasoning_effort") {
        let at = search + found;
        search = at + "reasoning_effort".len();
        let window = &template[search..(search + 40).min(template.len())];
        if let Some(offset) = window.find("|default(") {
            let after = &template[search + offset + "|default(".len()..];
            default_level = first_quoted_token(after).or(default_level);
        }
        let window = &template[search..(search + 160).min(template.len())];
        if let Some(offset) = window.find("in (") {
            let after = &template[search + offset + "in (".len()..];
            if let Some(closing) = after.find(')') {
                for token in after[..closing].split(',') {
                    if let Some(value) = first_quoted_token(token) {
                        if !levels.contains(&value) {
                            levels.push(value);
                        }
                    }
                }
            }
        }
    }
    (levels, default_level)
}

/// Reads a short single-quoted or double-quoted identifier, or nothing.
fn first_quoted_token(text: &str) -> Option<String> {
    let trimmed = text.trim_start();
    let quote = trimmed.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let value: String = trimmed[1..].chars().take_while(|&c| c != quote).collect();
    if value.is_empty() || value.len() > 8 || value.contains('\\') {
        return None;
    }
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::next_think_level;
    use serde_json::json;

    fn assistant_with_calls(ids: &[&str]) -> Message {
        Message::assistant(
            Some(json!(null)),
            Some(
                ids.iter()
                    .map(|id| ToolCall {
                        id: id.to_string(),
                        kind: "function".to_string(),
                        function: FunctionCall {
                            name: "read_file".to_string(),
                            arguments: "{\"path\":\"a\"}".to_string(),
                        },
                    })
                    .collect(),
            ),
        )
    }

    #[test]
    fn recoverable_errors_are_classified_like_minagent() {
        let ended = anyhow::anyhow!(
            "[STREAM_ENDED] Endpoint stream ended before a complete response was received."
        );
        assert!(is_recoverable_endpoint_error(&ended, 0));
        assert!(!is_recoverable_endpoint_error(&ended, 5));
        let stream_err = anyhow::anyhow!("[STREAM_ERROR] Endpoint streaming error: boom");
        assert!(is_recoverable_endpoint_error(&stream_err, 0));
        assert!(!is_recoverable_endpoint_error(&stream_err, 1));
        let shape =
            anyhow::anyhow!("[ENDPOINT_SHAPE] The endpoint did not return a streaming response.");
        assert!(is_recoverable_endpoint_error(&shape, 12));
        let connect =
            anyhow::anyhow!("Could not connect to the OpenAI-compatible endpoint: timed out");
        assert!(is_recoverable_endpoint_error(&connect, 0));
        let rate_limited = anyhow::anyhow!("Endpoint returned HTTP 429: too many requests");
        assert!(is_recoverable_endpoint_error(&rate_limited, 0));
        let server_error = anyhow::anyhow!("Endpoint returned HTTP 503: unavailable");
        assert!(is_recoverable_endpoint_error(&server_error, 0));
        let bad_request = anyhow::anyhow!("Endpoint returned HTTP 400: bad request");
        assert!(!is_recoverable_endpoint_error(&bad_request, 0));
        let invalid_tool =
            anyhow::anyhow!("[INVALID_TOOL_CALL] Endpoint returned an incomplete tool call.");
        assert!(!is_recoverable_endpoint_error(&invalid_tool, 0));
    }

    #[test]
    fn repeated_tool_call_ids_are_renamed_with_their_results() {
        let messages = vec![
            assistant_with_calls(&["call_0"]),
            Message::tool("call_0", "one"),
            assistant_with_calls(&["call_0"]),
            Message::tool("call_0", "two"),
        ];
        let unique = unique_tool_call_ids(&messages);
        let first = unique[0].tool_calls.as_ref().unwrap()[0].id.clone();
        let second = unique[2].tool_calls.as_ref().unwrap()[0].id.clone();
        assert_eq!(first, "call_0");
        assert_eq!(second, "call_0__d1");
        assert_eq!(unique[1].tool_call_id.as_deref(), Some("call_0"));
        assert_eq!(unique[3].tool_call_id.as_deref(), Some("call_0__d1"));
    }

    #[test]
    fn unique_tool_call_ids_are_untouched() {
        let messages = vec![
            assistant_with_calls(&["call_a"]),
            Message::tool("call_a", "one"),
            assistant_with_calls(&["call_b"]),
            Message::tool("call_b", "two"),
        ];
        let unique = unique_tool_call_ids(&messages);
        assert_eq!(unique[0].tool_calls.as_ref().unwrap()[0].id, "call_a");
        assert_eq!(unique[2].tool_calls.as_ref().unwrap()[0].id, "call_b");
    }

    #[test]
    fn duplicate_tool_ids_are_detected() {
        let unique = vec![
            assistant_with_calls(&["call_a"]),
            Message::tool("call_a", "one"),
            assistant_with_calls(&["call_b"]),
            Message::tool("call_b", "two"),
        ];
        assert!(!has_duplicate_tool_call_ids(&unique));
        let repeated = vec![
            assistant_with_calls(&["call_0"]),
            Message::tool("call_0", "one"),
            assistant_with_calls(&["call_0"]),
            Message::tool("call_0", "two"),
        ];
        assert!(has_duplicate_tool_call_ids(&repeated));
    }

    #[test]
    fn ollama_messages_turn_arguments_into_objects() {
        let messages = vec![assistant_with_calls(&["call_0"])];
        let mapped = to_ollama_messages(&messages);
        assert!(mapped[0]["tool_calls"][0]["function"]["arguments"].is_object());
    }

    #[test]
    fn ollama_url_is_derived_from_openai_endpoint() {
        assert_eq!(
            ollama_chat_url("http://127.0.0.1:11434/v1/chat/completions").as_deref(),
            Some("http://127.0.0.1:11434/api/chat")
        );
        assert_eq!(
            ollama_chat_url("https://host.example/chat/completions").as_deref(),
            Some("https://host.example/api/chat")
        );
        assert_eq!(
            ollama_chat_url("https://host.example/v1/chat/completions").as_deref(),
            Some("https://host.example/api/chat")
        );
    }

    #[test]
    fn sse_stream_accumulates_content_and_tool_calls() {
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut calls = String::new();
        let mut on_text = |chunk: &str| text.push_str(chunk);
        let mut on_reasoning = |chunk: &str| reasoning.push_str(chunk);
        let mut on_tool_call = |name: &str| calls = name.to_string();
        let mut parser = StreamParser::new(
            &mut on_text,
            &mut on_reasoning,
            &mut on_tool_call,
            false,
            "text/event-stream".to_string(),
        );
        parser
            .feed_sse_frame("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}")
            .unwrap();
        parser
            .feed_sse_frame("data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}")
            .unwrap();
        parser
            .feed_sse_frame(
                "data: {\"choices\":[{\"delta\":{\"content\":\"lo\",\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}",
            )
            .unwrap();
        parser
            .feed_sse_frame(
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"a\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}",
            )
            .unwrap();
        parser.feed_sse_frame("data: [DONE]").unwrap();
        let completion = parser.finish().unwrap();
        assert_eq!(completion.message.text(), "hello");
        assert_eq!(completion.finish_reason.as_deref(), Some("tool_calls"));
        let call = &completion.message.tool_calls.as_ref().unwrap()[0];
        assert_eq!(call.function.name, "read_file");
        assert_eq!(call.function.arguments, "{\"path\":\"a\"}");
        assert_eq!(calls, "read_file");
        assert_eq!(text, "hello");
        assert_eq!(reasoning, "think");
    }

    #[test]
    fn sse_usage_is_parsed() {
        let mut on_text = |_chunk: &str| {};
        let mut on_reasoning = |_chunk: &str| {};
        let mut on_tool_call = |_chunk: &str| {};
        let mut parser = StreamParser::new(
            &mut on_text,
            &mut on_reasoning,
            &mut on_tool_call,
            false,
            "text/event-stream".to_string(),
        );
        parser
            .feed_sse_frame(
                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}",
            )
            .unwrap();
        parser.feed_sse_frame("data: [DONE]").unwrap();
        let completion = parser.finish().unwrap();
        assert_eq!(completion.usage.unwrap().completion_tokens, 2);
    }

    #[test]
    fn ndjson_replaces_complete_repeated_calls() {
        let mut on_text = |_chunk: &str| {};
        let mut on_reasoning = |_chunk: &str| {};
        let mut on_tool_call = |_chunk: &str| {};
        let mut parser = StreamParser::new(
            &mut on_text,
            &mut on_reasoning,
            &mut on_tool_call,
            true,
            "application/x-ndjson".to_string(),
        );
        parser
            .feed_ndjson_line(
                "{\"message\":{\"content\":\"\",\"tool_calls\":[{\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a\\\"}\"}}]}}",
            )
            .unwrap();
        parser
            .feed_ndjson_line(
                "{\"message\":{\"tool_calls\":[{\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a\\\"}\"}}]},\"done\":true,\"eval_count\":7,\"prompt_eval_count\":3}",
            )
            .unwrap();
        let completion = parser.finish().unwrap();
        let call = &completion.message.tool_calls.as_ref().unwrap()[0];
        assert_eq!(call.function.name, "read_file");
        assert_eq!(call.function.arguments, "{\"path\":\"a\"}");
        assert_eq!(completion.usage.unwrap().completion_tokens, 7);
    }

    #[test]
    fn ndjson_accumulates_fragmented_arguments() {
        let mut on_text = |_chunk: &str| {};
        let mut on_reasoning = |_chunk: &str| {};
        let mut on_tool_call = |_chunk: &str| {};
        let mut parser = StreamParser::new(
            &mut on_text,
            &mut on_reasoning,
            &mut on_tool_call,
            true,
            "application/x-ndjson".to_string(),
        );
        parser
            .feed_ndjson_line(
                "{\"message\":{\"tool_calls\":[{\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\"}}]}}",
            )
            .unwrap();
        parser
            .feed_ndjson_line(
                "{\"message\":{\"tool_calls\":[{\"function\":{\"arguments\":\"\\\"a\\\"}\"}}]},\"done\":true}",
            )
            .unwrap();
        let completion = parser.finish().unwrap();
        let call = &completion.message.tool_calls.as_ref().unwrap()[0];
        assert_eq!(call.function.name, "read_file");
        assert_eq!(call.function.arguments, "{\"path\":\"a\"}");
    }

    #[test]
    fn ndjson_error_as_first_line_is_endpoint_shape() {
        let mut on_text = |_chunk: &str| {};
        let mut on_reasoning = |_chunk: &str| {};
        let mut on_tool_call = |_chunk: &str| {};
        let mut parser = StreamParser::new(
            &mut on_text,
            &mut on_reasoning,
            &mut on_tool_call,
            true,
            "application/json".to_string(),
        );
        let error = parser
            .feed_ndjson_line("{\"error\":\"model not found\"}")
            .unwrap_err();
        assert!(error.to_string().contains("ENDPOINT_SHAPE"));
    }

    #[test]
    fn incomplete_stream_is_rejected() {
        let mut on_text = |_chunk: &str| {};
        let mut on_reasoning = |_chunk: &str| {};
        let mut on_tool_call = |_chunk: &str| {};
        let parser = StreamParser::new(
            &mut on_text,
            &mut on_reasoning,
            &mut on_tool_call,
            false,
            "text/event-stream".to_string(),
        );
        let error = parser.finish().unwrap_err();
        assert!(error.to_string().contains("STREAM_ENDED"));
    }

    #[tokio::test]
    async fn complete_streams_sse_from_a_mock_endpoint() {
        use crate::config::PermissionMode;
        use std::path::PathBuf;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock server");
        let address = listener.local_addr().expect("mock address");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut request = [0u8; 16 * 1024];
            let _ = socket.read(&mut request).await.expect("read request");
            let body = concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],",
                "\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}\n\n",
                "data: [DONE]\n\n"
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write response");
        });

        let config = Config {
            endpoint: format!("http://{address}/v1/chat/completions"),
            api_key: None,
            model: "test-model".to_string(),
            context_window: 8192,
            timeout_ms: 5_000,
            ollama_think: None,
            input_modalities: vec!["text".to_string()],
            terminal_mode: PermissionMode::Off,
            skills_mode: PermissionMode::Off,
            mcp_mode: PermissionMode::Off,
            root: PathBuf::from("."),
            workspace_name: "test".to_string(),
        };
        let client = Client::new(&config).expect("client");
        let mut text = String::new();
        let mut on_text = |chunk: &str| text.push_str(chunk);
        let mut on_reasoning = |_chunk: &str| {};
        let mut on_tool_call = |_name: &str| {};
        let completion = client
            .complete(
                &[Message::user("hi")],
                None,
                &mut on_text,
                &mut on_reasoning,
                &mut on_tool_call,
            )
            .await
            .expect("completion");
        server.await.expect("server task");
        assert_eq!(text, "hi");
        assert_eq!(completion.message.text(), "hi");
        assert_eq!(completion.usage.expect("usage").completion_tokens, 1);
    }

    /// A model that rejects the `think` option with HTTP 400 must be retried
    /// once without it, and later requests must omit `think` entirely.
    #[tokio::test]
    async fn unsupported_thinking_is_dropped_and_remembered() {
        use crate::config::PermissionMode;
        use std::path::PathBuf;
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;
        use tokio::sync::Mutex;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let bodies: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let server_bodies = Arc::clone(&bodies);
        let server = tokio::spawn(async move {
            const THINK_REJECT: &str =
                "{\"error\":\"\\\"ministral-3:14b\\\" does not support thinking\"}";
            const BODY: &str = "{\"message\":{\"content\":\"ok\"}}\n{\"done\":true,\"eval_count\":3,\"prompt_eval_count\":2}\n";
            const SECOND: &str = "{\"message\":{\"content\":\"second\"}}\n{\"done\":true,\"eval_count\":3,\"prompt_eval_count\":2}\n";
            // complete() now probes /api/show before the first chat request;
            // answer those with 404 so the capabilities stay unknown and the
            // 400-based recovery flow below runs unchanged. Chat requests:
            // 0 carries think -> 400; 1 drops it -> success; the memo keeps
            // request 2 think-less as well.
            let mut chats = 0u32;
            loop {
                let (mut socket, _) = listener.accept().await.expect("accept");
                let mut buffer = vec![0u8; 16 * 1024];
                let count = socket.read(&mut buffer).await.expect("read");
                let request = String::from_utf8_lossy(&buffer[..count]).to_string();
                if request.lines().next().unwrap_or("").contains("/api/show") {
                    socket
                        .write_all(
                            b"HTTP/1.1 404 Not Found\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
                        )
                        .await
                        .expect("write probe reply");
                    continue;
                }
                let body = request
                    .split("\r\n\r\n")
                    .nth(1)
                    .unwrap_or_default()
                    .to_string();
                let has_think = body.contains("\"think\"");
                server_bodies.lock().await.push(body);
                let response = if has_think {
                    format!(
                        "HTTP/1.1 400 Bad Request\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        THINK_REJECT.len(),
                        THINK_REJECT
                    )
                } else {
                    let payload = if chats == 2 { SECOND } else { BODY };
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson\r\ncontent-length: {}\r\n\r\n{}",
                        payload.len(),
                        payload
                    )
                };
                socket.write_all(response.as_bytes()).await.expect("write");
                chats += 1;
                if chats == 3 {
                    break;
                }
            }
        });

        // ollama_think is set, so the request goes to the native /api/chat path.
        let config = Config {
            endpoint: format!("http://{address}/v1/chat/completions"),
            api_key: None,
            model: "ministral-3:14b".to_string(),
            context_window: 8192,
            timeout_ms: 5_000,
            ollama_think: Some("true".to_string()),
            input_modalities: vec!["text".to_string()],
            terminal_mode: PermissionMode::Off,
            skills_mode: PermissionMode::Off,
            mcp_mode: PermissionMode::Off,
            root: PathBuf::from("."),
            workspace_name: "test".to_string(),
        };
        let client = Client::new(&config).expect("client");
        let mut text = String::new();
        let mut on_text = |chunk: &str| text.push_str(chunk);
        let mut on_reasoning = |_chunk: &str| {};
        let mut on_tool_call = |_name: &str| {};
        let completion = client
            .complete(
                &[Message::user("hi")],
                None,
                &mut on_text,
                &mut on_reasoning,
                &mut on_tool_call,
            )
            .await
            .expect("completion after dropping thinking");

        let bodies_view = bodies.lock().await;
        assert_eq!(bodies_view.len(), 2, "exactly one retry without thinking");
        assert!(
            bodies_view[0].contains("\"think\""),
            "first attempt sends think"
        );
        assert!(
            !bodies_view[1].contains("\"think\""),
            "retry must not send think; body: {}",
            bodies_view[1]
        );
        assert_eq!(completion.message.text(), "ok");
        // The session memo persists for further requests: a second call must
        // immediately succeed with a think-less body and no extra 400.
        drop(bodies_view);
        let mut again = String::new();
        let mut on_text_two = |chunk: &str| again.push_str(chunk);
        let mut on_reasoning_two = |_chunk: &str| {};
        let mut on_tool_call_two = |_name: &str| {};
        let completion_two = client
            .complete(
                &[Message::user("hi again")],
                None,
                &mut on_text_two,
                &mut on_reasoning_two,
                &mut on_tool_call_two,
            )
            .await
            .expect("second call");
        drop(completion_two);
        server.await.expect("server task");
        assert_eq!(again, "second");
        let bodies_view = bodies.lock().await;
        assert_eq!(
            bodies_view.len(),
            3,
            "three requests total, no extra retries"
        );
        assert!(
            !bodies_view[2].contains("\"think\""),
            "memo keeps think off"
        );
    }

    #[test]
    fn template_think_levels_are_parsed_from_the_chat_template() {
        let (levels, default) = template_think_levels(concat!(
            "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n",
            "{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}"
        ));
        assert_eq!(
            levels,
            &["xhigh".to_string(), "medium".to_string(), "low".to_string()]
        );
        assert_eq!(default.as_deref(), Some("xhigh"));
    }

    #[test]
    fn templates_without_effort_clause_have_no_levels() {
        let (levels, default) = template_think_levels("{{ Why normal reasoning? |do }}");
        assert!(levels.is_empty());
        assert_eq!(default, None);
    }

    #[test]
    fn think_info_cycle_respects_declared_levels() {
        let declared = ThinkInfo {
            has_thinking: true,
            vision: false,
            audio: false,
            levels: vec!["xhigh".to_string(), "medium".to_string(), "low".to_string()],
            default_level: Some("xhigh".to_string()),
            gguf_arch: Some("qwen35".to_string()),
            context_length: None,
        };
        assert_eq!(
            declared.cycle(),
            &[
                "false".to_string(),
                "xhigh".to_string(),
                "medium".to_string(),
                "low".to_string()
            ]
        );
        let plain = ThinkInfo {
            has_thinking: true,
            vision: false,
            audio: false,
            levels: Vec::new(),
            default_level: None,
            gguf_arch: None,
            context_length: None,
        };
        assert_eq!(plain.cycle(), &["false".to_string(), "true".to_string()]);
        let mute = ThinkInfo {
            has_thinking: false,
            vision: false,
            audio: false,
            levels: Vec::new(),
            default_level: None,
            gguf_arch: None,
            context_length: None,
        };
        assert_eq!(mute.cycle(), &["false".to_string()]);
        // Ctrl+T follows the declared cycle and wraps back to off.
        assert_eq!(
            next_think_level(Some("false"), Some(&declared.cycle())),
            Some("xhigh".to_string())
        );
        assert_eq!(
            next_think_level(Some("low"), Some(&declared.cycle())),
            Some("false".to_string())
        );
    }

    #[test]
    fn think_cycle_falls_back_to_the_generic_minagent_cycle() {
        assert_eq!(next_think_level(None, None), Some("false".to_string()));
        assert_eq!(
            next_think_level(Some("medium"), None),
            Some("high".to_string())
        );
        assert_eq!(
            next_think_level(Some("high"), None),
            Some("false".to_string())
        );
    }

    /// A model whose /api/show lacks the thinking capability must never send
    /// `think`: the request drops it up front instead of waiting for a 400.
    #[tokio::test]
    async fn unsupported_capability_suppresses_think_upfront() {
        use crate::config::PermissionMode;
        use std::path::PathBuf;
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;
        use tokio::sync::Mutex;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let bodies: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let server_bodies = Arc::clone(&bodies);
        let server = tokio::spawn(async move {
            const SHOW: &str = "{\"capabilities\":[\"completion\"],\"template\":\"\"}";
            const CHAT: &str = "{\"message\":{\"content\":\"ok\"}}\n{\"done\":true,\"eval_count\":1,\"prompt_eval_count\":1}\n";
            let mut chats = 0u32;
            loop {
                let (mut socket, _) = listener.accept().await.expect("accept");
                let mut buffer = vec![0u8; 16 * 1024];
                let count = socket.read(&mut buffer).await.expect("read");
                let request = String::from_utf8_lossy(&buffer[..count]).to_string();
                let response = if request.lines().next().unwrap_or("").contains("/api/show") {
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                        SHOW.len(),
                        SHOW
                    )
                } else {
                    server_bodies.lock().await.push(
                        request
                            .split("\r\n\r\n")
                            .nth(1)
                            .unwrap_or_default()
                            .to_string(),
                    );
                    chats += 1;
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson\r\ncontent-length: {}\r\n\r\n{}",
                        CHAT.len(),
                        CHAT
                    )
                };
                socket.write_all(response.as_bytes()).await.expect("write");
                if chats == 2 {
                    break;
                }
            }
        });

        let config = Config {
            endpoint: format!("http://{address}/v1/chat/completions"),
            api_key: None,
            model: "qwen-nothink".to_string(),
            context_window: 8192,
            timeout_ms: 5_000,
            ollama_think: Some("low".to_string()),
            input_modalities: vec!["text".to_string()],
            terminal_mode: PermissionMode::Off,
            skills_mode: PermissionMode::Off,
            mcp_mode: PermissionMode::Off,
            root: PathBuf::from("."),
            workspace_name: "test".to_string(),
        };
        let client = Client::new(&config).expect("client");
        for _call in 0..2u8 {
            let mut text = String::new();
            let mut on_text = |chunk: &str| text.push_str(chunk);
            let mut on_reasoning = |_chunk: &str| {};
            let mut on_tool_call = |_name: &str| {};
            let completion = client
                .complete(
                    &[Message::user("hi")],
                    None,
                    &mut on_text,
                    &mut on_reasoning,
                    &mut on_tool_call,
                )
                .await
                .expect("completion");
            assert_eq!(completion.message.text(), "ok");
        }
        server.await.expect("server task");
        let bodies_view = bodies.lock().await;
        assert_eq!(bodies_view.len(), 2, "two chat requests, no extra retries");
        for (index, body) in bodies_view.iter().enumerate() {
            assert!(
                !body.contains("\"think\""),
                "chat {index} must not carry think; body: {body}"
            );
        }
    }

    #[test]
    fn gpt_oss_action_tags_become_tool_calls_and_reasoning() {
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut calls = String::new();
        let mut on_text = |chunk: &str| text.push_str(chunk);
        let mut on_reasoning = |chunk: &str| reasoning.push_str(chunk);
        let mut on_tool_call = |name: &str| calls.push_str(name);
        let mut parser = StreamParser::new(
            &mut on_text,
            &mut on_reasoning,
            &mut on_tool_call,
            true,
            "application/x-ndjson".to_string(),
        );
        // Sent split across several NDJSON lines exactly like a real stream.
        parser
            .feed_ndjson_line("{\"message\":{\"content\":\"<|START_THINKING|>remaining \"}}")
            .unwrap();
        parser
            .feed_ndjson_line("{\"message\":{\"content\":\"two pages.<|END_THINKING|><|START_ACTION|>[{\\\"tool_call_id\\\":\\\"21\\\",\\\"tool_name\\\":\\\"write_file\\\",\\\"parameters\\\":{\\\"path\\\":\\\"r.html\\\",\\\"content\\\":\\\"hi\\\"}}]\"}}")
            .unwrap();
        parser
            .feed_ndjson_line("{\"message\":{\"content\":\"<|END_ACTION|>\"}}")
            .unwrap();
        parser
            .feed_ndjson_line("{\"done\":true,\"eval_count\":12,\"prompt_eval_count\":4}")
            .unwrap();
        let completion = parser.finish().unwrap();
        assert_eq!(text, "", "no action text should reach the chat");
        assert_eq!(reasoning, "remaining two pages.");
        assert_eq!(completion.message.text(), "");
        let tool_calls = completion.message.tool_calls.as_ref().expect("tool calls");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "21");
        assert_eq!(tool_calls[0].function.name, "write_file");
        let args: Value = serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
        assert_eq!(args["path"], "r.html");
        assert_eq!(args["content"], "hi");
        assert_eq!(calls, "write_file");
    }

    #[test]
    fn plain_text_with_stray_angle_prefix_still_streams() {
        let mut text = String::new();
        let mut on_text = |chunk: &str| text.push_str(chunk);
        let mut on_reasoning = |_chunk: &str| {};
        let mut on_tool_call = |_name: &str| {};
        let mut parser = StreamParser::new(
            &mut on_text,
            &mut on_reasoning,
            &mut on_tool_call,
            false,
            "text/event-stream".to_string(),
        );
        parser
            .feed_sse_frame("data: {\"choices\":[{\"delta\":{\"content\":\"a < b and c > d\"}}]}")
            .unwrap();
        parser.feed_sse_frame("data: [DONE]").unwrap();
        let completion = parser.finish().unwrap();
        assert_eq!(completion.message.text(), "a < b and c > d");
        assert_eq!(text, "a < b and c > d");
    }

    #[test]
    fn unclosed_action_is_still_parsed_at_stream_end() {
        let mut text = String::new();
        let mut on_text = |chunk: &str| text.push_str(chunk);
        let mut on_reasoning = |_chunk: &str| {};
        let mut on_tool_call = |_name: &str| {};
        let mut parser = StreamParser::new(
            &mut on_text,
            &mut on_reasoning,
            &mut on_tool_call,
            true,
            "application/x-ndjson".to_string(),
        );
        // No <|END_ACTION|> arrives; the buffered action should still be parsed.
        parser
            .feed_ndjson_line("{\"message\":{\"content\":\"<|START_ACTION|>[{\\\"tool_name\\\":\\\"read_file\\\",\\\"parameters\\\":{\\\"path\\\":\\\"a.txt\\\"}}]\"}}")
            .unwrap();
        parser
            .feed_ndjson_line("{\"done\":true,\"eval_count\":5,\"prompt_eval_count\":2}")
            .unwrap();
        let completion = parser.finish().unwrap();
        assert_eq!(text, "");
        let tool_calls = completion.message.tool_calls.as_ref().expect("tool calls");
        assert_eq!(tool_calls[0].function.name, "read_file");
        assert!(tool_calls[0].function.arguments.contains("a.txt"));
    }

    /// `/api/pull` NDJSON streaming: per-event statuses with completed/total
    /// render a percent, the stream's `error` field fails the pull, and the
    /// final `success` closes the call.
    #[tokio::test]
    async fn pull_model_renders_progress_and_succeeds() {
        use crate::config::PermissionMode;
        use std::path::PathBuf;
        use std::sync::Arc;
        use std::sync::Mutex as StdMutex;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let bodies: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
        let server_bodies = Arc::clone(&bodies);
        let server = tokio::spawn(async move {
            const PULL: &str = "{\"status\":\"pulling manifest\"}\n\
                {\"status\":\"pulling abc123\",\"completed\":25,\"total\":100}\n\
                {\"status\":\"pulling abc123 55%\"}\n\
                {\"status\":\"success\"}\n";
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut buffer = vec![0u8; 16 * 1024];
            let count = socket.read(&mut buffer).await.expect("read");
            server_bodies
                .lock()
                .expect("bodies")
                .push(String::from_utf8_lossy(&buffer[..count]).to_string());
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson\r\ncontent-length: {}\r\n\r\n{}",
                PULL.len(),
                PULL
            );
            socket.write_all(response.as_bytes()).await.expect("write");
        });

        let config = Config {
            endpoint: format!("http://{address}/v1/chat/completions"),
            api_key: None,
            model: "qwen3:0.6b".to_string(),
            context_window: 8192,
            timeout_ms: 5_000,
            ollama_think: None,
            input_modalities: vec!["text".to_string()],
            terminal_mode: PermissionMode::Off,
            skills_mode: PermissionMode::Off,
            mcp_mode: PermissionMode::Off,
            root: PathBuf::from("."),
            workspace_name: "test".to_string(),
        };
        let client = Client::new(&config).expect("client");
        let events: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
        let recorder = events.clone();
        client
            .pull_model("qwen3:0.6b", &mut |text| {
                recorder.lock().expect("events").push(text.to_string());
            })
            .await
            .expect("pull");
        server.await.expect("server task");

        let body_view = bodies.lock().expect("bodies");
        assert!(
            body_view[0].starts_with("POST /api/pull"),
            "pull must use the native API: {}",
            body_view[0]
        );
        assert!(
            body_view[0].contains("\"name\":\"qwen3:0.6b\""),
            "pull body carries the model name: {}",
            body_view[0]
        );
        let events_view = events.lock().expect("events");
        assert_eq!(
            *events_view,
            vec![
                "pulling manifest".to_string(),
                "pulling abc123 25%".to_string(),
                "pulling abc123 55%".to_string(),
                "success".to_string(),
            ]
        );
    }

    /// A stream `error` field must abort the pull with that message.
    #[tokio::test]
    async fn pull_model_surfaces_stream_errors() {
        use crate::config::PermissionMode;
        use std::path::PathBuf;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let server = tokio::spawn(async move {
            const PULL: &str = "{\"status\":\"pulling manifest\"}\n{\"error\":\"pull model manifest not found\"}\n";
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut buffer = vec![0u8; 16 * 1024];
            let _count = socket.read(&mut buffer).await.expect("read");
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson\r\ncontent-length: {}\r\n\r\n{}",
                PULL.len(),
                PULL
            );
            socket.write_all(response.as_bytes()).await.expect("write");
        });

        let config = Config {
            endpoint: format!("http://{address}/v1/chat/completions"),
            api_key: None,
            model: "qwen3:0.6b".to_string(),
            context_window: 8192,
            timeout_ms: 5_000,
            ollama_think: None,
            input_modalities: vec!["text".to_string()],
            terminal_mode: PermissionMode::Off,
            skills_mode: PermissionMode::Off,
            mcp_mode: PermissionMode::Off,
            root: PathBuf::from("."),
            workspace_name: "test".to_string(),
        };
        let client = Client::new(&config).expect("client");
        let mut seen = 0usize;
        let error = client
            .pull_model("nope", &mut |_| seen += 1)
            .await
            .expect_err("pull must fail");
        server.await.expect("server task");
        assert_eq!(seen, 1, "events before the error still surface");
        assert!(
            error.to_string().contains("pull model manifest not found"),
            "error carries the stream message: {error}"
        );
    }
}

#[cfg(test)]
mod context_show_tests {
    use super::*;
    use crate::config::PermissionMode;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::Mutex as StdMutex;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// /api/show metadata carries the declared context length under an
    /// architecture-prefixed key; the snapshot exposes it alongside thinking.
    #[tokio::test]
    async fn show_snapshot_exposes_the_declared_context_length() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let bodies: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
        let server_bodies = Arc::clone(&bodies);
        let server = tokio::spawn(async move {
            const SHOW: &str = "{\"capabilities\":[\"thinking\",\"vision\",\"audio\",\"tools\"],\
                \"model_info\":{\"qwen35.context_length\":262144,\"qwen35.architecture\":\"qwen35\"},\
                \"template\":\"\"}";
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut buffer = vec![0u8; 16 * 1024];
            let count = socket.read(&mut buffer).await.expect("read");
            server_bodies
                .lock()
                .expect("bodies")
                .push(String::from_utf8_lossy(&buffer[..count]).to_string());
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                SHOW.len(),
                SHOW
            );
            socket.write_all(response.as_bytes()).await.expect("write");
        });

        let config = Config {
            endpoint: format!("http://{address}/v1/chat/completions"),
            api_key: None,
            model: "qwen3.5:9b".to_string(),
            context_window: 8192,
            timeout_ms: 5_000,
            ollama_think: None,
            input_modalities: vec!["text".to_string()],
            terminal_mode: PermissionMode::Off,
            skills_mode: PermissionMode::Off,
            mcp_mode: PermissionMode::Off,
            root: PathBuf::from("."),
            workspace_name: "test".to_string(),
        };
        let client = Client::new(&config).expect("client");
        let info = client.think_info_for("qwen3.5:9b").await.expect("snapshot");
        server.await.expect("server task");
        assert!(info.has_thinking);
        assert!(info.vision);
        assert!(info.audio);
        assert_eq!(info.gguf_arch.as_deref(), Some("qwen35"));
        assert_eq!(info.context_length, Some(262_144));
    }
}
