use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;

use crate::config::PermissionMode;
use crate::footer::Footer;
use crate::input::{self, Input};
use crate::prompt::{EditorEvent, Session};
use crate::ui;

/// Outcome of a tool call. `aborted` is set when the user pressed Esc while a
/// long-running tool (the terminal command) was executing.
pub struct ToolOutcome {
    pub result: Result<String>,
    pub aborted: bool,
}

pub const MAX_READ_LINES: usize = 2_000;
pub const MAX_READ_OUTPUT_BYTES: usize = 96 * 1024;
pub const MAX_WRITE_BYTES: usize = 10 * 1024 * 1024;
pub const MAX_DIRECTORY_ENTRIES: usize = 10_000;
pub const MAX_DIRECTORY_OUTPUT_BYTES: usize = 64 * 1024;
pub const MAX_TOOL_CALLS: usize = 16;
const MAX_SEARCH_ENTRIES: usize = 10_000;
const MAX_SEARCH_RESULTS: usize = 500;
const MAX_SEARCH_READ_BYTES: u64 = 10 * 1024 * 1024;
const EXCLUDED_DIRECTORIES: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".next",
    ".cache",
    "dist",
    "build",
    "coverage",
];

/// Tool definitions advertised to the model. `run_terminal` is only offered when
/// terminal access is enabled, matching MinAgent.
pub fn definitions(terminal_mode: PermissionMode) -> Vec<Value> {
    let mut tools = vec![
        json!({
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Recursively search literal text in workspace filenames, UTF-8 contents, or both. Returns paths/matching lines, not full files; excludes generated directories and reports limits/omissions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "minLength": 1, "description": "Single-line literal; no regex/glob" },
                        "mode": { "type": "string", "enum": ["filename", "content", "both"], "default": "both" },
                        "path": { "type": "string", "description": "Workspace directory; default: root" },
                        "case_sensitive": { "type": "boolean", "default": false },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 500, "default": 100 }
                    },
                    "required": ["query"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read workspace files or explicitly user-named outside files; outside directories cannot be listed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "offset": { "type": "integer", "minimum": 1, "description": "1-based first line" },
                        "limit": { "type": "integer", "minimum": 1, "description": "Maximum returned lines" },
                        "column": { "type": "integer", "minimum": 1, "description": "1-based column; follow continuation for long lines" }
                    },
                    "required": ["path"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List immediate workspace entries (including hidden; root by default). Never follow links; raise limit if truncated.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 10000, "description": "Maximum entries; default: 500" }
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace one unique, exact block in a workspace file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "old_text": { "type": "string", "description": "Exact text to replace" },
                        "new_text": { "type": "string" }
                    },
                    "required": ["path", "old_text", "new_text"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or replace a workspace file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" }
                    },
                    "required": ["path", "content"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "delete_file",
                "description": "Delete one regular file inside the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "delete_directory",
                "description": "Recursively delete a workspace subdirectory; linked/special entries are blocked.",
                "parameters": {
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }
            }
        }),
    ];
    if terminal_mode != PermissionMode::Off {
        tools.push(json!({
            "type": "function",
            "function": {
                "name": "run_terminal",
                "description": "Run a shell command in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": { "command": { "type": "string" } },
                    "required": ["command"]
                }
            }
        }));
    }
    tools
}

pub fn tool_names(tools: &[Value]) -> Vec<String> {
    tools
        .iter()
        .filter_map(|tool| tool.pointer("/function/name").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

/// Parses and validates tool arguments against the advertised schema. This mirrors
/// MinAgent's local-tool validation (unknown keys, required keys, primitive types).
pub fn validate_arguments(name: &str, arguments: &str, tools: &[Value]) -> Result<Value> {
    let definition = tools
        .iter()
        .find(|tool| tool.pointer("/function/name").and_then(Value::as_str) == Some(name));
    let Some(definition) = definition else {
        let names = tool_names(tools);
        let close = names
            .iter()
            .find(|candidate| candidate.contains(name) || name.contains(candidate.as_str()));
        let hint = match close {
            Some(close) => format!(" Did you mean '{close}'?"),
            None => format!(" Advertised tools: {}.", names.join(", ")),
        };
        bail!("[UNAVAILABLE_TOOL] Tool '{name}' is unavailable.{hint}");
    };
    let args: Value = serde_json::from_str(arguments)
        .map_err(|_| anyhow!("[INVALID_ARGUMENTS] Tool arguments must be valid JSON."))?;
    if !args.is_object() {
        bail!("[INVALID_ARGUMENTS] Tool arguments must be a JSON object.");
    }
    let schema = definition
        .pointer("/function/parameters")
        .unwrap_or(&Value::Null);
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for key in required {
        let Some(key) = key.as_str() else { continue };
        if !args.get(key).is_some() {
            bail!("[INVALID_ARGUMENTS] Missing required argument: {key}.");
        }
    }
    let properties = schema.get("properties").and_then(Value::as_object);
    if let Some(properties) = properties {
        for (key, value) in args.as_object().expect("object checked above") {
            let Some(rule) = properties.get(key) else {
                bail!("[INVALID_ARGUMENTS] Unknown argument: {key}.");
            };
            if let Some(kind) = rule.get("type").and_then(Value::as_str) {
                let valid = match kind {
                    "string" => value.is_string(),
                    "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
                    "boolean" => value.is_boolean(),
                    "object" => value.is_object(),
                    "array" => value.is_array(),
                    _ => true,
                };
                if !valid {
                    bail!("[INVALID_ARGUMENTS] {key} must be {kind}.");
                }
            }
            if let Some(values) = rule.get("enum").and_then(Value::as_array) {
                if !values.iter().any(|candidate| candidate == value) {
                    let allowed = values
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(", ");
                    bail!("[INVALID_ARGUMENTS] {key} must be one of {allowed}.");
                }
            }
            let numeric = value.as_f64();
            if let (Some(minimum), Some(number)) =
                (rule.get("minimum").and_then(Value::as_f64), numeric)
            {
                if number < minimum {
                    bail!("[INVALID_ARGUMENTS] {key} is outside the advertised limits.");
                }
            }
            if let (Some(maximum), Some(number)) =
                (rule.get("maximum").and_then(Value::as_f64), numeric)
            {
                if number > maximum {
                    bail!("[INVALID_ARGUMENTS] {key} is outside the advertised limits.");
                }
            }
            if let (Some(minimum), Some(text)) = (
                rule.get("minLength").and_then(Value::as_u64),
                value.as_str(),
            ) {
                if (text.chars().count() as u64) < minimum {
                    bail!("[INVALID_ARGUMENTS] {key} is outside the advertised limits.");
                }
            }
        }
    }
    Ok(args)
}

fn required_str<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("[INVALID_ARGUMENTS] {key} must be a string."))
}

fn optional_usize(args: &Value, key: &str, fallback: usize) -> Result<usize> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(fallback),
        Some(value) => match value.as_u64() {
            Some(number) if number >= 1 => Ok(number as usize),
            _ => bail!("[INVALID_ARGUMENTS] {key} must be an integer of at least 1."),
        },
    }
}

fn normalize(path: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                result.pop();
            }
            Component::CurDir => {}
            other => result.push(other.as_os_str()),
        }
    }
    result
}

fn is_within(root: &Path, path: &Path) -> bool {
    path.starts_with(root)
}

fn ensure_within(root: &Path, path: &Path) -> Result<()> {
    let existing = if path.exists() {
        fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
    } else {
        match path.parent() {
            Some(parent) if parent.exists() => fs::canonicalize(parent)
                .map(|parent| parent.join(path.file_name().unwrap_or_default()))
                .unwrap_or_else(|_| path.to_path_buf()),
            _ => path.to_path_buf(),
        }
    };
    if !is_within(root, &existing) {
        bail!("[TOOL_ERROR] Path resolved outside the current workspace.");
    }
    Ok(())
}

/// Resolves a path inside the workspace. Absolute paths that escape the workspace are
/// rejected for every tool except `read_file` (which mirrors MinAgent's explicit
/// user-named outside read).
fn resolve_workspace_path(root: &Path, input: &str) -> Result<PathBuf> {
    let candidate = Path::new(input);
    let joined = if candidate.is_absolute() {
        normalize(candidate)
    } else {
        normalize(&root.join(candidate))
    };
    ensure_within(root, &joined)?;
    Ok(joined)
}

fn resolve_read_path(root: &Path, input: &str) -> Result<PathBuf> {
    let candidate = Path::new(input);
    if candidate.is_absolute() {
        return Ok(normalize(candidate));
    }
    resolve_workspace_path(root, input)
}

fn display_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .map(|relative| {
            let text = relative.to_string_lossy().to_string();
            if text.is_empty() {
                ".".to_string()
            } else {
                text
            }
        })
        .unwrap_or_else(|_| path.to_string_lossy().to_string())
}

fn json_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

fn read_file(root: &Path, args: &Value) -> Result<String> {
    let path_text = required_str(args, "path")?;
    let target = resolve_read_path(root, path_text)?;
    let metadata = fs::metadata(&target)
        .with_context(|| format!("[TOOL_ERROR] Could not read {path_text}."))?;
    if metadata.is_dir() {
        bail!("[TOOL_ERROR] read_file requires a file: {path_text}.");
    }
    if metadata.len() > MAX_WRITE_BYTES as u64 {
        bail!("[TOOL_ERROR] File exceeds the 10 MiB read limit.");
    }
    let bytes =
        fs::read(&target).with_context(|| format!("[TOOL_ERROR] Could not read {path_text}."))?;
    let content = String::from_utf8(bytes).map_err(|_| {
        anyhow!("[TOOL_ERROR] File is not valid UTF-8 text; read_file handles text only.")
    })?;
    let lines: Vec<&str> = content.split('\n').collect();
    let offset = optional_usize(args, "offset", 1)?;
    let column = optional_usize(args, "column", 1)?;
    let limit = optional_usize(args, "limit", MAX_READ_LINES)?.min(MAX_READ_LINES);
    if offset > lines.len() {
        bail!(
            "[TOOL_ERROR] offset is beyond the end of the file ({} lines).",
            lines.len()
        );
    }
    let content_budget = MAX_READ_OUTPUT_BYTES - 160;
    let mut output = String::new();
    let mut output_bytes = 0usize;
    let mut returned = 0usize;
    for line_index in (offset - 1)..lines.len().min(offset - 1 + limit) {
        let line = lines[line_index];
        let start_column = if line_index == offset - 1 { column } else { 1 };
        let mut fragment = String::new();
        let mut fragment_bytes = 0usize;
        let mut current_column = 1usize;
        for character in line.chars() {
            if current_column < start_column {
                current_column += 1;
                continue;
            }
            let character_bytes = character.len_utf8();
            let prefix_bytes = if returned == 0 { 0 } else { 1 };
            if output_bytes + prefix_bytes + fragment_bytes + character_bytes > content_budget {
                output.push_str(&fragment);
                output.push_str(&format!(
                    "\n\n[Read stopped at the output limit. Continue with offset={}, column={}.]",
                    line_index + 1,
                    current_column
                ));
                return Ok(output);
            }
            fragment.push(character);
            fragment_bytes += character_bytes;
            current_column += 1;
        }
        if start_column > current_column {
            bail!(
                "[TOOL_ERROR] column is beyond the end of line {}.",
                line_index + 1
            );
        }
        if returned > 0 {
            output.push('\n');
            output_bytes += 1;
        }
        output.push_str(&fragment);
        output_bytes += fragment_bytes;
        returned += 1;
    }
    let next_offset = offset + returned;
    if next_offset <= lines.len() {
        output.push_str(&format!(
            "\n\n[{} more lines. Continue with offset={}.]",
            lines.len() - next_offset + 1,
            next_offset
        ));
    }
    Ok(output)
}

fn list_directory(root: &Path, args: &Value) -> Result<String> {
    let input = args.get("path").and_then(Value::as_str).unwrap_or(".");
    let limit = optional_usize(args, "limit", 500)?;
    if limit > MAX_DIRECTORY_ENTRIES {
        bail!("[INVALID_ARGUMENTS] limit must be an integer from 1 to {MAX_DIRECTORY_ENTRIES}.");
    }
    let target = resolve_workspace_path(root, input)?;
    let metadata = fs::symlink_metadata(&target).map_err(|error| {
        anyhow!(
            "[TOOL_ERROR] Could not inspect directory {}: {error}.",
            json_string(input)
        )
    })?;
    if metadata.file_type().is_symlink() {
        bail!(
            "[TOOL_ERROR] Symbolic links and junctions are blocked to keep file access inside the workspace."
        );
    }
    if !metadata.is_dir() {
        bail!(
            "[TOOL_ERROR] list_directory requires a directory: {}.",
            json_string(input)
        );
    }
    let display = display_path(root, &target);
    let mut entries: Vec<(String, &'static str, bool)> = Vec::new();
    for entry in fs::read_dir(&target).map_err(|error| {
        anyhow!(
            "[TOOL_ERROR] Could not list directory {}: {error}.",
            json_string(&display)
        )
    })? {
        let entry = entry.map_err(|error| {
            anyhow!(
                "[TOOL_ERROR] Could not list directory {}: {error}.",
                json_string(&display)
            )
        })?;
        let file_type = entry.file_type().map_err(|error| {
            anyhow!(
                "[TOOL_ERROR] Could not inspect {}: {error}.",
                entry.path().display()
            )
        })?;
        let kind = if file_type.is_symlink() {
            "LINK, not traversed"
        } else if file_type.is_dir() {
            "DIR"
        } else if file_type.is_file() {
            "FILE"
        } else {
            "SPECIAL, not readable"
        };
        entries.push((
            entry.file_name().to_string_lossy().to_string(),
            kind,
            file_type.is_dir(),
        ));
    }
    entries.sort_by(|left, right| {
        left.0
            .to_lowercase()
            .cmp(&right.0.to_lowercase())
            .then_with(|| left.0.cmp(&right.0))
    });
    let total = entries.len();
    if total == 0 {
        return Ok(format!(
            "Directory: {}\n(empty directory)",
            json_string(&display)
        ));
    }
    let mut output_lines = vec![format!("Directory: {}", json_string(&display))];
    let mut output_bytes = output_lines[0].len();
    let mut shown = 0usize;
    for (name, kind, is_dir) in &entries {
        let line = format!("[{kind}] {name}{}", if *is_dir { "/" } else { "" });
        let line_bytes = line.len() + 1;
        if output_bytes + line_bytes > MAX_DIRECTORY_OUTPUT_BYTES - 512 {
            break;
        }
        output_lines.push(line);
        output_bytes += line_bytes;
        shown += 1;
    }
    let omitted = total - shown;
    if omitted > 0 {
        if limit == MAX_DIRECTORY_ENTRIES {
            output_lines.push(format!(
                "[Showing {shown} of {total} entries; list a subdirectory to inspect the remainder beyond the {MAX_DIRECTORY_ENTRIES}-entry limit.]"
            ));
        } else {
            output_lines.push(format!(
                "[Showing {shown} of {total} entries. Call list_directory with a larger limit to see more.]"
            ));
        }
    }
    Ok(output_lines.join("\n"))
}

fn edit_file(root: &Path, args: &Value) -> Result<String> {
    let path_text = required_str(args, "path")?;
    let old_text = required_str(args, "old_text")?;
    let new_text = required_str(args, "new_text")?;
    if old_text.is_empty() {
        bail!("[INVALID_ARGUMENTS] old_text must be a non-empty string.");
    }
    if old_text.len() > MAX_WRITE_BYTES || new_text.len() > MAX_WRITE_BYTES {
        bail!(
            "[TOOL_ERROR] old_text and new_text must each fit within the 10 MiB byte edit limit."
        );
    }
    let target = resolve_workspace_path(root, path_text)?;
    let metadata = fs::symlink_metadata(&target)
        .map_err(|_| anyhow!("[TOOL_ERROR] Could not read {path_text}."))?;
    if metadata.file_type().is_symlink() {
        bail!(
            "[TOOL_ERROR] Symbolic links and junctions are blocked to keep file access inside the workspace."
        );
    }
    if !metadata.is_file() {
        bail!("[TOOL_ERROR] edit_file requires a regular file: {path_text}.");
    }
    let content = fs::read_to_string(&target).map_err(|_| {
        anyhow!("[TOOL_ERROR] Could not read {path_text}; it must be valid UTF-8 text.")
    })?;
    let Some(first) = content.find(old_text) else {
        bail!(
            "[TOOL_ERROR] old_text not found in {path_text}; no changes made. Reread it, then rebuild the edit."
        );
    };
    if content[first + old_text.len()..].contains(old_text) {
        bail!(
            "[TOOL_ERROR] old_text is not unique in {path_text}; no changes made. Reread it and choose a unique block."
        );
    }
    let changed = format!(
        "{}{}{}",
        &content[..first],
        new_text,
        &content[first + old_text.len()..]
    );
    fs::write(&target, changed)
        .with_context(|| format!("[TOOL_ERROR] Could not write {path_text}."))?;
    Ok(format!("Updated {path_text}."))
}

fn write_file(root: &Path, args: &Value) -> Result<String> {
    let path_text = required_str(args, "path")?;
    let content = required_str(args, "content")?;
    if content.len() > MAX_WRITE_BYTES {
        bail!("[TOOL_ERROR] File content exceeds the 10 MiB byte write limit.");
    }
    let target = resolve_workspace_path(root, path_text)?;
    if let Ok(metadata) = fs::symlink_metadata(&target) {
        if metadata.file_type().is_symlink() {
            bail!(
                "[TOOL_ERROR] Symbolic links and junctions are blocked to keep file access inside the workspace."
            );
        }
        if metadata.is_dir() {
            bail!("[TOOL_ERROR] The target path is a directory.");
        }
        if !metadata.is_file() {
            bail!("[TOOL_ERROR] Only regular files can be overwritten.");
        }
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!("[TOOL_ERROR] Could not create parent folders for {path_text}.")
        })?;
    }
    ensure_within(root, &target)?;
    fs::write(&target, content)
        .with_context(|| format!("[TOOL_ERROR] Could not write {path_text}."))?;
    Ok(format!("Wrote {path_text}."))
}

fn delete_file(root: &Path, args: &Value) -> Result<String> {
    let path_text = required_str(args, "path")?;
    let target = resolve_workspace_path(root, path_text)?;
    let metadata = fs::symlink_metadata(&target)
        .map_err(|_| anyhow!("[TOOL_ERROR] Could not find {path_text}."))?;
    if metadata.is_dir() {
        bail!("[TOOL_ERROR] Directories cannot be deleted. delete_file removes files only.");
    }
    if metadata.file_type().is_symlink() {
        bail!(
            "[TOOL_ERROR] Symbolic links and junctions are blocked to keep file access inside the workspace."
        );
    }
    if !metadata.is_file() {
        bail!("[TOOL_ERROR] Only regular files can be deleted.");
    }
    fs::remove_file(&target)
        .with_context(|| format!("[TOOL_ERROR] Could not delete {path_text}."))?;
    Ok(format!("Deleted {path_text}."))
}

fn validate_deletable_directory(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        bail!(
            "[TOOL_ERROR] Symbolic links and junctions are blocked inside deletable directories."
        );
    }
    if metadata.is_dir() {
        for child in fs::read_dir(path)? {
            validate_deletable_directory(&child?.path())?;
        }
        return Ok(());
    }
    if !metadata.is_file() {
        bail!("[TOOL_ERROR] Directories containing special files cannot be deleted.");
    }
    Ok(())
}

fn delete_directory(root: &Path, args: &Value) -> Result<String> {
    let path_text = required_str(args, "path")?;
    let target = resolve_workspace_path(root, path_text)?;
    if target == root {
        bail!("[TOOL_ERROR] The workspace root cannot be deleted.");
    }
    let metadata = fs::symlink_metadata(&target)
        .map_err(|_| anyhow!("[TOOL_ERROR] Could not find {path_text}."))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        bail!("[TOOL_ERROR] delete_directory only works on a regular subdirectory.");
    }
    validate_deletable_directory(&target)?;
    fs::remove_dir_all(&target)
        .with_context(|| format!("[TOOL_ERROR] Could not delete {path_text}."))?;
    Ok(format!("Deleted directory {path_text} and its contents."))
}

fn snippet(line: &str, match_char_index: usize) -> String {
    let characters: Vec<char> = line.chars().collect();
    let start = match_char_index.saturating_sub(120);
    let end = (start + 400).min(characters.len());
    let body: String = characters[start..end].iter().collect();
    format!(
        "{}{}{}",
        if start > 0 { "…" } else { "" },
        body,
        if end < characters.len() { "…" } else { "" }
    )
}

fn search_files(root: &Path, args: &Value) -> Result<String> {
    let query = required_str(args, "query")?;
    if query.trim().is_empty() || query.contains('\n') || query.contains('\r') {
        bail!("[INVALID_ARGUMENTS] query must be a single-line non-empty literal.");
    }
    let mode = args.get("mode").and_then(Value::as_str).unwrap_or("both");
    if !matches!(mode, "filename" | "content" | "both") {
        bail!("[INVALID_ARGUMENTS] mode must be one of filename, content, both.");
    }
    let case_sensitive = args
        .get("case_sensitive")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let limit = optional_usize(args, "limit", 100)?.min(MAX_SEARCH_RESULTS);
    let base_input = args.get("path").and_then(Value::as_str).unwrap_or(".");
    let base = resolve_workspace_path(root, base_input)?;
    if !base.is_dir() {
        bail!("[TOOL_ERROR] search path must be a workspace directory: {base_input}.");
    }
    let needle = if case_sensitive {
        query.to_string()
    } else {
        query.to_lowercase()
    };
    let mut results: Vec<String> = Vec::new();
    let mut entries_examined = 0usize;
    let mut files_examined = 0usize;
    let mut files_read = 0usize;
    let mut incomplete = false;
    let mut reasons: Vec<&str> = Vec::new();

    let walker = walkdir::WalkDir::new(&base)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| {
            if entry.file_type().is_dir() {
                let name = entry.file_name().to_string_lossy().to_lowercase();
                !EXCLUDED_DIRECTORIES.contains(&name.as_str())
            } else {
                true
            }
        });
    for entry in walker {
        if results.len() >= limit {
            incomplete = true;
            reasons.push("result limit");
            break;
        }
        if entries_examined >= MAX_SEARCH_ENTRIES {
            incomplete = true;
            reasons.push("entry limit");
            break;
        }
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => {
                incomplete = true;
                reasons.push("read error");
                continue;
            }
        };
        entries_examined += 1;
        if entry.file_type().is_dir() {
            continue;
        }
        if !entry.file_type().is_file() {
            continue;
        }
        files_examined += 1;
        let relative = display_path(root, entry.path());
        if mode != "content" {
            let name = entry.file_name().to_string_lossy();
            let lowered;
            let haystack = if case_sensitive {
                name.as_ref()
            } else {
                lowered = name.to_lowercase();
                lowered.as_str()
            };
            if haystack.contains(&needle) {
                results.push(format!("[filename] {}", json_string(&relative)));
            }
        }
        if mode == "filename" {
            continue;
        }
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(_) => {
                incomplete = true;
                reasons.push("read error");
                continue;
            }
        };
        if metadata.len() > MAX_SEARCH_READ_BYTES {
            incomplete = true;
            reasons.push("file size limit");
            continue;
        }
        let bytes = match fs::read(entry.path()) {
            Ok(bytes) => bytes,
            Err(_) => {
                incomplete = true;
                reasons.push("read error");
                continue;
            }
        };
        if bytes.contains(&0) {
            continue;
        }
        let text = match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(_) => continue,
        };
        files_read += 1;
        for (line_index, line) in text.split('\n').enumerate() {
            let line = line.strip_suffix('\r').unwrap_or(line);
            let lowered;
            let haystack = if case_sensitive {
                line
            } else {
                lowered = line.to_lowercase();
                lowered.as_str()
            };
            let Some(byte_index) = haystack.find(&needle) else {
                continue;
            };
            let column = haystack[..byte_index].chars().count() + 1;
            let safe_snippet = snippet(line, column - 1);
            results.push(format!(
                "[content] {}:{}:{}\n  {}",
                json_string(&relative),
                line_index + 1,
                column,
                json_string(&safe_snippet)
            ));
            if results.len() >= limit {
                break;
            }
        }
    }

    let status = if incomplete { "incomplete" } else { "complete" };
    let summary = format!(
        "{} results · {files_examined} files examined · {files_read} files read · {status}",
        results.len()
    );
    let metadata = json!({
        "status": status,
        "reasons": reasons,
        "stats": {
            "entriesExamined": entries_examined,
            "filesExamined": files_examined,
            "filesRead": files_read
        },
        "returnedResults": results.len(),
        "exclusions": EXCLUDED_DIRECTORIES
    });
    Ok(format!(
        "{summary}\n{}\nSearch metadata: {metadata}\nSearch snippets are not full-file reads; narrow path/query or use read_file for context.",
        results.join("\n")
    ))
}

pub struct ToolInput<'a> {
    pub input: &'a mut Input,
    pub session: &'a mut Session,
    pub footer: &'a Footer,
}

async fn run_terminal(
    root: &Path,
    args: &Value,
    mode: PermissionMode,
    on_activity: &dyn Fn(&str),
    mut interactive: Option<ToolInput<'_>>,
) -> Result<ToolOutcome> {
    if mode == PermissionMode::Off {
        bail!("[UNAVAILABLE_TOOL] Tool 'run_terminal' is unavailable.");
    }
    let command = required_str(args, "command")?;
    if command.trim().is_empty() {
        bail!("[INVALID_ARGUMENTS] command must be a non-empty string.");
    }
    if command.len() > 20_000 {
        bail!("[INVALID_ARGUMENTS] command is longer than the 20,000 character limit.");
    }
    if mode == PermissionMode::Ask {
        on_activity("Approval needed");
        let approved = match interactive.as_mut() {
            Some(io) => {
                io.session
                    .read_approval(io.input, io.footer, command)
                    .await
            }
            None => true,
        };
        if !approved {
            return Ok(ToolOutcome {
                result: Ok(
                    "Permission denied by the user. The command was not executed.".to_string(),
                ),
                aborted: false,
            });
        }
    }
    on_activity("Running Terminal");

    #[cfg(unix)]
    let mut process = {
        let mut process = tokio::process::Command::new("sh");
        process.arg("-c").arg(command);
        process
    };
    #[cfg(windows)]
    let mut process = {
        let mut process = tokio::process::Command::new("cmd");
        process.arg("/C").arg(command);
        process
    };
    process
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let child = process
        .spawn()
        .map_err(|error| anyhow!("Could not start the command: {error}"))?;

    let mut aborted = false;
    let output: Option<std::process::Output> = match interactive.as_mut() {
        Some(io) => {
            let wait = tokio::time::timeout(
                std::time::Duration::from_secs(120),
                child.wait_with_output(),
            );
            tokio::pin!(wait);
            loop {
                tokio::select! {
                    result = &mut wait => {
                        match result {
                            Ok(Ok(output)) => break Some(output),
                            Ok(Err(error)) => bail!("Could not run the command: {error}"),
                            Err(_) => {
                                return Ok(ToolOutcome {
                                    result: Ok(
                                        "Exit code: terminated (timeout)\n\
                                         Command stopped after 120 seconds."
                                            .to_string(),
                                    ),
                                    aborted: false,
                                });
                            }
                        }
                    }
                    chunk = input::read_chunk(io.input) => {
                        if let Some((value, bulk, more)) = chunk {
                            let event = io.session.feed(value, bulk, more);
                            if !more {
                                io.session.render(io.footer);
                            }
                            match event {
                                EditorEvent::Abort | EditorEvent::Cancel => {
                                    aborted = true;
                                    break None;
                                }
                                EditorEvent::Exit => {
                                    io.session.request_exit();
                                    aborted = true;
                                    break None;
                                }
                                EditorEvent::Submit(text) => {
                                    let trimmed = text.trim();
                                    if trimmed.is_empty() {
                                        continue;
                                    }
                                    if matches!(trimmed, "/exit" | "/quit") {
                                        io.session.clear_queue();
                                        io.session.request_exit();
                                        aborted = true;
                                        break None;
                                    }
                                    io.session.queue_submission(text);
                                    on_activity("Running Terminal");
                                }
                                EditorEvent::ToggleMode => {
                                    let next = io.session.toggle_steering_mode();
                                    io.session.set_prompt(ui::mode_prompt(next));
                                    io.session.render(io.footer);
                                }
                                EditorEvent::CycleThink => {
                                    let level = io.session.cycle_steering_think(None);
                                    let label = if level == "false" {
                                        "off"
                                    } else {
                                        level.as_str()
                                    };
                                    ui::print_info(&ui::warning(&format!(
                                        "· Thinking: {label} · applies to the next request."
                                    )));
                                }
                                EditorEvent::None => {}
                            }
                        }
                    }
                }
            }
        }
        None => match tokio::time::timeout(
            std::time::Duration::from_secs(120),
            child.wait_with_output(),
        )
        .await
        {
            Ok(Ok(output)) => Some(output),
            Ok(Err(error)) => bail!("Could not run the command: {error}"),
            Err(_) => {
                return Ok(ToolOutcome {
                    result: Ok(
                        "Exit code: terminated (timeout)\n\
                         Command stopped after 120 seconds."
                            .to_string(),
                    ),
                    aborted: false,
                });
            }
        },
    };
    if aborted {
        return Ok(ToolOutcome {
            result: Ok("Response stopped by the user. The command was interrupted.".to_string()),
            aborted: true,
        });
    }
    let output = output.unwrap();
    let mut combined = output.stdout;
    combined.extend_from_slice(&output.stderr);
    let mut truncated = false;
    if combined.len() > 64 * 1024 {
        combined.truncate(64 * 1024);
        truncated = true;
    }
    let text = String::from_utf8_lossy(&combined).to_string();
    let code = output
        .status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "terminated".to_string());
    let mut lines = vec![format!("Exit code: {code}")];
    if !text.is_empty() {
        lines.push(text);
    }
    if truncated {
        lines.push("Output truncated at 64 KiB; command was stopped.".to_string());
    }
    Ok(ToolOutcome {
        result: Ok(lines.join("\n")),
        aborted: false,
    })
}

#[cfg(test)]
pub async fn execute(
    root: &Path,
    terminal_mode: PermissionMode,
    tools: &[Value],
    name: &str,
    arguments: &str,
) -> Result<String> {
    let outcome = execute_with_activity(
        root,
        terminal_mode,
        tools,
        name,
        arguments,
        &|_| {},
        None,
    )
    .await?;
    outcome.result
}

/// Like [`execute`], but reports activity transitions (used by the footer to
/// show `Approval needed` while a terminal command waits for confirmation) and
/// accepts an optional [`ToolInput`] bundle so the shared editor can collect
/// approval lines, queue prompts, and abort a long terminal command.
/// `aborted` in the [`ToolOutcome`] signals that the user pressed Esc.
pub async fn execute_with_activity(
    root: &Path,
    terminal_mode: PermissionMode,
    tools: &[Value],
    name: &str,
    arguments: &str,
    on_activity: &dyn Fn(&str),
    interactive: Option<ToolInput<'_>>,
) -> Result<ToolOutcome> {
    let args = validate_arguments(name, arguments, tools)?;
    let outcome: ToolOutcome = match name {
        "read_file" => ToolOutcome { result: read_file(root, &args), aborted: false },
        "list_directory" => ToolOutcome { result: list_directory(root, &args), aborted: false },
        "edit_file" => ToolOutcome { result: edit_file(root, &args), aborted: false },
        "write_file" => ToolOutcome { result: write_file(root, &args), aborted: false },
        "delete_file" => ToolOutcome { result: delete_file(root, &args), aborted: false },
        "delete_directory" => ToolOutcome { result: delete_directory(root, &args), aborted: false },
        "search_files" => ToolOutcome { result: search_files(root, &args), aborted: false },
        "run_terminal" => {
            run_terminal(root, &args, terminal_mode, on_activity, interactive).await?
        }
        other => bail!("[UNAVAILABLE_TOOL] Tool '{other}' is unavailable."),
    };
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn workspace() -> TempDir {
        TempDir::new().expect("temp dir")
    }

    #[tokio::test]
    async fn write_read_edit_delete_round_trip() {
        let workspace = workspace();
        let root = workspace.path();
        let tools = definitions(PermissionMode::Off);
        let written = execute(
            root,
            PermissionMode::Off,
            &tools,
            "write_file",
            r#"{"path":"notes/a.txt","content":"hello world"}"#,
        )
        .await
        .unwrap();
        assert_eq!(written, "Wrote notes/a.txt.");
        let read = execute(
            root,
            PermissionMode::Off,
            &tools,
            "read_file",
            r#"{"path":"notes/a.txt"}"#,
        )
        .await
        .unwrap();
        assert_eq!(read, "hello world");
        let edited = execute(
            root,
            PermissionMode::Off,
            &tools,
            "edit_file",
            r#"{"path":"notes/a.txt","old_text":"world","new_text":"rust"}"#,
        )
        .await
        .unwrap();
        assert_eq!(edited, "Updated notes/a.txt.");
        let read = execute(
            root,
            PermissionMode::Off,
            &tools,
            "read_file",
            r#"{"path":"notes/a.txt"}"#,
        )
        .await
        .unwrap();
        assert_eq!(read, "hello rust");
        execute(
            root,
            PermissionMode::Off,
            &tools,
            "delete_file",
            r#"{"path":"notes/a.txt"}"#,
        )
        .await
        .unwrap();
        assert!(!root.join("notes/a.txt").exists());
    }

    #[tokio::test]
    async fn edit_requires_unique_old_text() {
        let workspace = workspace();
        let root = workspace.path();
        let tools = definitions(PermissionMode::Off);
        execute(
            root,
            PermissionMode::Off,
            &tools,
            "write_file",
            r#"{"path":"a.txt","content":"x x"}"#,
        )
        .await
        .unwrap();
        let error = execute(
            root,
            PermissionMode::Off,
            &tools,
            "edit_file",
            r#"{"path":"a.txt","old_text":"x","new_text":"y"}"#,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("not unique"));
    }

    #[tokio::test]
    async fn paths_outside_the_workspace_are_rejected() {
        let workspace = workspace();
        let root = workspace.path();
        let tools = definitions(PermissionMode::Off);
        let error = execute(
            root,
            PermissionMode::Off,
            &tools,
            "write_file",
            r#"{"path":"../escape.txt","content":"x"}"#,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("outside"));
    }

    #[tokio::test]
    async fn read_file_honors_offset_and_limit() {
        let workspace = workspace();
        let root = workspace.path();
        let tools = definitions(PermissionMode::Off);
        execute(
            root,
            PermissionMode::Off,
            &tools,
            "write_file",
            r#"{"path":"a.txt","content":"one\ntwo\nthree\n"}"#,
        )
        .await
        .unwrap();
        let read = execute(
            root,
            PermissionMode::Off,
            &tools,
            "read_file",
            r#"{"path":"a.txt","offset":2,"limit":1}"#,
        )
        .await
        .unwrap();
        assert!(read.starts_with("two"));
        assert!(read.contains("2 more lines"));
    }

    #[tokio::test]
    async fn list_directory_marks_kinds_and_sorts() {
        let workspace = workspace();
        let root = workspace.path();
        let tools = definitions(PermissionMode::Off);
        execute(
            root,
            PermissionMode::Off,
            &tools,
            "write_file",
            r#"{"path":"b.txt","content":"b"}"#,
        )
        .await
        .unwrap();
        fs::create_dir(root.join("A")).unwrap();
        let listing = execute(root, PermissionMode::Off, &tools, "list_directory", "{}")
            .await
            .unwrap();
        let lines: Vec<&str> = listing.lines().collect();
        assert_eq!(lines[0], "Directory: \".\"");
        assert_eq!(lines[1], "[DIR] A/");
        assert_eq!(lines[2], "[FILE] b.txt");
    }

    #[tokio::test]
    async fn search_finds_filenames_and_content() {
        let workspace = workspace();
        let root = workspace.path();
        let tools = definitions(PermissionMode::Off);
        execute(
            root,
            PermissionMode::Off,
            &tools,
            "write_file",
            r#"{"path":"src/needle.txt","content":"a NEEDLE here\n"}"#,
        )
        .await
        .unwrap();
        execute(
            root,
            PermissionMode::Off,
            &tools,
            "write_file",
            r#"{"path":"src/other.txt","content":"nothing\n"}"#,
        )
        .await
        .unwrap();
        let found = execute(
            root,
            PermissionMode::Off,
            &tools,
            "search_files",
            r#"{"query":"needle"}"#,
        )
        .await
        .unwrap();
        assert!(found.contains("[filename] \"src/needle.txt\""));
        assert!(found.contains("[content] \"src/needle.txt\":1:3"));
        assert!(found.contains("complete"));
    }

    #[tokio::test]
    async fn search_excludes_generated_directories() {
        let workspace = workspace();
        let root = workspace.path();
        let tools = definitions(PermissionMode::Off);
        execute(
            root,
            PermissionMode::Off,
            &tools,
            "write_file",
            r#"{"path":"node_modules/a.txt","content":"needle"}"#,
        )
        .await
        .unwrap();
        let found = execute(
            root,
            PermissionMode::Off,
            &tools,
            "search_files",
            r#"{"query":"needle"}"#,
        )
        .await
        .unwrap();
        assert!(found.starts_with("0 results"));
    }

    #[tokio::test]
    async fn unavailable_tool_suggests_close_name() {
        let workspace = workspace();
        let root = workspace.path();
        let tools = definitions(PermissionMode::Auto);
        let error = execute(root, PermissionMode::Auto, &tools, "terminal", "{}")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("Did you mean 'run_terminal'"));
    }

    #[tokio::test]
    async fn missing_required_argument_is_reported() {
        let workspace = workspace();
        let root = workspace.path();
        let tools = definitions(PermissionMode::Off);
        let error = execute(root, PermissionMode::Off, &tools, "read_file", "{}")
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Missing required argument: path")
        );
    }

    #[tokio::test]
    async fn unknown_arguments_are_rejected() {
        let workspace = workspace();
        let root = workspace.path();
        let tools = definitions(PermissionMode::Off);
        let error = execute(
            root,
            PermissionMode::Off,
            &tools,
            "read_file",
            r#"{"path":"a","extra":true}"#,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("Unknown argument"));
    }

    #[tokio::test]
    async fn terminal_is_not_advertised_when_off() {
        let tools = definitions(PermissionMode::Off);
        assert!(!tool_names(&tools).contains(&"run_terminal".to_string()));
        let tools = definitions(PermissionMode::Auto);
        assert!(tool_names(&tools).contains(&"run_terminal".to_string()));
    }

    #[tokio::test]
    async fn terminal_runs_in_the_workspace() {
        let workspace = workspace();
        let root = workspace.path();
        let tools = definitions(PermissionMode::Auto);
        let result = execute(
            root,
            PermissionMode::Auto,
            &tools,
            "run_terminal",
            r#"{"command":"pwd"}"#,
        )
        .await
        .unwrap();
        assert!(result.starts_with("Exit code: 0"));
        assert!(result.contains(&root.to_string_lossy().to_string()));
    }
}
