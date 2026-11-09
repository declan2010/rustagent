use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermissionMode {
    Auto,
    Ask,
    Off,
}

impl PermissionMode {
    pub fn parse(value: Option<&str>, name: &str, fallback: &str) -> Result<Self> {
        let normalized = value.unwrap_or(fallback).trim();
        match normalized {
            "auto" => Ok(Self::Auto),
            "ask" => Ok(Self::Ask),
            "off" => Ok(Self::Off),
            _ => bail!("{name} must be lowercase: auto, ask, or off."),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Ask => "ask",
            Self::Off => "off",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub endpoint: String,
    pub api_key: Option<String>,
    pub model: String,
    pub context_window: u64,
    pub timeout_ms: u64,
    pub max_rounds: usize,
    pub ollama_think: Option<String>,
    pub input_modalities: Vec<String>,
    pub terminal_mode: PermissionMode,
    pub skills_mode: PermissionMode,
    pub mcp_mode: PermissionMode,
    /// The `.env` files that contributed values, joined for the banner's
    /// Config row (`.env + ~/.config/rustagent/.env`).
    pub config_files: String,
    pub root: PathBuf,
    pub workspace_name: String,
}

fn parse_env_file(contents: &str) -> HashMap<String, String> {
    let mut values = HashMap::new();
    for (index, line) in contents.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((name, raw)) = trimmed.split_once('=') else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        let mut value = raw.trim().to_string();
        if (value.starts_with('"') && value.ends_with('"') && value.len() >= 2)
            || (value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2)
        {
            value = value[1..value.len() - 1].to_string();
        } else if let Some((before, _)) = value.split_once(" #") {
            value = before.trim().to_string();
        }
        values.entry(name.to_string()).or_insert(value);
        let _ = index;
    }
    values
}

/// Process environment wins over `.env`, matching MinAgent's `loadEnvFile` behavior.
fn load_env(path: &Path, env: &mut HashMap<String, String>) -> usize {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return 0;
    };
    let mut inserted = 0;
    for (key, value) in parse_env_file(&contents) {
        // First value loaded wins; a file only counts as a source when it
        // actually contributed a value.
        if !env.contains_key(&key) {
            env.insert(key, value);
            inserted += 1;
        }
    }
    inserted
}

/// A display path with the home directory abbreviated to `~` so the banner's
/// Config row stays short (`~/.config/rustagent/.env`).
fn display_path(path: &Path) -> String {
    if let Some(home) = std::env::var_os("HOME") {
        if let Ok(rest) = path.strip_prefix(Path::new(&home)) {
            return format!("~/{}", rest.display());
        }
    }
    path.display().to_string()
}

fn positive_integer(value: Option<&String>, name: &str, fallback: u64) -> Result<u64> {
    match value.map(|value| value.trim()) {
        None | Some("") => Ok(fallback),
        Some(text) => match text.parse::<u64>() {
            Ok(parsed) if parsed > 0 => Ok(parsed),
            _ => bail!("{name} must be a positive integer."),
        },
    }
}

fn non_negative_integer(value: Option<&String>, name: &str, fallback: u64) -> Result<u64> {
    match value.map(|value| value.trim()) {
        None | Some("") => Ok(fallback),
        Some(text) => match text.parse::<u64>() {
            Ok(parsed) => Ok(parsed),
            _ => bail!("{name} must be a non-negative integer (0 disables timeouts)."),
        },
    }
}

fn parse_ollama_think(value: Option<&String>) -> Result<Option<String>> {
    let Some(raw) = value
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    let normalized = raw.to_lowercase();
    let mapped = match normalized.as_str() {
        "off" | "no" | "none" => "false",
        "on" | "yes" => "true",
        other => other,
    };
    let allowed = ["false", "true", "low", "medium", "high", "xhigh", "max"];
    if !allowed.contains(&mapped) {
        bail!(
            "OPENAI_OLLAMA_THINK must be one of: {} (aliases: off, on, no, yes).",
            allowed.join(", ")
        );
    }
    Ok(Some(mapped.to_string()))
}

fn parse_input_modalities(value: Option<&String>) -> Result<Vec<String>> {
    let raw = value
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .unwrap_or("text,image");
    let mut modalities: Vec<String> = raw
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|item| !item.is_empty())
        .map(|item| item.to_lowercase())
        .collect();
    modalities.dedup();
    if !modalities.iter().any(|item| item == "text")
        || modalities
            .iter()
            .any(|item| item != "text" && item != "image")
    {
        bail!("OPENAI_INPUT must include text and only supports the values text,image.");
    }
    Ok(modalities)
}

fn make_endpoint(base_url: &str) -> Result<String> {
    let trimmed = base_url.trim().trim_end_matches('/');
    if !(trimmed.starts_with("http://") || trimmed.starts_with("https://")) {
        bail!("OPENAI_BASE_URL must be a valid HTTP or HTTPS URL.");
    }
    if trimmed.ends_with("/chat/completions") {
        Ok(trimmed.to_string())
    } else {
        Ok(format!("{trimmed}/chat/completions"))
    }
}

/// Path to the user-level RustAgent config file (`~/.config/rustagent/.env`).
pub fn global_config_path() -> Option<PathBuf> {
    let config_home = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(config_home.join("rustagent").join(".env"))
}

/// Persists `OPENAI_MODEL=<model>` in the user config file so the selected model
/// becomes the default for future sessions, matching MinAgent's
/// `persistModelDefault`. Replaces an existing line or appends one.
pub fn persist_model_default(model: &str) -> Result<PathBuf> {
    let path = global_config_path()
        .ok_or_else(|| anyhow!("Could not determine the user config directory."))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| anyhow!("Could not create {}: {error}", parent.display()))?;
    }
    write_model_default(&path, model)?;
    Ok(path)
}

/// Rewrites `path` so that the `OPENAI_MODEL=` line reflects `model`, preserving
/// every other line. Creates the file if it does not exist.
fn write_model_default(path: &Path, model: &str) -> Result<()> {
    let contents = std::fs::read_to_string(path).unwrap_or_default();
    let target = format!("OPENAI_MODEL={model}");
    let mut output = String::new();
    let mut replaced = false;
    for line in contents.lines() {
        if line.starts_with("OPENAI_MODEL=") {
            output.push_str(&target);
            output.push('\n');
            replaced = true;
        } else {
            output.push_str(line);
            output.push('\n');
        }
    }
    if !replaced {
        if !contents.is_empty() && !contents.ends_with('\n') {
            output.push('\n');
        }
        output.push_str(&target);
        output.push('\n');
    }
    std::fs::write(path, &output)
        .map_err(|error| anyhow!("Could not write {}: {error}", path.display()))?;
    // Keep the config readable only by the owner, like the initial file.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

pub fn load() -> Result<Config> {
    let mut env: HashMap<String, String> = std::env::vars().collect();
    // Precedence (highest first): process environment, project `.env`, user config,
    // `.env` next to the binary. First value loaded wins.
    let mut config_files: Vec<String> = Vec::new();
    if load_env(Path::new(".env"), &mut env) > 0 {
        config_files.push(".env".to_string());
    }
    if let Some(config_home) = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
    {
        let path = config_home.join("rustagent").join(".env");
        if load_env(&path, &mut env) > 0 {
            config_files.push(display_path(&path));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(directory) = exe.parent() {
            let path = directory.join(".env");
            if load_env(&path, &mut env) > 0 {
                config_files.push(display_path(&path));
            }
            if let Some(parent) = directory.parent() {
                let path = parent.join(".env");
                if load_env(&path, &mut env) > 0 {
                    config_files.push(display_path(&path));
                }
            }
        }
    }

    let model = env
        .get("OPENAI_MODEL")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let Some(model) = model else {
        bail!(
            "Set OPENAI_MODEL to the model identifier available on your endpoint \
             (environment, project .env, or ~/.config/rustagent/.env)."
        );
    };

    let root = std::fs::canonicalize(std::env::current_dir()?)
        .unwrap_or_else(|_| std::env::current_dir().unwrap());
    let workspace_name = root
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "workspace".to_string());

    let endpoint = make_endpoint(
        env.get("OPENAI_BASE_URL")
            .map(String::as_str)
            .unwrap_or("https://api.openai.com/v1"),
    )?;

    Ok(Config {
        endpoint,
        api_key: env
            .get("OPENAI_API_KEY")
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
        model,
        context_window: positive_integer(
            env.get("OPENAI_CONTEXT_WINDOW"),
            "OPENAI_CONTEXT_WINDOW",
            262_144,
        )?,
        timeout_ms: non_negative_integer(env.get("OPENAI_TIMEOUT_MS"), "OPENAI_TIMEOUT_MS", 0)?,
        max_rounds: positive_integer(env.get("OPENAI_MAX_ROUNDS"), "OPENAI_MAX_ROUNDS", 32)?
            as usize,
        ollama_think: parse_ollama_think(env.get("OPENAI_OLLAMA_THINK"))?,
        input_modalities: parse_input_modalities(env.get("OPENAI_INPUT"))?,
        terminal_mode: PermissionMode::parse(
            env.get("TERMINAL_MODE").map(String::as_str),
            "TERMINAL_MODE",
            "ask",
        )?,
        skills_mode: PermissionMode::parse(
            env.get("SKILLS_MODE").map(String::as_str),
            "SKILLS_MODE",
            "off",
        )?,
        mcp_mode: PermissionMode::parse(
            env.get("MCP_MODE").map(String::as_str),
            "MCP_MODE",
            "off",
        )?,
        config_files: config_files.join(" + "),
        root,
        workspace_name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_normalization_appends_chat_completions() {
        assert_eq!(
            make_endpoint("http://127.0.0.1:11434/v1").unwrap(),
            "http://127.0.0.1:11434/v1/chat/completions"
        );
        assert_eq!(
            make_endpoint("https://api.example.com/v1/").unwrap(),
            "https://api.example.com/v1/chat/completions"
        );
        assert_eq!(
            make_endpoint("https://api.example.com/v1/chat/completions").unwrap(),
            "https://api.example.com/v1/chat/completions"
        );
        assert!(make_endpoint("ftp://example.com").is_err());
    }

    #[test]
    fn load_env_counts_only_the_values_a_file_actually_contributed() {
        use tempfile::TempDir;
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join(".env");
        std::fs::write(&path, "A=1\nB=2\n").expect("seed");
        let mut env: std::collections::HashMap<String, String> =
            [("A".to_string(), "from process".to_string())]
                .into_iter()
                .collect();
        // A is already taken by the process environment; only B counts.
        assert_eq!(load_env(&path, &mut env), 1);
        assert_eq!(env.get("A").unwrap(), "from process");
        assert_eq!(env.get("B").unwrap(), "2");
        // A second pass contributes nothing: first value loaded wins.
        assert_eq!(load_env(&path, &mut env), 0);
        // A missing file is not a source.
        assert_eq!(load_env(&dir.path().join("nope"), &mut env), 0);
    }

    #[test]
    fn display_path_abbreviates_the_home_directory() {
        let home = std::path::PathBuf::from(
            std::env::var("HOME").unwrap_or_else(|_| "/home/tester".to_string()),
        );
        let path = home.join(".config").join("rustagent").join(".env");
        assert_eq!(display_path(&path), "~/.config/rustagent/.env");
        assert_eq!(
            display_path(std::path::Path::new("/opt/project/.env")),
            "/opt/project/.env"
        );
    }

    #[test]
    fn max_rounds_defaults_to_32_and_requires_a_positive_integer() {
        // The default matches MinAgent's MAX_TOOL_ROUNDS.
        assert_eq!(positive_integer(None, "OPENAI_MAX_ROUNDS", 32).unwrap(), 32);
        assert_eq!(
            positive_integer(Some(&"8".to_string()), "OPENAI_MAX_ROUNDS", 32).unwrap(),
            8
        );
        // Zero, negatives, and garbage are rejected so a turn can never
        // start with an empty round budget.
        assert!(positive_integer(Some(&"0".to_string()), "OPENAI_MAX_ROUNDS", 32).is_err());
        assert!(positive_integer(Some(&"-4".to_string()), "OPENAI_MAX_ROUNDS", 32).is_err());
        assert!(positive_integer(Some(&"many".to_string()), "OPENAI_MAX_ROUNDS", 32).is_err());
        // Empty falls back to the default like every other setting.
        assert_eq!(
            positive_integer(Some(&"".to_string()), "OPENAI_MAX_ROUNDS", 32).unwrap(),
            32
        );
    }

    #[test]
    fn env_file_parsing_handles_quotes_comments_and_precedence() {
        let parsed = parse_env_file("A=1\nB=\"two words\"\nC=three # comment\n# skip\nD=\n");
        assert_eq!(parsed.get("A").unwrap(), "1");
        assert_eq!(parsed.get("B").unwrap(), "two words");
        assert_eq!(parsed.get("C").unwrap(), "three");
        assert_eq!(parsed.get("D").unwrap(), "");
    }

    #[test]
    fn persist_model_default_replaces_or_appends() {
        use tempfile::TempDir;
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join(".env");

        // No file yet: the line is appended and the file created.
        write_model_default(&path, "qwen3:0.6b").expect("write new");
        let written = std::fs::read_to_string(&path).expect("read");
        assert!(written.contains("OPENAI_MODEL=qwen3:0.6b\n"));

        // Existing file with a different model: the line is replaced, others preserved.
        std::fs::write(
            &path,
            "OPENAI_BASE_URL=http://x/v1\nOPENAI_MODEL=old:1b\nOPENAI_API_KEY=k\n",
        )
        .expect("seed");
        write_model_default(&path, "qwen3:0.6b").expect("replace");
        let written = std::fs::read_to_string(&path).expect("read");
        let model_lines: Vec<&str> = written
            .lines()
            .filter(|line| line.starts_with("OPENAI_MODEL="))
            .collect();
        assert_eq!(model_lines, ["OPENAI_MODEL=qwen3:0.6b"]);
        assert!(written.contains("OPENAI_BASE_URL=http://x/v1"));
        assert!(written.contains("OPENAI_API_KEY=k"));
    }

    #[test]
    fn think_aliases_are_normalized() {
        let value = |text: &str| Some(text.to_string());
        assert_eq!(
            parse_ollama_think(value("off").as_ref()).unwrap().unwrap(),
            "false"
        );
        assert_eq!(
            parse_ollama_think(value("ON").as_ref()).unwrap().unwrap(),
            "true"
        );
        assert_eq!(
            parse_ollama_think(value("low").as_ref()).unwrap().unwrap(),
            "low"
        );
        assert!(parse_ollama_think(value("extreme").as_ref()).is_err());
    }

    #[test]
    fn permission_modes_validate() {
        assert_eq!(
            PermissionMode::parse(Some("auto"), "TERMINAL_MODE", "ask").unwrap(),
            PermissionMode::Auto
        );
        assert!(PermissionMode::parse(Some("AUTO"), "TERMINAL_MODE", "ask").is_err());
    }

    #[test]
    fn input_modalities_require_text() {
        assert_eq!(
            parse_input_modalities(Some(&"text,image".to_string())).unwrap(),
            vec!["text", "image"]
        );
        assert!(parse_input_modalities(Some(&"image".to_string())).is_err());
        assert!(parse_input_modalities(Some(&"text,audio".to_string())).is_err());
    }
}
