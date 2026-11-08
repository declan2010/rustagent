//! Local skill discovery and the `load_skill` tool, matching MinAgent's
//! `src/skills.mjs`: SKILL.md files with YAML frontmatter in fixed roots,
//! bounded context injection, and hardened reads (no symlinks, hard links,
//! escapes, or binary payloads).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use serde_json::{Value, json};
use std::sync::LazyLock;

use regex::Regex;

pub const MAX_SKILL_BYTES: usize = 64 * 1024;
pub const MAX_RESOURCE_BYTES: usize = 32 * 1024;
pub const MAX_SKILLS: usize = 24;
const MAX_SKILL_CONTEXT_CHARS: usize = 8 * 1024;
const MAX_SKILL_DESCRIPTION_CHARS: usize = 160;

pub const SKILLS_GUIDANCE: &str = "Available skills; load relevant instructions as needed:";

/// One discovered skill: instructions plus the directory holding its bundled
/// resources.
pub struct Skill {
    pub name: String,
    pub description: String,
    pub instructions: String,
    pub directory: PathBuf,
}

/// Skill root directories matching MinAgent: application root first, then the
/// workspace, each in its `skills/` and `.agents/skills/` form (deduplicated).
pub fn skill_roots(workspace: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(app) = exe.parent() {
            roots.push(app.join("skills"));
            roots.push(app.join(".agents").join("skills"));
        }
    }
    roots.push(workspace.join("skills"));
    roots.push(workspace.join(".agents").join("skills"));
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for root in roots {
        if seen.insert(root.clone()) {
            unique.push(root);
        }
    }
    unique
}

/// Reads a skill file with the same hardening as MinAgent's `readSkillText`:
/// the target and its root must stay regular, unescaped entries of stable
/// identity across the read, and the payload must be UTF-8 without NUL bytes.
fn read_skill_text(target: &Path, max_bytes: usize, root_directory: &Path) -> Result<String> {
    let root_before =
        std::fs::symlink_metadata(root_directory).map_err(|error| anyhow::anyhow!("{error}"))?;
    if root_before.is_symlink() || !root_before.is_dir() {
        bail!("Skill directory is no longer a regular directory.");
    }
    let before = std::fs::symlink_metadata(target).map_err(|error| anyhow::anyhow!("{error}"))?;
    if before.is_symlink() || !before.is_file() || linked_file(&before) {
        bail!("Skill text must be a regular, unlinked file.");
    }
    if before.len() > max_bytes as u64 {
        bail!("Skill text exceeds {max_bytes} bytes.");
    }

    let root = root_directory.canonicalize()?;
    let resolved = target.canonicalize()?;
    let relative = resolved
        .strip_prefix(&root)
        .map_err(|_| anyhow::anyhow!("Skill file resolved outside its directory."))?;
    if relative.as_os_str().is_empty() {
        bail!("Skill file resolved outside its directory.");
    }

    let buffer = std::fs::read(target)?;
    if buffer.len() > max_bytes {
        bail!("Skill text exceeds {max_bytes} bytes.");
    }
    let Ok(after) = std::fs::symlink_metadata(target) else {
        bail!("Skill file changed while it was being read.");
    };
    let Ok(root_after) = std::fs::symlink_metadata(root_directory) else {
        bail!("Skill directory changed while it was being read.");
    };
    if root_after.is_symlink()
        || !root_after.is_dir()
        || identity_mismatch(&root_before, &root_after)
    {
        bail!("Skill directory changed while it was being read.");
    }
    if after.is_symlink()
        || !after.is_file()
        || after.len() != before.len()
        || modified_since(&before, &after)
        || identity_mismatch(&before, &after)
    {
        bail!("Skill file changed while it was being read.");
    }
    if buffer.contains(&0) {
        bail!("Skill text cannot contain binary data.");
    }
    String::from_utf8(buffer).map_err(|_| anyhow::anyhow!("Skill text is not valid UTF-8."))
}

/// Hard links make a file's mutations observable elsewhere; skill text must
/// have an exclusive link.
#[cfg(unix)]
fn linked_file(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    metadata.nlink() > 1
}

#[cfg(windows)]
fn linked_file(_metadata: &std::fs::Metadata) -> bool {
    false
}

#[cfg(unix)]
fn identity_mismatch(before: &std::fs::Metadata, after: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    before.dev() != after.dev() || before.ino() != after.ino()
}

#[cfg(windows)]
fn identity_mismatch(_before: &std::fs::Metadata, _after: &std::fs::Metadata) -> bool {
    false
}

#[cfg(unix)]
fn modified_since(before: &std::fs::Metadata, after: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    before.mtime() != after.mtime() || before.mtime_nsec() != after.mtime_nsec()
}

#[cfg(windows)]
fn modified_since(_before: &std::fs::Metadata, _after: &std::fs::Metadata) -> bool {
    false
}

/// Discovers SKILL.md files under each root, returning the first
/// [`MAX_SKILLS`] valid ones plus human-readable warnings for skipped paths.
pub fn discover(skill_roots: &[PathBuf]) -> (Vec<Skill>, Vec<String>) {
    let mut skills = Vec::new();
    let mut warnings = Vec::new();
    let mut names: HashSet<String> = HashSet::new();
    let mut reached_skill_limit = false;

    for root in skill_roots {
        if skills.len() >= MAX_SKILLS {
            reached_skill_limit = true;
            break;
        }
        let entries = match std::fs::read_dir(root) {
            Ok(contents) => contents,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                warnings.push(format!(
                    "Could not inspect skills directory {}: {error}",
                    root.display()
                ));
                continue;
            }
        };
        let mut entries: Vec<std::fs::DirEntry> = entries.filter_map(|entry| entry.ok()).collect();
        entries.sort_by(|left, right| left.file_name().cmp(&right.file_name()));

        for entry in entries {
            if skills.len() >= MAX_SKILLS {
                reached_skill_limit = true;
                break;
            }
            let is_directory = std::fs::symlink_metadata(entry.path())
                .map(|metadata| metadata.is_dir())
                .unwrap_or(false);
            if !is_directory {
                continue;
            }
            let directory_name = entry.file_name();
            let directory_name = directory_name.to_string_lossy().to_string();
            let skill_directory = entry.path();
            let skill_file = skill_directory.join("SKILL.md");
            match read_skill_text(&skill_file, MAX_SKILL_BYTES, &skill_directory)
                .and_then(|source| parse_skill_file(&source, &skill_directory, &directory_name))
            {
                Ok(skill) => {
                    if names.contains(&skill.name) {
                        warnings.push(format!(
                            "Skipping duplicate skill name \"{}\" at {}",
                            skill.name,
                            skill_file.display()
                        ));
                        continue;
                    }
                    names.insert(skill.name.clone());
                    skills.push(skill);
                }
                Err(error) if not_found(&error) => continue,
                Err(error) => warnings.push(format!(
                    "Skipping invalid skill at {}: {}",
                    skill_file.display(),
                    error
                )),
            }
        }
    }
    if reached_skill_limit {
        warnings.push(format!(
            "Only the first {MAX_SKILLS} skills are loaded to keep the model context bounded."
        ));
    }
    (skills, warnings)
}

fn not_found(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
    })
}

/// Parses one SKILL.md: `---` frontmatter with name/description, then a
/// non-empty instruction body. Name must match its directory.
fn parse_skill_file(source: &str, skill_directory: &Path, directory_name: &str) -> Result<Skill> {
    let normalized = source.strip_prefix('\u{FEFF}').unwrap_or(source);
    let (frontmatter, end) = frontmatter_block(normalized).ok_or_else(|| {
        anyhow::anyhow!("SKILL.md must start with YAML frontmatter delimited by --- lines.")
    })?;
    let metadata = parse_frontmatter(&frontmatter);
    let name = metadata.name.trim().to_string();
    let description = metadata.description.trim().to_string();
    if !is_valid_skill_name(&name) {
        bail!(
            "frontmatter name must use lowercase letters, digits, and hyphens (up to 64 characters)."
        );
    }
    if name != directory_name {
        bail!("frontmatter name \"{name}\" must match its directory \"{directory_name}\".");
    }
    let description_chars = description.chars().count();
    if description.is_empty() || description_chars > 1024 {
        bail!("frontmatter description must contain 1 to 1,024 characters.");
    }
    let instructions = normalized[end..].trim().to_string();
    if instructions.is_empty() {
        bail!("SKILL.md must include instructions after the frontmatter.");
    }
    Ok(Skill {
        name,
        description,
        instructions,
        directory: skill_directory.to_path_buf(),
    })
}

fn is_valid_skill_name(name: &str) -> bool {
    static NAME: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$").expect("skill name regex")
    });
    !name.contains("--") && NAME.is_match(name)
}

/// Splits a leading `--- … ---` frontmatter block, returning its body and the
/// offset just past the closing delimiter (MimAgent's frontmatter regex).
fn frontmatter_block(source: &str) -> Option<(String, usize)> {
    // The first line must be exactly "---".
    let first_end = source.find('\n')?;
    let first_line = source[..first_end]
        .strip_suffix('\r')
        .unwrap_or(&source[..first_end]);
    if first_line != "---" {
        return None;
    }
    let mut offset = first_end + 1;
    if offset > source.len() {
        return None;
    }
    let mut body_lines: Vec<&str> = Vec::new();
    loop {
        if offset >= source.len() {
            return None;
        }
        let line_end = source[offset..]
            .find('\n')
            .map(|at| offset + at)
            .unwrap_or(source.len());
        let line = source[offset..line_end]
            .strip_suffix('\r')
            .unwrap_or(&source[offset..line_end]);
        let past = (line_end + 1).min(source.len());
        if line == "---" {
            return Some((body_lines.join("\n"), past));
        }
        body_lines.push(line);
        offset = past;
    }
}

struct Frontmatter {
    name: String,
    description: String,
}

/// Minimal frontmatter parser mirroring MinAgent's `parseFrontmatter`: plain
/// `key: value` lines, folded `>`/`|` descriptions, quote stripping, and
/// comment removal.
fn parse_frontmatter(source: &str) -> Frontmatter {
    let mut name = String::new();
    let mut description = String::new();
    static KEY: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"^([A-Za-z][A-Za-z0-9_-]*):\s*(.*)$").expect("frontmatter key regex")
    });
    let lines: Vec<&str> = source.split('\n').collect();
    let mut index = 0usize;
    while index < lines.len() {
        let line = lines[index];
        let Some(capture) = KEY.captures(line) else {
            index += 1;
            continue;
        };
        let key = capture.get(1).expect("key group").as_str();
        let mut value = capture.get(2).expect("value group").as_str().to_string();
        if key == "description" && [">", ">-", "|", "|-"].contains(&value.trim()) {
            let style = if value.trim().starts_with('>') {
                " "
            } else {
                "\n"
            };
            let mut folded: Vec<String> = Vec::new();
            while index + 1 < lines.len() {
                let next = lines[index + 1];
                // Mirrors MinAgent's `/^\s+\S/`: a continuation line starts
                // with whitespace and carries content.
                if (next.starts_with(' ') || next.starts_with('\t')) && !next.trim().is_empty() {
                    folded.push(next.trim().to_string());
                    index += 1;
                } else {
                    break;
                }
            }
            value = folded.join(style);
        } else {
            value = strip_yaml_comment(value.trim().to_string());
            if (value.starts_with('"') && value.ends_with('"') && value.len() >= 2)
                || (value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2)
            {
                value = value[1..value.len() - 1].to_string();
            }
        }
        if key == "name" {
            name = value;
        } else if key == "description" {
            description = value;
        }
        index += 1;
    }
    Frontmatter { name, description }
}

fn strip_yaml_comment(value: String) -> String {
    let characters: Vec<char> = value.chars().collect();
    let mut quote = '\0';
    let bytes_offset = {
        let mut end: Option<usize> = None;
        for (offset, character) in characters.iter().enumerate() {
            if (*character == '"' || *character == '\'') && (quote == '\0' || quote == *character) {
                quote = if quote == '\0' { *character } else { '\0' };
            } else if *character == '#'
                && quote == '\0'
                && (offset == 0 || characters[offset - 1].is_whitespace())
            {
                end = Some(offset);
                break;
            }
        }
        end
    };
    match bytes_offset {
        Some(offset) => characters[..offset]
            .iter()
            .collect::<String>()
            .trim_end()
            .to_string(),
        None => value,
    }
}

/// Renders the skills section injected into the system prompt: the guidance
/// line followed by `- name: description` entries, truncated at the character
/// budget with an omission note.
pub fn format_context(skills: &[Skill]) -> String {
    if skills.is_empty() {
        return String::new();
    }
    let mut parts: Vec<String> = vec![SKILLS_GUIDANCE.to_string()];
    let mut current = SKILLS_GUIDANCE.to_string();
    for (index, skill) in skills.iter().enumerate() {
        let clean: String = skill
            .description
            .chars()
            .map(|character| {
                if character.is_control() || character == '\u{7f}' {
                    ' '
                } else {
                    character
                }
            })
            .collect();
        let character_count = clean.chars().count();
        let description: String = if character_count > MAX_SKILL_DESCRIPTION_CHARS {
            let mut cut: String = clean
                .chars()
                .take(MAX_SKILL_DESCRIPTION_CHARS - 1)
                .collect();
            cut.push('…');
            cut
        } else {
            clean
        };
        let entry = format!("- {}: {}", json!(skill.name), json!(description));
        let next = format!("{current}\n{entry}");
        if next.chars().count() > MAX_SKILL_CONTEXT_CHARS {
            let omitted = skills.len() - index;
            let note = format!("[{omitted} skill descriptions omitted by the context size limit.]");
            if current.chars().count() + 1 + note.chars().count() <= MAX_SKILL_CONTEXT_CHARS {
                parts.push(note);
            }
            break;
        }
        parts.push(entry.clone());
        current = next;
    }
    parts.join("\n")
}

/// Appends the `load_skill` tool definition once skills are available.
pub fn push_tool(tools: &mut Vec<Value>) {
    tools.push(json!({
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "Load discovered skill instructions or a bundled text resource.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Exact discovered skill name" },
                    "path": { "type": "string", "description": "Skill-relative resource path; omit for instructions" }
                },
                "required": ["name"]
            }
        }
    }));
}

/// Executes `load_skill` on validated arguments: instructions by default, or a
/// hardened read of one skill-relative resource. Returns the tool text fed
/// back to the model.
pub fn execute_load_skill(args: &Value, skills: &[Skill]) -> Result<String> {
    let Some(name) = args.get("name").and_then(Value::as_str) else {
        bail!("[INVALID_ARGUMENTS] name must be a string.");
    };
    let Some(skill) = skills.iter().find(|entry| entry.name == name) else {
        bail!("Skill not found: {name}");
    };
    let Some(path) = args.get("path") else {
        return Ok(format!(
            "Skill instructions for {}:\n\n{}",
            skill.name, skill.instructions
        ));
    };
    let path = path.as_str().ok_or_else(|| {
        anyhow::anyhow!("[INVALID_ARGUMENTS] A non-empty skill resource path is required.")
    })?;
    if path.trim().is_empty() {
        bail!("[INVALID_ARGUMENTS] A non-empty skill resource path is required.");
    }
    if Path::new(path).is_absolute() {
        bail!("Skill resource paths must be relative to the skill directory.");
    }
    // Reject anything but plain relative segments up front.
    if !Path::new(path)
        .components()
        .all(|segment| matches!(segment, std::path::Component::Normal(_)))
    {
        bail!("Skill resource path is outside the skill directory.");
    }
    // Walk every skill-relative segment: parent directories and links are
    // rejected before any read, mirroring MinAgent's segment walk.
    for offset in 1..=Path::new(path).components().count() {
        let probe: PathBuf = skill.directory.join(
            Path::new(path)
                .components()
                .map(|segment| segment.as_os_str())
                .take(offset)
                .collect::<PathBuf>(),
        );
        let entry =
            std::fs::symlink_metadata(&probe).map_err(|error| anyhow::anyhow!("{error}"))?;
        if entry.is_symlink() || (entry.is_file() && linked_file(&entry)) {
            bail!("Symbolic links and hard links are blocked in skill resources.");
        }
    }
    let root = skill.directory.canonicalize()?;
    let joined = skill.directory.join(path);
    let target = joined
        .canonicalize()
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    if target
        .strip_prefix(&root)
        .map(|relative| relative.as_os_str().is_empty())
        .unwrap_or(true)
    {
        bail!("Skill resource path is outside the skill directory.");
    }
    let content = read_skill_text(&target, MAX_RESOURCE_BYTES, &skill.directory)?;
    Ok(format!(
        "Skill resource {}/{}:\n\n{}",
        skill.name, path, content
    ))
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Writes `skills/<name>/SKILL.md` with the given frontmatter and body.
    fn write_skill(root: &TempDir, name: &str, frontmatter: &str, body: &str) {
        let directory = root.path().join("skills").join(name);
        fs::create_dir_all(&directory).expect("skill dir");
        fs::write(directory.join("SKILL.md"), format!("{frontmatter}\n{body}"))
            .expect("write SKILL.md");
    }

    /// Frontmatter whose name matches the target directory.
    fn plain(name: &str) -> String {
        format!("---\nname: {name}\ndescription: An example skill\n---")
    }

    #[test]
    fn discovers_parsed_skills_from_sorted_roots() {
        let workspace = TempDir::new().expect("temp dir");
        write_skill(&workspace, "beta", &plain("beta"), "Beta instructions.");
        write_skill(&workspace, "alpha", &plain("alpha"), "Alpha body.");
        let (skills, warnings) = discover(&skill_roots(workspace.path()));
        assert!(warnings.is_empty(), "no warnings: {warnings:?}");
        assert_eq!(skills.len(), 2);
        assert_eq!(skills[0].name, "alpha", "entries are sorted by directory");
        assert_eq!(skills[1].name, "beta");
        assert_eq!(skills[0].instructions, "Alpha body.");
        assert_eq!(
            skills[0].directory,
            workspace.path().join("skills").join("alpha")
        );
    }

    #[test]
    fn skips_invalid_and_unnamed_entries_with_warnings() {
        let workspace = TempDir::new().expect("temp dir");
        // A loose file (not a directory) is invisible to discovery.
        fs::create_dir_all(workspace.path().join("skills")).expect("skills dir");
        fs::write(
            workspace.path().join("skills").join("loose.md"),
            "---\nname: loose\ndescription: not a directory\n---\nbody",
        )
        .expect("loose file");
        write_skill(
            &workspace,
            "mismatched",
            "---\nname: other\ndescription: Nope\n---",
            "x",
        );
        write_skill(&workspace, "nofrontmatter", "plain text\n", "");
        write_skill(&workspace, "valid", &plain("valid"), "ok");
        let (skills, warnings) = discover(&skill_roots(workspace.path()));
        assert_eq!(skills.len(), 1, "only the valid skill survives");
        assert_eq!(skills[0].name, "valid");
        assert_eq!(warnings.len(), 2, "warnings: {warnings:?}");
        assert!(
            warnings
                .iter()
                .all(|warning| warning.starts_with("Skipping invalid skill at ")),
            "warnings: {warnings:?}"
        );
    }

    #[test]
    fn name_must_match_directory_and_charset() {
        let workspace = TempDir::new().expect("temp dir");
        write_skill(
            &workspace,
            "bad--name",
            "---\nname: bad--name\ndescription: x\n---\nbody",
            "",
        );
        write_skill(
            &workspace,
            "Upper",
            "---\nname: Upper\ndescription: x\n---\nbody",
            "",
        );
        let (skills, warnings) = discover(&[workspace.path().join("skills")]);
        assert!(skills.is_empty());
        assert_eq!(warnings.len(), 2);
    }

    #[test]
    fn enforces_max_count_and_oversized_files() {
        let workspace = TempDir::new().expect("temp dir");
        let roots = vec![workspace.path().join("skills")];
        // 30 valid skills but only the first 24 (sorted) are kept.
        for index in 0..30 {
            let name = format!("skill-{index:02}");
            write_skill(&workspace, &name, &plain(&name), "body");
        }
        // An oversized SKILL.md sorts first and is skipped by size.
        write_skill(
            &workspace,
            "aaa-oversized",
            "---\nname: aaa-oversized\ndescription: d\n---",
            "",
        );
        fs::write(
            workspace
                .path()
                .join("skills")
                .join("aaa-oversized")
                .join("SKILL.md"),
            "---\nname: aaa-oversized\ndescription: d\n---\n".repeat(64 * 1024),
        )
        .expect("oversized file");
        let (skills, warnings) = discover(&roots);
        assert_eq!(skills.len(), MAX_SKILLS);
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("first 24 skills")),
            "limit warning present: {warnings:?}"
        );
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("Skill text exceeds 65536 bytes.")),
            "oversized warning present: {warnings:?}"
        );
        assert!(!skills.iter().any(|skill| skill.name == "aaa-oversized"));
        assert!(skills.iter().all(|skill| skill.name != "aaa-oversized"));
    }

    #[test]
    fn folded_and_quoted_descriptions_parse() {
        let workspace = TempDir::new().expect("temp dir");
        write_skill(
            &workspace,
            "folded",
            "---\nname: folded\ndescription: >\n  long folded line one\n  and two\n---",
            "body",
        );
        write_skill(
            &workspace,
            "quoted",
            "---\nname: quoted\ndescription: \"with # hash\"\n---",
            "body",
        );
        let (skills, warnings) = discover(&[workspace.path().join("skills")]);
        assert!(warnings.is_empty(), "{warnings:?}");
        let folded = skills
            .iter()
            .find(|skill| skill.name == "folded")
            .expect("folded");
        assert_eq!(folded.description, "long folded line one and two");
        let quoted = skills
            .iter()
            .find(|skill| skill.name == "quoted")
            .expect("quoted");
        assert_eq!(quoted.description, "with # hash");
    }

    #[test]
    fn skill_roots_cover_and_deduplicate() {
        let workspace = TempDir::new().expect("temp dir");
        let roots = skill_roots(workspace.path());
        assert_eq!(roots.len(), roots.iter().collect::<HashSet<_>>().len());
        assert!(
            roots
                .iter()
                .any(|root| *root == workspace.path().join("skills"))
        );
        assert!(
            roots
                .iter()
                .any(|root| *root == workspace.path().join(".agents/skills"))
        );
        // Application-root paths end in the same directory names.
        assert!(
            roots
                .iter()
                .filter(|root| root.starts_with(workspace.path()))
                .count()
                == 2
        );
    }

    #[test]
    fn context_formats_entries_and_respects_budget() {
        let skills: Vec<Skill> = (0..3)
            .map(|index| Skill {
                name: format!("skill-{index}"),
                description: "A skill".repeat(60),
                instructions: "body".to_string(),
                directory: PathBuf::from("."),
            })
            .collect();
        let context = format_context(&skills);
        assert!(context.starts_with(SKILLS_GUIDANCE));
        assert!(context.contains("- \"skill-0\": "));
        // Long descriptions get truncated with an ellipsis.
        assert!(context.contains("…"));
        assert_eq!(format_context(&[]), "");
    }

    #[test]
    fn context_omission_note_fits_many_skills() {
        let skills: Vec<Skill> = (0..200)
            .map(|index| Skill {
                name: format!("skill-{index}"),
                description: "x".repeat(120),
                instructions: "body".to_string(),
                directory: PathBuf::from("."),
            })
            .collect();
        let context = format_context(&skills);
        assert!(context.chars().count() <= MAX_SKILL_CONTEXT_CHARS);
        assert!(
            context.contains("skill descriptions omitted by the context size limit"),
            "large lists mention the omission: tail={:?}",
            &context[context.len().saturating_sub(160)..]
        );
    }

    #[test]
    fn load_skill_instructions_and_resources() {
        let workspace = TempDir::new().expect("temp dir");
        let directory = workspace.path().join("skills").join("tools");
        fs::create_dir_all(directory.join("docs")).expect("dirs");
        fs::write(
            directory.join("SKILL.md"),
            "---\nname: tools\ndescription: Tooling Skill\n---\nUse tools.",
        )
        .expect("skill");
        fs::write(directory.join("docs/handbook.md"), "# Handbook\nDetailed.").expect("resource");
        let (skills, warnings) = discover(&[workspace.path().join("skills")]);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(skills.len(), 1);

        let instructions = execute_load_skill(&json!({ "name": "tools" }), skills.as_slice())
            .expect("instructions");
        assert_eq!(instructions, "Skill instructions for tools:\n\nUse tools.");

        let resource = execute_load_skill(
            &json!({ "name": "tools", "path": "docs/handbook.md" }),
            skills.as_slice(),
        )
        .expect("resource");
        assert_eq!(
            resource,
            "Skill resource tools/docs/handbook.md:\n\n# Handbook\nDetailed."
        );
    }

    #[test]
    fn load_skill_blocks_escapes_and_unknown_names() {
        let workspace = TempDir::new().expect("temp dir");
        write_skill(&workspace, "safe", &plain("safe"), "body");
        fs::create_dir_all(
            workspace
                .path()
                .join("skills")
                .join("safe")
                .join("docs/inside/.."),
        )
        .expect("nested");
        // A hard-linked file is blocked even inside the skill directory.
        let inside = workspace
            .path()
            .join("skills")
            .join("safe")
            .join("docs")
            .join("linked.md");
        fs::write(&inside, "duplicated content").expect("original");
        fs::hard_link(
            &inside,
            workspace.path().join("skills").join("safe").join("hard.md"),
        )
        .expect("hardlink");
        fs::write(workspace.path().join("secret.txt"), "top secret").expect("outside");
        let (skills, _) = discover(&skill_roots(workspace.path()));
        assert_eq!(skills.len(), 1);

        let escape = execute_load_skill(
            &json!({ "name": "safe", "path": "../secret.txt" }),
            skills.as_slice(),
        )
        .expect_err("parent escape rejected");
        assert!(
            escape.to_string().contains("outside the skill directory"),
            "escape error: {escape}"
        );

        let absolute = execute_load_skill(
            &json!({ "name": "safe", "path": "/etc/passwd" }),
            skills.as_slice(),
        )
        .expect_err("absolute path rejected");
        assert!(
            absolute.to_string().contains("must be relative"),
            "absolute error: {absolute}"
        );

        let empty = execute_load_skill(&json!({ "name": "safe", "path": "" }), skills.as_slice())
            .expect_err("empty path rejected");
        assert!(
            empty.to_string().contains("non-empty"),
            "empty error: {empty}"
        );

        let hard = execute_load_skill(
            &json!({ "name": "safe", "path": "hard.md" }),
            skills.as_slice(),
        )
        .expect_err("hard link rejected");
        assert!(
            hard.to_string().contains("hard links are blocked"),
            "hard link error: {hard}"
        );

        let unknown = execute_load_skill(&json!({ "name": "ghost" }), skills.as_slice())
            .expect_err("unknown skill");
        assert_eq!(unknown.to_string(), "Skill not found: ghost");
    }
}
