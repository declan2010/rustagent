use std::io::{IsTerminal, Write};
use std::sync::OnceLock;
use std::time::Duration;

use terminal_size::{Width, terminal_size};
use unicode_width::UnicodeWidthStr;

use crate::config::{Config, PermissionMode};

// MinAgent's truecolor palette, kept identical so both agents look the same.
pub const CYAN: (u8, u8, u8) = (31, 226, 220);
pub const MAGENTA: (u8, u8, u8) = (255, 48, 167);
pub const PALE: (u8, u8, u8) = (226, 239, 241);
pub const MUTED: (u8, u8, u8) = (130, 153, 164);
pub const SUCCESS: (u8, u8, u8) = (112, 224, 154);
pub const WARNING: (u8, u8, u8) = (255, 177, 109);
pub const YELLOW: (u8, u8, u8) = (255, 214, 74);
pub const ERROR: (u8, u8, u8) = (255, 108, 132);
pub const PLAN_ORANGE: (u8, u8, u8) = (255, 197, 92);

fn use_color() -> bool {
    // TTY status and NO_COLOR are fixed for the life of the process; caching
    // avoids an ioctl plus getenv on every styled segment while streaming.
    static USE_COLOR: OnceLock<bool> = OnceLock::new();
    *USE_COLOR.get_or_init(|| {
        std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
    })
}

pub fn styled(text: &str, color: (u8, u8, u8), bold: bool) -> String {
    if !use_color() {
        return text.to_string();
    }
    let (red, green, blue) = color;
    if bold {
        format!("\x1b[1;38;2;{red};{green};{blue}m{text}\x1b[0m")
    } else {
        format!("\x1b[38;2;{red};{green};{blue}m{text}\x1b[0m")
    }
}

pub fn cyan(text: &str) -> String {
    styled(text, CYAN, false)
}

pub fn cyan_bold(text: &str) -> String {
    styled(text, CYAN, true)
}

pub fn magenta_bold(text: &str) -> String {
    styled(text, MAGENTA, true)
}

pub fn pale(text: &str) -> String {
    styled(text, PALE, false)
}

pub fn pale_bold(text: &str) -> String {
    styled(text, PALE, true)
}

pub fn muted(text: &str) -> String {
    styled(text, MUTED, false)
}

pub fn success_bold(text: &str) -> String {
    styled(text, SUCCESS, true)
}

pub fn warning(text: &str) -> String {
    styled(text, WARNING, false)
}

pub fn warning_bold(text: &str) -> String {
    styled(text, WARNING, true)
}

pub fn yellow(text: &str) -> String {
    styled(text, YELLOW, false)
}

pub fn error_color(text: &str) -> String {
    styled(text, ERROR, false)
}

pub fn dim(text: &str) -> String {
    muted(text)
}

pub fn red(text: &str) -> String {
    error_color(text)
}

pub fn sep() -> String {
    muted(" · ")
}

pub fn terminal_columns() -> usize {
    terminal_size()
        .map(|(Width(width), _)| width as usize)
        .unwrap_or(80)
        .max(1)
}

pub fn permission_label(mode: PermissionMode) -> &'static str {
    match mode {
        PermissionMode::Auto => "Auto",
        PermissionMode::Ask => "Ask",
        PermissionMode::Off => "Off",
    }
}

pub fn format_number(value: u64) -> String {
    let digits = value.to_string();
    let mut output = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            output.push(',');
        }
        output.push(character);
    }
    output
}

pub fn usage_meter(percent: f64, width: usize) -> String {
    let filled = ((percent / 100.0) * width as f64)
        .round()
        .clamp(0.0, width as f64) as usize;
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

pub fn format_elapsed(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds < 60 {
        return format!("{seconds}s");
    }
    let minutes = seconds / 60;
    if minutes < 60 {
        return format!("{}m{:02}s", minutes, seconds % 60);
    }
    format!("{}h{:02}m", minutes / 60, minutes % 60)
}

fn panel_line(content: &str, color: (u8, u8, u8), inner: usize) {
    let padding = " ".repeat(inner.saturating_sub(UnicodeWidthStr::width(content)));
    println!("{} {} {} {}", cyan("│"), styled(content, color, false), padding, cyan("│"));
}

/// Startup panel, mirroring MinAgent's model/context/input/terminal/workspace rows.
pub fn print_banner(config: &Config, used_tokens: u64) {
    let window = config.context_window.max(1);
    let percent = used_tokens as f64 / window as f64 * 100.0;
    let mut rows = vec![
        ("Model", config.model.clone()),
        (
            "Context",
            format!(
                "~{} / {} tokens  {:.1}%  {}",
                format_number(used_tokens),
                format_number(window),
                percent,
                usage_meter(percent, 10)
            ),
        ),
        ("Input", config.input_modalities.join(" · ")),
        ("Terminal", permission_label(config.terminal_mode).to_string()),
        ("Workspace", config.workspace_name.clone()),
    ];
    if config.skills_mode != PermissionMode::Off || config.mcp_mode != PermissionMode::Off {
        rows.push((
            "Extensions",
            format!(
                "Skills {} · MCP {}",
                permission_label(config.skills_mode),
                permission_label(config.mcp_mode)
            ),
        ));
    }

    let title = "RustAgent · SESSION".to_string();
    let contents: Vec<String> = std::iter::once(title.clone())
        .chain(rows.iter().map(|(label, value)| format!("{label:<10} {value}")))
        .collect();
    let max_inner = terminal_columns().saturating_sub(4).max(4);
    let inner = contents
        .iter()
        .map(|content| UnicodeWidthStr::width(content.as_str()))
        .max()
        .unwrap_or(4)
        .min(max_inner)
        .max(4);
    let edge = |left: &str, right: &str| format!("{left}{}{right}", "─".repeat(inner + 2));

    println!();
    println!("{}", cyan(&edge("╭", "╮")));
    panel_line(&title, MAGENTA, inner);
    for (label, value) in &rows {
        let content = format!("{label:<10} {value}");
        if UnicodeWidthStr::width(content.as_str()) <= inner {
            let padding = " ".repeat(inner.saturating_sub(UnicodeWidthStr::width(content.as_str())));
            println!(
                "{} {} {}{} {}",
                cyan("│"),
                muted(&format!("{label:<10}")),
                pale(value),
                padding,
                cyan("│")
            );
        } else {
            panel_line(label, MUTED, inner);
            panel_line(&format!("  {value}"), PALE, inner);
        }
    }
    println!("{}", cyan(&edge("╰", "╯")));
    println!();
}

pub fn clear_screen() {
    if std::io::stdout().is_terminal() {
        print!("\x1b[2J\x1b[H");
        let _ = std::io::stdout().flush();
    }
}

/// Prompt label for the current agent mode, matching MinAgent's `modeInputPrompt`.
pub fn mode_prompt(mode: crate::agent::Mode) -> String {
    let label = mode.as_str();
    if !use_color() {
        return format!("{label} › ");
    }
    format!("{} {} ", styled(label, mode.color(), true), cyan("›"))
}

pub fn stream_text(chunk: &str) {
    let mut stdout = std::io::stdout().lock();
    let _ = stdout.write_all(chunk.as_bytes());
    let _ = stdout.flush();
}

pub fn stream_reasoning(chunk: &str) {
    let mut stdout = std::io::stdout().lock();
    if use_color() {
        let (red, green, blue) = MUTED;
        let _ = write!(stdout, "\x1b[38;2;{red};{green};{blue}m{chunk}\x1b[0m");
    } else {
        let _ = stdout.write_all(chunk.as_bytes());
    }
    let _ = stdout.flush();
}

pub fn print_round_separator() {
    println!();
}

pub fn print_tool_call(name: &str, arguments: &str) {
    let mut preview: String = arguments.chars().take(200).collect();
    if arguments.chars().count() > 200 {
        preview.push('…');
    }
    println!();
    println!("{} {}", magenta_bold("[tool]"), cyan_bold(name));
    if !preview.is_empty() && preview != "{}" {
        println!("{}", muted(&format!("       {preview}")));
    }
}

pub fn print_tool_result(summary: &str) {
    if summary.is_empty() {
        return;
    }
    println!("{}", muted(&format!("       {summary}")));
}

pub fn print_error(message: &str) {
    println!("{} {}", red("[error]"), message);
}

pub fn print_info(message: &str) {
    println!("{message}");
}

/// Echoes a submitted prompt above the footer, keeping the transcript readable.
pub fn print_submitted(prompt: &str, text: &str) {
    println!("{prompt}{text}");
}

pub fn print_help() {
    println!("{}", magenta_bold("╭─ COMMANDS"));
    println!("  {} {}", cyan_bold("/new     "), pale("Start a new conversation and clear the screen"));
    println!("  {} {}", cyan_bold("/model   "), pale("List or switch API models (/model <number|name>)"));
    println!("  {} {}", cyan_bold("/compact "), pale("Compact conversation history manually"));
    println!("  {} {}", cyan_bold("/help    "), pale("Show this help"));
    println!("  {} {}", cyan_bold("/exit    "), pale("Exit RustAgent"));
    println!("{}", muted("╰─ Enter to send · / for commands"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_use_thousands_separators() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1_000), "1,000");
        assert_eq!(format_number(40_960), "40,960");
        assert_eq!(format_number(262_144), "262,144");
    }

    #[test]
    fn meters_fill_and_clamp() {
        assert_eq!(usage_meter(0.0, 10), "░".repeat(10));
        assert_eq!(usage_meter(100.0, 10), "█".repeat(10));
        assert_eq!(usage_meter(150.0, 10), "█".repeat(10));
        assert_eq!(usage_meter(50.0, 10), format!("{}{}", "█".repeat(5), "░".repeat(5)));
    }

    #[test]
    fn elapsed_matches_minagent_units() {
        assert_eq!(format_elapsed(Duration::from_secs(9)), "9s");
        assert_eq!(format_elapsed(Duration::from_secs(83)), "1m23s");
        assert_eq!(format_elapsed(Duration::from_secs(3_723)), "1h02m");
    }
}
