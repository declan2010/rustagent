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
    *USE_COLOR
        .get_or_init(|| std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none())
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

/// Reverse-video highlight used for the selected row in a selection panel
/// (MinAgent's `32;93;112` background with the pale foreground).
pub fn highlight(text: &str) -> String {
    if !use_color() {
        return text.to_string();
    }
    format!("\x1b[48;2;32;93;112;38;2;226;239;241m{text}\x1b[0m")
}

/// Builds the `/model` selection panel: a title, a centered window of up to 5
/// candidate rows (with a `›` marker and reverse-video highlight on the
/// selected one), and a footer hint. Mirrors MinAgent's `formatAutocompletePanel`.
pub fn model_panel(
    ids: &[String],
    notes: &[String],
    current_id: &str,
    selected: usize,
    columns: usize,
) -> Vec<String> {
    const MAX_ITEMS: usize = 5;
    let total = ids.len();
    let mut lines: Vec<String> = Vec::with_capacity(MAX_ITEMS + 2);
    let title = format!(
        "  ┌─ MODELS · {} match{}",
        total,
        if total == 1 { "" } else { "es" }
    );
    lines.push(magenta_bold(&title));
    let first = selected
        .saturating_sub(MAX_ITEMS / 2)
        .min(total.saturating_sub(MAX_ITEMS));
    for visible in 0..MAX_ITEMS {
        let index = first + visible;
        if index >= total {
            lines.push(String::new());
            continue;
        }
        let marker = if index == selected { "›" } else { " " };
        let mut label = ids[index].clone();
        if label == current_id {
            label.push_str(" (current)");
        }
        if let Some(note) = notes.get(index).filter(|note| !note.is_empty()) {
            label.push_str(note);
        }
        let label = label
            .chars()
            .filter(|c| !c.is_control())
            .collect::<String>();
        let budget = columns.saturating_sub(6).max(1);
        let mut shown = String::new();
        let mut width = 0usize;
        for character in label.chars() {
            let cw = unicode_width::UnicodeWidthChar::width(character).unwrap_or(0);
            if width + cw > budget {
                shown.push('…');
                break;
            }
            shown.push(character);
            width += cw;
        }
        let option = format!("  {marker} {shown}");
        if index == selected {
            lines.push(highlight(&option));
        } else {
            lines.push(muted(&option));
        }
    }
    lines.push(muted("  └─ ↑/↓ select · Enter switch · Esc close"));
    lines
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

/// Banner title with the running version, kept in sync with Cargo.toml via
/// `CARGO_PKG_VERSION`.
fn banner_title() -> String {
    format!("RustAgent {} · SESSION", env!("CARGO_PKG_VERSION"))
}

/// Banner rows for one content line, wrapped to the frame like MinAgent's
/// `panelLine`: `│ {line}{padding} │` measures inner + 4, same as the edges.
fn panel_rows(content: &str, color: (u8, u8, u8), inner: usize) -> Vec<String> {
    wrap_message(content, inner)
        .into_iter()
        .map(|line| {
            let padding = " ".repeat(inner.saturating_sub(UnicodeWidthStr::width(line.as_str())));
            format!(
                "{} {}{} {}",
                cyan("│"),
                styled(&line, color, false),
                padding,
                cyan("│")
            )
        })
        .collect()
}

fn panel_line(content: &str, color: (u8, u8, u8), inner: usize) {
    for row in panel_rows(content, color, inner) {
        println!("{row}");
    }
}

/// Startup panel, mirroring MinAgent's model/context/input/terminal/workspace rows.
pub fn print_banner(config: &Config, used_tokens: u64, skills_loaded: usize, mcp_tools: usize) {
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
        (
            "Terminal",
            permission_label(config.terminal_mode).to_string(),
        ),
        ("Workspace", config.workspace_name.clone()),
        (
            "Config",
            if config.config_files.is_empty() {
                "none (environment variables)".to_string()
            } else {
                config.config_files.clone()
            },
        ),
    ];
    if config.skills_mode != PermissionMode::Off || config.mcp_mode != PermissionMode::Off {
        rows.push((
            "Extensions",
            format!(
                "Skills {}{} · MCP {}{}",
                permission_label(config.skills_mode),
                if config.skills_mode != PermissionMode::Off {
                    format!(" ({skills_loaded} loaded)")
                } else {
                    String::new()
                },
                permission_label(config.mcp_mode),
                if config.mcp_mode != PermissionMode::Off {
                    format!(" ({mcp_tools} tools)")
                } else {
                    String::new()
                }
            ),
        ));
    }

    let title = banner_title();
    let contents: Vec<String> = std::iter::once(title.clone())
        .chain(
            rows.iter()
                .map(|(label, value)| format!("{label:<10} {value}")),
        )
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
            let padding =
                " ".repeat(inner.saturating_sub(UnicodeWidthStr::width(content.as_str())));
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

/// MinAgent's `assistantBackground` navy for the model's bubble.
const ASSISTANT_BACKGROUND: (u8, u8, u8) = (13, 21, 35);

/// Colors for one framed stream block: MinAgent's assistant bubble is a cyan
/// frame with a magenta title and pale text on the navy background, while the
/// reasoning frame reuses the exact same geometry fully muted.
pub struct BubbleStyle {
    border: (u8, u8, u8),
    title: (u8, u8, u8),
    title_bold: bool,
    text: (u8, u8, u8),
    background: Option<(u8, u8, u8)>,
}

impl BubbleStyle {
    pub fn assistant() -> Self {
        Self {
            border: CYAN,
            title: MAGENTA,
            title_bold: true,
            text: PALE,
            background: Some(ASSISTANT_BACKGROUND),
        }
    }

    pub fn thinking() -> Self {
        Self {
            border: MUTED,
            title: MUTED,
            title_bold: false,
            text: MUTED,
            background: None,
        }
    }
}

/// Streams the model's output inside a left-aligned bubble, MinAgent's
/// `createAssistantBubbleWriter`: cyan frame, label in the title, content on
/// the navy background word-wrapped to the content width, and a closing line
/// carrying the stream status (complete/interrupted/incomplete). The bubble
/// opens lazily on the first write, so tool-call-only rounds print nothing.
pub struct AssistantBubble {
    label: String,
    content_width: usize,
    style: BubbleStyle,
    opened: bool,
    line_width: usize,
    line_started: bool,
    line_content: String,
    pending_word: String,
    pending_whitespace: String,
    emit: Box<dyn FnMut(&str)>,
}

impl AssistantBubble {
    pub fn new(label: String) -> Self {
        Self::with_style(label, BubbleStyle::assistant())
    }

    /// Builds a bubble with an explicit style; the muted reasoning frame
    /// reuses the assistant geometry this way.
    pub fn with_style(label: String, style: BubbleStyle) -> Self {
        let columns = terminal_columns();
        Self {
            label,
            // The frame row is content_width + 4 cells wide; keeping it short
            // of the terminal's last column avoids the wrap-pending state on
            // every row (see `user_bubble_geometry` for why that desyncs the
            // transcript at the scroll-region bottom on some terminals).
            content_width: 8.max(96.min(columns.saturating_sub(5))),
            style,
            opened: false,
            line_width: 0,
            line_started: false,
            line_content: String::new(),
            pending_word: String::new(),
            pending_whitespace: String::new(),
            emit: Box::new(|fragment| {
                use std::io::Write;
                let mut stdout = std::io::stdout().lock();
                let _ = stdout.write_all(fragment.as_bytes());
                let _ = stdout.flush();
            }),
        }
    }

    /// Test constructor that captures rows instead of printing them, with a
    /// fixed content width so wrapping is deterministic.
    #[cfg(test)]
    pub fn with_emit(label: String, emit: Box<dyn FnMut(&str)>, content_width: usize) -> Self {
        Self::with_emit_style(label, emit, content_width, BubbleStyle::assistant())
    }

    #[cfg(test)]
    pub fn with_emit_style(
        label: String,
        emit: Box<dyn FnMut(&str)>,
        content_width: usize,
        style: BubbleStyle,
    ) -> Self {
        let mut bubble = Self::new(label);
        bubble.emit = emit;
        bubble.content_width = content_width;
        bubble.style = style;
        bubble
    }

    /// The content width for the row being laid out: the bubble never renders
    /// wider than the live terminal, so a mid-stream resize adapts the frame
    /// instead of wrapping its rows off-screen (MinAgent freezes the width
    /// once per message; here every row clamps to the current terminal).
    fn live_content_width(&self) -> usize {
        match terminal_size() {
            Some((Width(width), _)) => {
                8.max(self.content_width.min(width.saturating_sub(5) as usize))
            }
            None => self.content_width,
        }
    }

    pub fn write(&mut self, text: &str) {
        if !self.opened {
            self.open();
        }
        for character in text.chars() {
            if character == '\r' {
                continue;
            }
            if character == '\n' {
                self.flush_word();
                self.pending_whitespace.clear();
                self.finish_line();
                continue;
            }
            if character == '\t' {
                // A tab renders as a jump to the next 8-column stop, not a
                // zero-width cell; widen it up front so the row's width math
                // measures what actually lands on screen.
                self.flush_word();
                let pending = self
                    .pending_whitespace
                    .chars()
                    .map(character_width)
                    .sum::<usize>();
                let spaces = tab_stop_spaces(2 + self.line_width + pending);
                for _ in 0..spaces {
                    self.pending_whitespace.push(' ');
                }
            } else if character.is_whitespace() {
                self.flush_word();
                self.pending_whitespace.push(character);
            } else {
                self.pending_word.push(character);
            }
        }
    }

    /// Closes the bubble with the stream status; never-opened bubbles stay
    /// silent (tool-call-only rounds print no frame).
    pub fn close(&mut self, status: &str) {
        if !self.opened {
            return;
        }
        self.flush_word();
        self.finish_line();
        let footer_prefix = "╰─";
        let footer_text = format!(" {status} ");
        let footer_fill = (self.live_content_width() + 4)
            .saturating_sub(
                unicode_width::UnicodeWidthStr::width(footer_prefix)
                    + unicode_width::UnicodeWidthStr::width(footer_text.as_str())
                    + 1,
            )
            .max(1);
        (self.emit)(&format!(
            "{}{}{}\n",
            styled(footer_prefix, self.style.border, false),
            muted(&footer_text),
            styled(
                &format!("{}╯", "─".repeat(footer_fill)),
                self.style.border,
                false
            )
        ));
        self.opened = false;
    }

    fn open(&mut self) {
        self.opened = true;
        let title_prefix = "╭─";
        let width = self.live_content_width();
        let budget = width.saturating_sub(3).max(5);
        let mut shown = String::new();
        let mut used = 0usize;
        for character in self.label.chars() {
            let width = character_width(character);
            if used + width > budget {
                break;
            }
            used += width;
            shown.push(character);
        }
        let title_text = format!(" {shown} ");
        let title_fill = (width + 4)
            .saturating_sub(
                unicode_width::UnicodeWidthStr::width(title_prefix)
                    + unicode_width::UnicodeWidthStr::width(title_text.as_str())
                    + 1,
            )
            .max(1);
        (self.emit)(&format!(
            "{}{}{}\n",
            styled(title_prefix, self.style.border, false),
            styled(&title_text, self.style.title, self.style.title_bold),
            styled(
                &format!("{}╮", "─".repeat(title_fill)),
                self.style.border,
                false
            )
        ));
    }

    fn flush_word(&mut self) {
        if self.pending_word.is_empty() {
            return;
        }
        let word_width = self
            .pending_word
            .chars()
            .map(character_width)
            .sum::<usize>();
        if word_width > 0 {
            let whitespace_width = self
                .pending_whitespace
                .chars()
                .map(character_width)
                .sum::<usize>();
            // A wrap discards the whitespace that preceded the word, so it
            // must not count towards the new line's width either; carrying
            // it over left the wrapped row one column short of the frame.
            let mut carried_whitespace = whitespace_width;
            if self.line_width > 0
                && self.line_width + whitespace_width + word_width > self.live_content_width()
            {
                self.finish_line();
                self.pending_whitespace.clear();
                carried_whitespace = 0;
            }
            // Mark the line started before the whitespace lands: the first
            // word cluster would otherwise clear what a leading widened tab
            // already wrote while its width stays counted.
            if !self.line_started {
                self.line_started = true;
                self.line_content.clear();
            }
            self.line_content.push_str(&self.pending_whitespace);
            self.line_width += carried_whitespace;
            self.pending_whitespace.clear();
        }
        let word = std::mem::take(&mut self.pending_word);
        for character in word.chars() {
            self.emit_cluster(character);
        }
    }

    fn emit_cluster(&mut self, character: char) {
        let width = character_width(character);
        if self.line_width > 0 && self.line_width + width > self.live_content_width() {
            self.finish_line();
        }
        if !self.line_started {
            self.line_started = true;
            self.line_content.clear();
        }
        self.line_content.push(character);
        self.line_width += width;
    }

    fn finish_line(&mut self) {
        if !self.line_started {
            self.line_started = true;
            self.line_content.clear();
        }
        let padding = self.live_content_width().saturating_sub(self.line_width) + 1;
        let content = format!(" {}{}", self.line_content, " ".repeat(padding));
        let content = match self.style.background {
            Some(background) => bubble_text(&content, self.style.text, background),
            None => styled(&content, self.style.text, false),
        };
        let row = format!(
            "{}{}{}\n",
            styled("│", self.style.border, false),
            content,
            styled("│", self.style.border, false)
        );
        (self.emit)(&row);
        self.line_width = 0;
        self.line_started = false;
        self.line_content.clear();
    }
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
/// Echoes the user's submitted message as a right-aligned YOU bubble with a
/// magenta frame and teal content, matching MinAgent's `printUserBubble`.
pub fn print_submitted(text: &str) {
    let (indent, content_width, lines) = user_bubble_geometry(text, terminal_columns());
    let title = "╭─ YOU ";
    let title_width = unicode_width::UnicodeWidthStr::width(title);
    let title_fill = (content_width + 4).saturating_sub(title_width + 1);
    println!(
        "{indent}{}{}",
        magenta_bold(title),
        magenta(&format!("{}╮", "─".repeat(title_fill)))
    );
    for line in &lines {
        let padding = " ".repeat(
            content_width.saturating_sub(unicode_width::UnicodeWidthStr::width(line.as_str())),
        );
        println!(
            "{indent}{}{}{}",
            magenta("│"),
            bubble_text(&format!(" {line}{padding} "), PALE, USER_BACKGROUND),
            magenta("│")
        );
    }
    println!(
        "{indent}{}",
        magenta(&format!("╰{}╯", "─".repeat(content_width + 2)))
    );
}

/// Shared bubble geometry: the right-side indent, the content width, and the
/// wrapped message lines (the same computation `print_submitted` renders).
fn user_bubble_geometry(text: &str, columns: usize) -> (String, usize, Vec<String>) {
    let columns = columns.max(1);
    let max_content_width = 4.max(66.min(columns.saturating_sub(8)));
    let lines = wrap_message(text, max_content_width);
    let content_width = lines
        .iter()
        .map(|line| unicode_width::UnicodeWidthStr::width(line.as_str()))
        .max()
        .unwrap_or(4)
        .max(4);
    let outer_width = content_width + 4;
    // One spare column keeps every echo row short of the terminal's last
    // column: a row that lands exactly on the right edge leaves the terminal
    // in the wrap-pending state, and the newline that follows then behaves
    // differently across terminals (Konsole vs xterm) right where the row
    // crosses the scroll-region bottom — which desyncs the transcript.
    let indent = " ".repeat(columns.saturating_sub(outer_width + 1));
    (indent, content_width, lines)
}

/// MinAgent's `userBackground` teal for the YOU bubble.
const USER_BACKGROUND: (u8, u8, u8) = (18, 49, 58);

pub fn magenta(text: &str) -> String {
    styled(text, MAGENTA, false)
}

/// Foreground on a background cell, MinAgent's `uiBubbleText`.
fn bubble_text(text: &str, foreground: (u8, u8, u8), background: (u8, u8, u8)) -> String {
    if !use_color() {
        return text.to_string();
    }
    let (red, green, blue) = foreground;
    let (back_red, back_green, back_blue) = background;
    format!("\x1b[38;2;{red};{green};{blue};48;2;{back_red};{back_green};{back_blue}m{text}\x1b[0m")
}

fn character_width(character: char) -> usize {
    unicode_width::UnicodeWidthChar::width(character).unwrap_or(0)
}

/// The spaces a tab occupies at this on-screen column: terminals advance the
/// cursor to the next 8-column stop, never to a zero-width cell.
fn tab_stop_spaces(column: usize) -> usize {
    8 - (column % 8)
}

/// Widens tabs into the spaces the terminal would actually render, counting
/// columns from `offset` (a frame's content edge sits two columns in, after
/// `│ `). Zero-width tabs used to under-measure rows, so the frame packed
/// more text than fit and the terminal wrapped the overflowed rows.
fn expand_tabs(value: &str, offset: usize) -> String {
    if !value.contains('\t') {
        return value.to_string();
    }
    let mut expanded = String::with_capacity(value.len());
    let mut column = offset;
    for character in value.chars() {
        if character == '\t' {
            let spaces = tab_stop_spaces(column);
            for _ in 0..spaces {
                expanded.push(' ');
            }
            column += spaces;
        } else {
            column += character_width(character);
            expanded.push(character);
        }
    }
    expanded
}

/// Word-wraps text like MinAgent's `wrapTextLine`: break at the last space
/// within the width when possible, hard-break words that do not fit, and trim
/// each line's trailing whitespace.
pub fn wrap_message(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for line in text.split('\n') {
        wrap_line(&expand_tabs(line, 2), width, &mut lines);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn wrap_line(value: &str, width: usize, lines: &mut Vec<String>) {
    let characters: Vec<char> = value.chars().collect();
    let widths: Vec<usize> = characters
        .iter()
        .map(|character| character_width(*character))
        .collect();
    let mut remaining_width: isize = widths.iter().sum::<usize>() as isize;
    let width = width as isize;
    let mut start = 0usize;
    while remaining_width > width {
        let mut used_width = 0isize;
        let mut cut = start;
        let mut last_space: Option<usize> = None;
        for index in start..characters.len() {
            let character_width = widths[index] as isize;
            if used_width + character_width > width {
                break;
            }
            used_width += character_width;
            cut = index + 1;
            if characters[index].is_whitespace() {
                last_space = Some(index);
            }
        }
        let break_at = if cut < characters.len() && characters[cut].is_whitespace() {
            cut
        } else if let Some(last_space) = last_space.filter(|at| *at > start) {
            last_space
        } else {
            cut.max(start + 1)
        };
        lines.push(
            characters[start..break_at]
                .iter()
                .collect::<String>()
                .trim_end()
                .to_string(),
        );
        let mut next_start = break_at;
        while next_start < characters.len() && characters[next_start].is_whitespace() {
            next_start += 1;
        }
        for index in start..next_start {
            remaining_width -= widths[index] as isize;
        }
        start = next_start;
    }
    lines.push(characters[start..].iter().collect());
}

pub fn print_help() {
    println!("{}", magenta_bold("╭─ COMMANDS"));
    println!(
        "  {} {}",
        cyan_bold("/new     "),
        pale("Start a new conversation and clear the screen")
    );
    println!(
        "  {} {}",
        cyan_bold("/model   "),
        pale("List or switch API models (/model <number|name>)")
    );
    println!(
        "  {} {}",
        cyan_bold("/compact "),
        pale("Compact conversation history manually")
    );
    println!("  {} {}", cyan_bold("/help    "), pale("Show this help"));
    println!("  {} {}", cyan_bold("/exit    "), pale("Exit RustAgent"));
    println!("{}", muted("╰─ Enter to send · / for commands"));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strips ANSI styling so emitted rows can be measured by column.
    fn plain_text(row: &str) -> String {
        let mut out = String::new();
        let mut chars = row.chars().peekable();
        while let Some(character) = chars.next() {
            if character == '\u{1b}' && chars.peek() == Some(&'[') {
                chars.next();
                for next in chars.by_ref() {
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
                continue;
            }
            out.push(character);
        }
        out
    }

    #[test]
    fn banner_title_carries_the_running_version() {
        assert_eq!(
            banner_title(),
            format!("RustAgent {} · SESSION", env!("CARGO_PKG_VERSION"))
        );
    }

    #[test]
    fn banner_rows_align_with_the_edges() {
        // The banner's panel rows measure inner + 4 like the edges; long
        // content wraps inside the frame instead of pushing the border out.
        for content in [
            "RustAgent · SESSION",
            "Context    ~1,246 / 65,536 tokens  1.9%  \u{2570}\u{2571}\u{2571}",
            "Workspace  a-very-long-workspace-name-that-exceeds-the-frame-and-must-wrap-inside",
        ] {
            for row in panel_rows(content, MAGENTA, 51) {
                let plain = plain_text(&row);
                assert_eq!(
                    unicode_width::UnicodeWidthStr::width(plain.as_str()),
                    51 + 4,
                    "misaligned banner row: {plain:?}"
                );
            }
        }
        let wrapped = panel_rows(
            "Workspace  a-very-long-workspace-name-that-exceeds-the-frame",
            PALE,
            51,
        );
        assert!(
            wrapped.len() >= 2,
            "long banner content wraps inside the frame: {wrapped:?}"
        );
    }

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
        assert_eq!(
            usage_meter(50.0, 10),
            format!("{}{}", "█".repeat(5), "░".repeat(5))
        );
    }

    #[test]
    fn elapsed_matches_minagent_units() {
        assert_eq!(format_elapsed(Duration::from_secs(9)), "9s");
        assert_eq!(format_elapsed(Duration::from_secs(83)), "1m23s");
        assert_eq!(format_elapsed(Duration::from_secs(3_723)), "1h02m");
    }
}

#[cfg(test)]
mod ui_bubble_tests {
    use super::*;

    #[test]
    fn wrap_breaks_at_spaces_and_hard_breaks_long_words() {
        let lines = wrap_message("hello world this is rustagent", 11);
        assert_eq!(
            lines,
            vec![
                "hello world".to_string(),
                "this is".to_string(),
                "rustagent".to_string(),
            ]
        );
        let lines = wrap_message("abcdefghij klm", 4);
        assert_eq!(
            lines,
            vec![
                "abcd".to_string(),
                "efgh".to_string(),
                "ij".to_string(),
                "klm".to_string(),
            ]
        );
        let lines = wrap_message("one\ntwo", 10);
        assert_eq!(lines, vec!["one".to_string(), "two".to_string()]);
        assert_eq!(wrap_message("", 10), vec![String::new()]);
    }

    #[test]
    fn bubble_geometry_pushes_right_and_keeps_the_frame_closed() {
        let (indent, content_width, lines) = user_bubble_geometry("hi there", 80);
        // The frame is outer = content + 4 wide, and the indent keeps one
        // spare column so no echo row ends on the terminal's last column
        // (wrap-pending at the scroll-region bottom desyncs the transcript).
        assert_eq!(indent.len(), 80 - (content_width + 4) - 1);
        assert_eq!(lines, vec!["hi there".to_string()]);
        assert_eq!(content_width, 8);
        // A long message wraps at 66 content columns max.
        let long = "word ".repeat(30);
        let (_, long_width, long_lines) = user_bubble_geometry(&long, 80);
        assert!(long_width <= 66);
        assert!(long_lines.len() > 1);
        // Narrow terminals shrink the content width but keep the frame math;
        // the bubble never renders narrower than a 4-column content floor.
        let (narrow_indent, narrow_width, _) = user_bubble_geometry("hi", 20);
        assert_eq!(narrow_width, 4);
        assert_eq!(
            narrow_indent.len(),
            20usize.saturating_sub(narrow_width + 4 + 1)
        );
    }
}

#[cfg(test)]
mod assistant_bubble_tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn captured_bubble(label: &str) -> (AssistantBubble, Rc<RefCell<Vec<String>>>) {
        let rows: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let sink_rows = Rc::clone(&rows);
        let bubble = AssistantBubble::with_emit(
            label.to_string(),
            Box::new(move |fragment| {
                sink_rows.borrow_mut().push(fragment.to_string());
            }),
            20,
        );
        (bubble, rows)
    }

    #[test]
    fn assistant_bubble_frames_and_wraps_streamed_text() {
        let (mut bubble, rows) = captured_bubble("Model · test");
        bubble.write("hello world this is a much longer message that must wrap");
        bubble.close("complete");
        let rows = rows.borrow();
        assert!(
            rows.first().expect("title").contains("╭─ Model · test"),
            "title row: {rows:?}"
        );
        let content: Vec<&String> = rows.iter().filter(|row| row.contains("│")).collect();
        assert!(content.len() >= 2, "the long message wraps: {rows:?}");
        let footer = rows.last().expect("footer");
        assert!(footer.contains("╰─ complete"), "footer: {rows:?}");
        assert!(footer.contains("╯"));
    }

    #[test]
    fn assistant_bubble_opens_lazily_and_silent_when_empty() {
        let (mut bubble, rows) = captured_bubble("Model · test");
        bubble.close("complete");
        assert!(rows.borrow().is_empty(), "no text, no bubble");
        bubble.write("ok");
        bubble.close("interrupted");
        let rows = rows.borrow();
        assert!(rows.first().expect("title").contains("╭─ Model · test"));
        assert!(
            rows.iter().any(|row| row.contains("ok")),
            "content: {rows:?}"
        );
        assert!(
            rows.last().expect("footer").contains("╰─ interrupted"),
            "footer: {rows:?}"
        );
    }

    /// Strips ANSI color codes so captured rows can be measured by column.
    fn plain_row(row: &str) -> String {
        let mut out = String::new();
        let mut chars = row.chars().peekable();
        while let Some(character) = chars.next() {
            if character == '\u{1b}' && chars.peek() == Some(&'[') {
                chars.next();
                for next in chars.by_ref() {
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
                continue;
            }
            out.push(character);
        }
        out
    }

    #[test]
    fn assistant_bubble_wrapped_rows_align_with_the_frame() {
        // A wrap discards the whitespace that led the wrapped word; that
        // discarded width used to leak into the padding and pulled the
        // right border one column short on exactly the wrapped rows.
        let rows: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let sink_rows = Rc::clone(&rows);
        let mut bubble = AssistantBubble::with_emit(
            "Model · test".to_string(),
            Box::new(move |fragment| {
                sink_rows.borrow_mut().push(fragment.to_string());
            }),
            96,
        );
        bubble.write(
            "**Phases written:** shared data layer \u{2192} navigation shell + CSS \u{2192} categories \
             \u{2192} products \u{2192} movements \u{2192} stock/min alerts \u{2192} reports \u{2192} \
             inventory value.\n",
        );
        bubble.close("complete");
        let content: Vec<String> = rows
            .borrow()
            .iter()
            .map(|row| plain_row(row))
            .map(|row| row.trim_end_matches('\n').to_string())
            .filter(|row| {
                (row.starts_with('\u{2502}') && row.ends_with('\u{2502}'))
                    || (row.starts_with("\u{256d}\u{2500}") && row.ends_with('\u{256e}'))
                    || (row.starts_with("\u{2570}\u{2500}") && row.ends_with('\u{256f}'))
            })
            .collect();
        assert_eq!(
            content.len(),
            rows.borrow().len(),
            "title, content, and footer rows are all framed: {:?}",
            rows.borrow()
        );
        assert!(
            content.len() >= 4,
            "the paragraph wraps: {:?}",
            rows.borrow()
        );
        assert!(
            content[2].starts_with("\u{2502} movements"),
            "the wrapped row drops the leading whitespace: {:?}",
            content[2]
        );
        for row in &content {
            assert_eq!(
                unicode_width::UnicodeWidthStr::width(row.as_str()),
                96 + 4,
                "misaligned row: {row:?}"
            );
        }
    }

    #[test]
    fn thinking_style_frames_muted_wrapped_rows() {
        // The reasoning frame reuses the assistant geometry, fully muted and
        // without a background, and closes with the round status.
        let rows: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let sink_rows = Rc::clone(&rows);
        let mut bubble = AssistantBubble::with_emit_style(
            "Thinking · test".to_string(),
            Box::new(move |fragment| {
                sink_rows.borrow_mut().push(fragment.to_string());
            }),
            96,
            BubbleStyle::thinking(),
        );
        bubble.write(&format!(
            "{} ",
            "the model rambles on and on here".repeat(6)
        ));
        bubble.close("complete");
        let rows = rows.borrow();
        assert!(
            rows.first().expect("title").contains("╭─ Thinking · test"),
            "title row: {rows:?}"
        );
        assert!(
            rows.last().expect("footer").contains("╰─ complete"),
            "footer: {rows:?}"
        );
        assert!(
            !rows.iter().any(|row| row.contains("48;2;")),
            "no background: {rows:?}"
        );
        let framed: Vec<String> = rows
            .iter()
            .map(|row| plain_row(row))
            .map(|row| row.trim_end_matches('\n').to_string())
            .filter(|row| {
                (row.starts_with('\u{2502}') && row.ends_with('\u{2502}'))
                    || (row.starts_with("\u{256d}\u{2500}") && row.ends_with('\u{256e}'))
                    || (row.starts_with("\u{2570}\u{2500}") && row.ends_with('\u{256f}'))
            })
            .collect();
        assert_eq!(framed.len(), rows.len(), "every row framed: {rows:?}");
        assert!(framed.len() >= 3, "the rambling wraps: {rows:?}");
        for row in &framed {
            assert_eq!(
                unicode_width::UnicodeWidthStr::width(row.as_str()),
                96 + 4,
                "misaligned row: {row:?}"
            );
        }
    }

    #[test]
    fn assistant_bubble_expands_tabs_to_terminal_stops() {
        // Models tab-indent bullets (`*\t**word**`); a zero-width tab used to
        // under-measure the row, the frame packed more text than fit, and the
        // terminal wrapped the border onto the next line.
        let text = "*\t**Capacitor:** La presencia de carpetas `android`, `ios`, \
             `android-native` y el archivo `capacitor.config.ts` indica que es una \
             aplicaci\u{f3}n h\u{ed}brida.\n\t*second* bullet starts with its own tab\n";
        for streamed in [false, true] {
            let rows: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
            let sink_rows = Rc::clone(&rows);
            let mut bubble = AssistantBubble::with_emit_style(
                "Thinking \u{b7} test".to_string(),
                Box::new(move |fragment| {
                    sink_rows.borrow_mut().push(fragment.to_string());
                }),
                96,
                BubbleStyle::thinking(),
            );
            if streamed {
                for character in text.chars() {
                    bubble.write(&character.to_string());
                }
            } else {
                bubble.write(&text);
            }
            bubble.close("complete");
            let framed: Vec<String> = rows
                .borrow()
                .iter()
                .map(|row| plain_row(row))
                .map(|row| row.trim_end_matches('\n').to_string())
                .filter(|row| row.starts_with('\u{2502}') && row.ends_with('\u{2502}'))
                .collect();
            assert!(framed.len() >= 3, "the bullets wrap: {:?}", rows.borrow());
            for row in &framed {
                assert!(!row.contains('\t'), "tabs are widened: {row:?}");
                assert_eq!(
                    unicode_width::UnicodeWidthStr::width(row.as_str()),
                    96 + 4,
                    "misaligned row: {row:?}"
                );
            }
            // `*` sits at content column 0 (screen column 2), so the tab jumps
            // to screen column 8: five spaces.
            assert!(
                framed[0].starts_with("\u{2502} *     **Capacitor:**"),
                "the tab widens to the 8-column stop: {:?}",
                framed[0]
            );
            // A leading tab of its own: six spaces from the content edge.
            assert!(
                framed
                    .iter()
                    .any(|row| row.starts_with("\u{2502}       *second*")),
                "a leading tab widens to six spaces: {:?}",
                framed
            );
        }
    }

    #[test]
    fn wrap_message_expands_tabs_to_terminal_stops() {
        assert_eq!(
            wrap_message("\tx\na\tb", 10),
            vec!["      x".to_string(), "a     b".to_string()]
        );
        // Consecutive tabs walk stop by stop: from column 2 to 8 is six
        // spaces, from 8 to 16 is eight more.
        assert_eq!(
            wrap_message("\t\tend", 30),
            vec![format!("{}end", " ".repeat(6 + 8))]
        );
    }

    #[test]
    fn assistant_bubble_hard_breaks_long_words() {
        let (mut bubble, rows) = captured_bubble("Model · test");
        bubble.write("aaaaaaaaaabbbbbbbbbbccccc");
        bubble.close("complete");
        let rows = rows.borrow();
        let content: Vec<&String> = rows.iter().filter(|row| row.contains("│")).collect();
        assert!(content.len() >= 2, "long word hard-breaks: {rows:?}");
    }
}
