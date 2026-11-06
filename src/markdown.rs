//! Streaming Markdown renderer focused on GFM table alignment, ported from
//! MinAgent's `markdown-terminal.mjs`. Only the table path is rendered
//! (aligned columns with `│` borders, bold header); all other Markdown is
//! passed through raw so nothing else changes. The renderer is streaming
//! aware: only lines that start with `|` at the line start are buffered, so
//! plain prose still streams character-by-character without delay.

use std::io::Write;

use regex::Regex;
use unicode_width::UnicodeWidthChar;

use crate::ui;

#[derive(Clone, Copy, PartialEq)]
enum Align {
    Left,
    Right,
    Center,
}

pub struct MarkdownRenderer {
    at_line_start: bool,
    table_line: Option<String>,
    pending_header: Option<(String, Vec<String>)>,
    table_mode: bool,
    widths: Vec<usize>,
    aligns: Vec<Align>,
    out: String,
}

impl MarkdownRenderer {
    pub fn new() -> Self {
        Self {
            at_line_start: true,
            table_line: None,
            pending_header: None,
            table_mode: false,
            widths: Vec::new(),
            aligns: Vec::new(),
            out: String::new(),
        }
    }

    /// Feeds a streamed chunk through the renderer.
    pub fn write(&mut self, chunk: &str) {
        for character in chunk.chars() {
            if character == '\r' {
                continue;
            }
            if character == '\n' {
                self.newline();
                continue;
            }
            if let Some(line) = self.table_line.as_mut() {
                line.push(character);
                continue;
            }
            if self.at_line_start {
                if character == '|' {
                    self.table_line = Some(String::from("|"));
                    self.at_line_start = false;
                    continue;
                }
                self.at_line_start = false;
                self.finish_table_before_text();
            }
            self.out.push(character);
        }
    }

    /// Writes any buffered text to the terminal. Call after each chunk so
    /// streaming output appears promptly.
    pub fn flush(&mut self) {
        self.flush_out();
    }

    /// Flushes any buffered table state at the end of an assistant message.
    pub fn end(&mut self) {
        if let Some(line) = self.table_line.take() {
            self.consume_table_line(&line, false);
        }
        self.table_mode = false;
        self.flush_pending_header(false);
        self.flush_out();
        *self = Self::new();
    }

    fn flush_out(&mut self) {
        if self.out.is_empty() {
            return;
        }
        let mut stdout = std::io::stdout().lock();
        let _ = stdout.write_all(self.out.as_bytes());
        let _ = stdout.flush();
        self.out.clear();
    }

    fn newline(&mut self) {
        if let Some(line) = self.table_line.take() {
            self.consume_table_line(&line, true);
            self.at_line_start = true;
            return;
        }
        self.finish_table_before_text();
        self.out.push('\n');
        self.at_line_start = true;
    }

    fn finish_table_before_text(&mut self) {
        if !self.table_mode && self.pending_header.is_none() {
            return;
        }
        self.table_mode = false;
        self.widths.clear();
        self.aligns.clear();
        self.flush_pending_header(true);
    }

    fn flush_pending_header(&mut self, include_newline: bool) {
        if let Some((raw, _cells)) = self.pending_header.take() {
            self.out.push_str(&raw);
            if include_newline {
                self.out.push('\n');
            }
        }
    }

    fn consume_table_line(&mut self, line: &str, include_newline: bool) {
        let cells = parse_table_cells(line);
        if self.table_mode {
            if !is_table_separator(&cells) {
                self.render_table_row(&cells, false);
                if include_newline {
                    self.out.push('\n');
                }
            }
            return;
        }
        if let Some((_, header)) = self.pending_header.clone() {
            if is_table_separator(&cells) {
                self.pending_header = None;
                self.initialize_table_columns(&header, &cells);
                self.render_table_row(&header, true);
                if include_newline {
                    self.out.push('\n');
                }
                self.render_table_separator();
                if include_newline {
                    self.out.push('\n');
                }
                self.table_mode = true;
                return;
            }
            // The buffered header was not followed by a separator: it was not a
            // real table. Emit it raw and fall through to handle this line.
            self.flush_pending_header(true);
        }
        if is_table_separator(&cells) {
            // A separator with no header before it is not a table.
            self.out.push_str(line);
            if include_newline {
                self.out.push('\n');
            }
            return;
        }
        self.pending_header = Some((line.to_string(), cells));
    }

    fn initialize_table_columns(&mut self, header: &[String], separator: &[String]) {
        let column_count = header.len().max(separator.len());
        let columns = ui::terminal_columns();
        let content_width = 8.max(96.min(columns.saturating_sub(4)));
        let available_width = column_count.max(content_width.saturating_sub(3 * (column_count - 1)));
        let minimum_width = 1.max(4.min(available_width / column_count));
        self.widths = (0..column_count)
            .map(|index| {
                minimum_width.max(visible_width(&format_table_cell(header.get(index).unwrap_or(&String::new()))))
            })
            .collect();
        let mut total: usize = self.widths.iter().sum();
        while total > available_width {
            let mut widest: Option<usize> = None;
            for (index, width) in self.widths.iter().enumerate() {
                if *width <= minimum_width {
                    continue;
                }
                if widest.map(|w| *width > self.widths[w]).unwrap_or(true) {
                    widest = Some(index);
                }
            }
            let Some(index) = widest else { break };
            self.widths[index] -= 1;
            total -= 1;
        }
        if total < available_width {
            if let Some(last) = self.widths.last_mut() {
                *last += available_width - total;
            }
        }
        self.aligns = (0..column_count)
            .map(|index| {
                let cell = separator.get(index).map(String::as_str).unwrap_or("");
                let starts = cell.starts_with(':');
                let ends = cell.ends_with(':');
                if starts && ends {
                    Align::Center
                } else if ends {
                    Align::Right
                } else {
                    Align::Left
                }
            })
            .collect();
    }

    fn render_table_row(&mut self, cells: &[String], header: bool) {
        let wrapped: Vec<Vec<String>> = self
            .widths
            .iter()
            .enumerate()
            .map(|(index, width)| {
                wrap_table_cell(cells.get(index).unwrap_or(&String::new()), *width)
            })
            .collect();
        let line_count = wrapped.iter().map(|lines| lines.len()).max().unwrap_or(1);
        for line_index in 0..line_count {
            if line_index > 0 {
                self.out.push('\n');
            }
            if header {
                self.out.push_str("\x1b[1;38;2;31;226;220m");
            }
            let rendered: Vec<String> = self
                .widths
                .iter()
                .enumerate()
                .map(|(column, width)| {
                    let cell = wrapped[column]
                        .get(line_index)
                        .map(String::as_str)
                        .unwrap_or("");
                    let padding = width.saturating_sub(visible_width(cell));
                    let left = match self.aligns.get(column).copied().unwrap_or(Align::Left) {
                        Align::Right => padding,
                        Align::Center => padding / 2,
                        Align::Left => 0,
                    };
                    format!("{}{}{}", " ".repeat(left), cell, " ".repeat(padding - left))
                })
                .collect();
            self.out.push_str(&rendered.join(" │ "));
            if header {
                self.out.push_str("\x1b[0m");
            }
        }
    }

    fn render_table_separator(&mut self) {
        let parts: Vec<String> = self
            .widths
            .iter()
            .map(|width| "─".repeat(*width))
            .collect();
        self.out.push_str(&parts.join("─┼─"));
    }
}

/// Splits a GFM table row into trimmed cells, honoring escaped pipes and inline
/// code spans (ported from `parseTableCells`).
fn parse_table_cells(line: &str) -> Vec<String> {
    let source = line.trim().strip_prefix('|').unwrap_or(line.trim());
    let bytes: Vec<char> = source.chars().collect();
    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut index = 0;
    let mut code_ticks = 0;
    while index < bytes.len() {
        let character = bytes[index];
        if character == '\\' && index + 1 < bytes.len() && bytes[index + 1] == '|' {
            cell.push('|');
            index += 2;
            continue;
        }
        if character == '`' {
            let mut run = 1;
            while index + run < bytes.len() && bytes[index + run] == '`' {
                run += 1;
            }
            if code_ticks == 0 {
                code_ticks = run;
            } else if code_ticks == run {
                code_ticks = 0;
            }
            for _ in 0..run {
                cell.push('`');
            }
            index += run;
            continue;
        }
        if character == '|' && code_ticks == 0 {
            cells.push(cell.trim().to_string());
            cell = String::new();
            index += 1;
            continue;
        }
        cell.push(character);
        index += 1;
    }
    cells.push(cell.trim().to_string());
    if source.ends_with('|') && cells.last().map(|c| c.is_empty()).unwrap_or(false) {
        cells.pop();
    }
    cells
}

fn is_table_separator(cells: &[String]) -> bool {
    cells.len() > 1
        && cells
            .iter()
            .all(|cell| separator_regex().is_match(cell))
}

fn separator_regex() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^:?-{3,}:?$").unwrap())
}

/// Strips inline Markdown (images, links, code, bold, italic) from a cell so the
/// aligned text is plain (ported from `formatTableCell`).
fn format_table_cell(value: &str) -> String {
    let text = safe_terminal_text(value);
    let text = image_re().replace_all(&text, "$1");
    let text = link_re().replace_all(&text, "$1");
    let text = code_re().replace_all(&text, "$1");
    let text = bold_re().replace_all(&text, "$2");
    let text = italic_re().replace_all(&text, "$2");
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn safe_terminal_text(value: &str) -> String {
    value
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .chars()
        .filter(|c| !((*c as u32) < 9 || (*c as u32) == 11 || (*c as u32) == 12 || (14..31).contains(&(*c as u32)) || (127..159).contains(&(*c as u32))))
        .collect()
}

fn image_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"!\[([^\]]*)\]\([^)]*\)").unwrap())
}
fn link_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\[([^\]]+)\]\([^)]*\)").unwrap())
}
fn code_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"`+([^`]+)`+").unwrap())
}
fn bold_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(\*\*|__)(.*?)(\*\*|__)").unwrap())
}
fn italic_re() -> &'static Regex {
    // The Rust `regex` crate has no backreferences, so match a single
    // emphasis pair instead of requiring matching markers.
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[*_]([^*_]+)[*_]").unwrap())
}

fn visible_width(value: &str) -> usize {
    let mut width = 0;
    let mut in_escape = false;
    for character in value.chars() {
        if in_escape {
            if character.is_ascii_alphabetic() {
                in_escape = false;
            }
            continue;
        }
        if character == '\x1b' {
            in_escape = true;
            continue;
        }
        width += UnicodeWidthChar::width(character).unwrap_or(0);
    }
    width
}

/// Word-wraps a cell to `width` cells (ported from `wrapMessage`/`wrapTextLine`).
fn wrap_table_cell(value: &str, width: usize) -> Vec<String> {
    let text = format_table_cell(value);
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut line_width = 0;
    let mut current = String::new();
    for word in text.split_whitespace() {
        let word_width = visible_width(word);
        let extra = if line_width > 0 { 1 + word_width } else { word_width };
        if line_width > 0 && line_width + extra > width {
            lines.push(current.trim_end().to_string());
            current = String::new();
            line_width = 0;
        }
        if line_width > 0 {
            current.push(' ');
            line_width += 1;
        }
        current.push_str(word);
        line_width += word_width;
    }
    if !current.is_empty() || lines.is_empty() {
        lines.push(current.trim_end().to_string());
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_table_cells() {
        assert_eq!(parse_table_cells("| a | b | c |"), vec!["a", "b", "c"]);
        assert_eq!(parse_table_cells("|x|"), vec!["x"]);
    }

    #[test]
    fn detects_separator_rows() {
        assert!(is_table_separator(&parse_table_cells("| --- | :---: | ---: |")));
        assert!(!is_table_separator(&parse_table_cells("| a | b |")));
    }

    #[test]
    fn renders_an_aligned_table() {
        let mut renderer = MarkdownRenderer::new();
        renderer.write("| Name | Age |\n");
        renderer.write("| :--- | ---: |\n");
        renderer.write("| Alice | 30 |\n");
        renderer.write("| Bob | 25 |\n");
        // Inspect the accumulated output before `end()` flushes it.
        let out = renderer.out.clone();
        renderer.end();
        // Header row is bold (cyan) and contains the cells.
        assert!(out.contains("Name"));
        assert!(out.contains("Age"));
        // Separator uses box-drawing joins.
        assert!(out.contains("─┼─"));
        // Body rows are aligned with vertical bars.
        assert!(out.contains("Alice"));
        assert!(out.contains("Bob"));
        assert!(out.contains("30"));
        assert!(out.contains("25"));
        assert!(out.contains("│"));
    }

    #[test]
    fn non_table_pipe_text_passes_through_raw() {
        let mut renderer = MarkdownRenderer::new();
        renderer.write("not a table | just text\n");
        let out = renderer.out.clone();
        renderer.end();
        assert!(out.contains("not a table | just text"));
        assert!(!out.contains("─┼─"));
    }

    #[test]
    fn header_without_separator_is_emitted_raw() {
        let mut renderer = MarkdownRenderer::new();
        renderer.write("| Header | Only |\n");
        renderer.write("plain text\n");
        let out = renderer.out.clone();
        renderer.end();
        assert!(out.contains("Header"));
        assert!(out.contains("plain text"));
        // No aligned table should have been rendered.
        assert!(!out.contains("─┼─"));
    }

    #[test]
    fn strips_inline_markdown_in_cells() {
        assert_eq!(format_table_cell("**bold**"), "bold");
        assert_eq!(format_table_cell("`code`"), "code");
        assert_eq!(format_table_cell("[label](http://x)"), "label");
        assert_eq!(format_table_cell("a  b"), "a b");
    }
}
