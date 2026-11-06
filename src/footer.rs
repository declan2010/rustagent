//! Persistent bottom footer, ported from MinAgent's `terminal-footer.mjs`.
//!
//! The transcript keeps the terminal's scrolling region (rows `1..bottom`); the
//! footer owns the rows below it: a status row, a hints row on tall terminals,
//! and a fixed input area. Rendering saves and restores the transcript cursor so
//! output continues where it left off. The real cursor stays hidden and a
//! reverse-video caret is drawn inside the input area. Non-interactive output
//! falls back to plain status lines.

use std::cell::RefCell;
use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

use terminal_size::{Height, Width, terminal_size};
use unicode_width::UnicodeWidthChar;

use crate::ui;

const SAVE_CURSOR: &str = "\x1b7";
const RESTORE_CURSOR: &str = "\x1b8";
const HIDE_CURSOR: &str = "\x1b[?25l";
const SHOW_CURSOR: &str = "\x1b[?25h";
const MIN_HEIGHT: u16 = 3;
const MIN_HEIGHT_FOR_STATUS: u16 = 6;
const MIN_HEIGHT_FOR_HINTS: u16 = 12;
const MAX_INPUT_ROWS_TALL: u16 = 3;
const RENDER_INTERVAL: Duration = Duration::from_millis(100);

fn dimensions() -> (u16, u16) {
    terminal_size()
        .map(|(Width(width), Height(height))| (height.max(1), width.max(1)))
        .unwrap_or((24, 80))
}

fn status_rows(rows: u16) -> u16 {
    if rows >= MIN_HEIGHT_FOR_STATUS { 1 } else { 0 }
}

fn hint_rows(rows: u16) -> u16 {
    if rows >= MIN_HEIGHT_FOR_HINTS { 1 } else { 0 }
}

fn input_rows(rows: u16) -> u16 {
    if rows < MIN_HEIGHT {
        0
    } else if rows >= MIN_HEIGHT_FOR_HINTS {
        MAX_INPUT_ROWS_TALL
    } else {
        1
    }
}

fn footer_height(rows: u16) -> u16 {
    status_rows(rows) + hint_rows(rows) + input_rows(rows)
}

fn region_bottom(rows: u16) -> u16 {
    rows - footer_height(rows)
}

fn status_row(rows: u16) -> u16 {
    if status_rows(rows) == 1 {
        rows - footer_height(rows) + 1
    } else {
        0
    }
}

fn hints_row(rows: u16) -> u16 {
    if hint_rows(rows) == 1 {
        rows - footer_height(rows) + 1 + status_rows(rows)
    } else {
        0
    }
}

fn input_top(rows: u16) -> u16 {
    rows - footer_height(rows) + 1 + status_rows(rows) + hint_rows(rows)
}

fn write(text: &str) {
    let mut stdout = std::io::stdout();
    let _ = stdout.write_all(text.as_bytes());
    let _ = stdout.flush();
}

/// Session processing time. It accumulates only while the agent works and pauses
/// on `Ready`, `Select model`, and `Approval needed`, matching MinAgent.
#[derive(Default)]
pub struct ProcessingClock {
    state: RefCell<ClockState>,
}

#[derive(Default)]
struct ClockState {
    accumulated: Duration,
    started: Option<Instant>,
}

impl ProcessingClock {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start(&self) {
        let mut state = self.state.borrow_mut();
        if state.started.is_none() {
            state.started = Some(Instant::now());
        }
    }

    pub fn pause(&self) {
        let mut state = self.state.borrow_mut();
        if let Some(started) = state.started.take() {
            state.accumulated += started.elapsed();
        }
    }

    pub fn reset(&self) {
        let mut state = self.state.borrow_mut();
        state.accumulated = Duration::ZERO;
        state.started = None;
    }

    pub fn elapsed(&self) -> Duration {
        let state = self.state.borrow();
        state.accumulated + state.started.map(|started| started.elapsed()).unwrap_or_default()
    }
}

struct FooterState {
    terminal: bool,
    active: bool,
    initialized: bool,
    rows: u16,
    columns: u16,
    last_render: Instant,
    status: String,
    hints: String,
    input_prompt: String,
    input_line: String,
    input_cursor: usize,
    queued: usize,
    /// When non-empty, the footer renders a selection panel (status row + panel
    /// rows) instead of the hints/input area. Used by the interactive
    /// `/model` selector.
    panel: Vec<String>,
    region_bottom: u16,
}

pub struct Footer {
    state: RefCell<FooterState>,
}

impl Default for Footer {
    fn default() -> Self {
        Self::new()
    }
}

impl Footer {
    pub fn new() -> Self {
        let (rows, columns) = dimensions();
        let terminal = std::io::stdout().is_terminal() && std::io::stdin().is_terminal();
        Self {
            state: RefCell::new(FooterState {
                terminal,
                active: terminal && rows >= MIN_HEIGHT,
                initialized: false,
                rows,
                columns,
                last_render: Instant::now() - Duration::from_secs(1),
                status: String::new(),
                hints: String::new(),
                input_prompt: String::new(),
                input_line: String::new(),
                input_cursor: 0,
                queued: 0,
                panel: Vec::new(),
                region_bottom: 0,
            }),
        }
    }

    pub fn is_active(&self) -> bool {
        self.state.borrow().active
    }

    /// Clears the screen, reserves the transcript region, hides the cursor, and
    /// draws nothing yet.
    pub fn init(&self) {
        let mut state = self.state.borrow_mut();
        if !state.terminal {
            return;
        }
        let (rows, columns) = dimensions();
        state.rows = rows;
        state.columns = columns;
        state.active = rows >= MIN_HEIGHT;
        if !state.active {
            return;
        }
        write(&format!(
            "\x1b[2J\x1b[H\x1b[1;{}r{HIDE_CURSOR}",
            region_bottom(state.rows)
        ));
        state.region_bottom = region_bottom(state.rows);
        state.initialized = true;
    }

    pub fn render(&self, status: &str, hints: &str) {
        let mut state = self.state.borrow_mut();
        state.status = status.to_string();
        state.hints = hints.to_string();
        drop(state);
        self.redraw();
    }

    /// Whether a throttled render is due. Callers should check this before
    /// building the status string, so per-delta formatting is skipped.
    pub fn should_render(&self) -> bool {
        let state = self.state.borrow();
        state.active && state.last_render.elapsed() >= RENDER_INTERVAL
    }

    /// Draws the footer when interactive; prints a plain status line otherwise.
    pub fn note(&self, status: &str, hints: &str) {
        if self.is_active() {
            self.render(status, hints);
        } else {
            println!("{status}");
        }
    }

    /// Updates the editable line and the queued-message count in one redraw.
    pub fn set_input(&self, prompt: &str, line: &str, cursor: usize, queued: usize) {
        let mut state = self.state.borrow_mut();
        state.input_prompt = prompt.to_string();
        state.input_line = line.to_string();
        state.input_cursor = cursor;
        state.queued = queued;
        drop(state);
        self.redraw();
    }

    /// Renders an interactive selection panel (e.g. the `/model` list) in the
    /// footer area: a status row followed by the given pre-formatted panel
    /// lines. The normal hints/input area is hidden while the panel is up.
    pub fn render_selector(&self, status: &str, panel: &[String]) {
        let mut state = self.state.borrow_mut();
        state.status = status.to_string();
        state.panel = panel.to_vec();
        drop(state);
        self.redraw();
    }

    /// Clears the selection panel and restores the standard footer layout.
    pub fn clear_panel(&self) {
        let mut state = self.state.borrow_mut();
        let was_active = !state.panel.is_empty();
        state.panel.clear();
        drop(state);
        if was_active {
            self.redraw();
        }
    }

    fn redraw(&self) {
        let mut state = self.state.borrow_mut();
        if !state.active {
            return;
        }
        refresh_size(&mut state);
        if !state.active {
            return;
        }
        let rows = state.rows;
        let columns = state.columns.max(1) as usize;

        // Selection panel mode: status row + panel rows, no hints/input.
        if !state.panel.is_empty() {
            let panel_rows = state.panel.len() as u16;
            let footer_h = 1 + panel_rows;
            let bottom = rows.saturating_sub(footer_h);
            let top = bottom + 1;
            let status = truncate_styled(&state.status, columns);
            let mut output = String::from(SAVE_CURSOR);
            if bottom != state.region_bottom {
                output.push_str(&format!("\x1b[1;{bottom}r"));
                state.region_bottom = bottom;
            }
            output.push_str(&format!("\x1b[{top};1H\x1b[0m\x1b[J"));
            output.push_str(&format!("\x1b[{top};1H\x1b[0m\x1b[2K{status}"));
            for (index, line) in state.panel.iter().enumerate() {
                let row_number = top + 1 + index as u16;
                output.push_str(&format!("\x1b[{row_number};1H\x1b[0m\x1b[2K{line}"));
            }
            output.push_str(RESTORE_CURSOR);
            write(&output);
            state.last_render = Instant::now();
            return;
        }

        // Standard layout: restore the scroll region if we were in panel mode.
        let std_bottom = region_bottom(rows);
        if state.region_bottom != 0 && state.region_bottom != std_bottom {
            let mut output = String::from(SAVE_CURSOR);
            output.push_str(&format!("\x1b[1;{std_bottom}r"));
            let top = std_bottom + 1;
            output.push_str(&format!("\x1b[{top};1H\x1b[0m\x1b[J"));
            output.push_str(RESTORE_CURSOR);
            write(&output);
            state.region_bottom = std_bottom;
        }

        let top = rows - footer_height(rows) + 1;
        let mut output = String::from(SAVE_CURSOR);
        output.push_str(&format!("\x1b[{top};1H\x1b[0m\x1b[J"));

        let status_row = status_row(rows);
        if status_row > 0 {
            let status = with_queued(&state.status, state.queued);
            let status = truncate_styled(&status, columns);
            output.push_str(&format!("\x1b[{status_row};1H\x1b[0m\x1b[2K{status}"));
        }

        let hints_row = hints_row(rows);
        if hints_row > 0 {
            let hints = truncate_styled(&state.hints, columns);
            output.push_str(&format!("\x1b[{hints_row};1H\x1b[0m\x1b[2K{hints}"));
        }

        let top = input_top(rows);
        let height = input_rows(rows) as usize;
        let layout = layout_input(
            &state.input_prompt,
            &state.input_line,
            state.input_cursor,
            columns,
        );
        let (caret_row, caret_column) = layout.cursor;
        let first = caret_row.saturating_sub(height.saturating_sub(1));
        for index in 0..height {
            let visual = first + index;
            let row_number = top + index as u16;
            output.push_str(&format!("\x1b[{row_number};1H\x1b[0m\x1b[2K"));
            if let Some(row) = layout.rows.get(visual) {
                let caret = if visual == caret_row {
                    Some(caret_column)
                } else {
                    None
                };
                output.push_str(&render_input_row(
                    row,
                    visual == 0,
                    &state.input_prompt,
                    &layout.plain_prompt,
                    caret,
                ));
            }
        }
        output.push_str(RESTORE_CURSOR);
        write(&output);
        state.last_render = Instant::now();
    }

    /// Clears the footer rows, restores the full scrolling region, shows the
    /// cursor, and leaves the cursor where the transcript ended.
    pub fn finish(&self) {
        let mut state = self.state.borrow_mut();
        if !state.initialized {
            return;
        }
        let mut output = String::new();
        if state.active {
            let rows = state.rows;
            let top = rows - footer_height(rows) + 1;
            output.push_str(SAVE_CURSOR);
            output.push_str(&format!("\x1b[{top};1H\x1b[0m\x1b[J"));
        }
        output.push_str("\x1b[r\x1b[0m");
        output.push_str(SHOW_CURSOR);
        if state.active {
            output.push_str(RESTORE_CURSOR);
        }
        write(&output);
        state.active = false;
        state.initialized = false;
    }
}

impl Drop for Footer {
    fn drop(&mut self) {
        self.finish();
    }
}

fn refresh_size(state: &mut FooterState) {
    let (rows, columns) = dimensions();
    if rows == state.rows && columns == state.columns {
        return;
    }
    state.rows = rows;
    state.columns = columns;
    state.active = state.terminal && rows >= MIN_HEIGHT;
    if !state.active {
        if state.initialized {
            write(&format!("\x1b[r{SHOW_CURSOR}"));
        }
        return;
    }
    if !state.initialized {
        return;
    }
    let bottom = region_bottom(rows);
    write(&format!("\x1b[1;{bottom}r{HIDE_CURSOR}"));
    state.region_bottom = bottom;
}

fn with_queued(status: &str, queued: usize) -> String {
    if queued == 0 {
        status.to_string()
    } else {
        format!(
            "{status}{}{}",
            ui::sep(),
            ui::warning(&format!("{queued} queued"))
        )
    }
}

/// Visual layout of the prompt and editable line.
struct InputLayout {
    rows: Vec<String>,
    cursor: (usize, usize),
    plain_prompt: String,
}

fn layout_input(prompt: &str, line: &str, cursor: usize, columns: usize) -> InputLayout {
    let columns = columns.max(1);
    let plain_prompt = strip_ansi(prompt);
    let mut rows = vec![String::new()];
    let mut row = 0usize;
    let mut column = 0usize;
    advance_text(
        &plain_prompt,
        columns,
        &mut rows,
        &mut row,
        &mut column,
    );
    advance_text(line, columns, &mut rows, &mut row, &mut column);

    // The cursor position is the layout of the text up to the cursor.
    let mut cursor_rows = vec![String::new()];
    let mut cursor_row = 0usize;
    let mut cursor_column = 0usize;
    advance_text(
        &plain_prompt,
        columns,
        &mut cursor_rows,
        &mut cursor_row,
        &mut cursor_column,
    );
    let cursor = cursor.min(line.len());
    let prefix = &line[..cursor];
    advance_text(
        prefix,
        columns,
        &mut cursor_rows,
        &mut cursor_row,
        &mut cursor_column,
    );
    if cursor_column == columns {
        cursor_row += 1;
        cursor_column = 0;
    }
    while rows.len() <= cursor_row {
        rows.push(String::new());
    }
    InputLayout {
        rows,
        cursor: (cursor_row, cursor_column),
        plain_prompt,
    }
}

fn advance_text(
    text: &str,
    columns: usize,
    rows: &mut Vec<String>,
    row: &mut usize,
    column: &mut usize,
) {
    for character in text.chars() {
        let width = char_width(character);
        if character != '\n' && width > 0 && *column + width > columns {
            rows.push(String::new());
            *row += 1;
            *column = 0;
        }
        if character == '\n' {
            rows.push(String::new());
            *row += 1;
            *column = 0;
        } else {
            rows[*row].push(character);
            *column += width;
        }
    }
}

fn char_width(character: char) -> usize {
    if character == '\n' || character == '\r' {
        return 0;
    }
    UnicodeWidthChar::width(character).unwrap_or(0)
}

fn strip_ansi(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut result = String::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == 0x1b && index + 1 < bytes.len() && bytes[index + 1] == b'[' {
            index += 2;
            while index < bytes.len() && !(0x40..=0x7e).contains(&bytes[index]) {
                index += 1;
            }
            if index < bytes.len() {
                index += 1;
            }
            continue;
        }
        let character = value[index..].chars().next().unwrap_or(' ');
        result.push(character);
        index += character.len_utf8();
    }
    result
}

fn render_input_row(
    row: &str,
    first: bool,
    styled_prompt: &str,
    plain_prompt: &str,
    caret: Option<usize>,
) -> String {
    let mut text = match caret {
        Some(column) => insert_caret(row, column),
        None => row.to_string(),
    };
    if first {
        if let Some(rest) = text.strip_prefix(plain_prompt) {
            text = format!("{styled_prompt}{rest}");
        }
    }
    text
}

fn insert_caret(row: &str, column: usize) -> String {
    let mut width = 0usize;
    let mut index = row.len();
    let mut character_len = 0usize;
    for (byte_index, character) in row.char_indices() {
        if width >= column {
            index = byte_index;
            character_len = character.len_utf8();
            break;
        }
        let character_width = char_width(character);
        if width + character_width > column {
            index = byte_index;
            character_len = character.len_utf8();
            break;
        }
        width += character_width;
    }
    let caret = if index < row.len() {
        let character = row[index..].chars().next().unwrap_or(' ');
        format!("\x1b[7m{character}\x1b[0m")
    } else {
        "\x1b[7m \x1b[0m".to_string()
    };
    let mut result = String::with_capacity(row.len() + caret.len());
    result.push_str(&row[..index]);
    result.push_str(&caret);
    if index < row.len() {
        result.push_str(&row[index + character_len..]);
    }
    result
}

/// Truncates visible text to `max_width` cells, preserving SGR sequences and
/// appending an ellipsis. The ANSI parser is intentionally small: it only needs
/// to skip CSI sequences emitted by this crate.
fn truncate_styled(value: &str, max_width: usize) -> String {
    if visible_width(value) <= max_width {
        return value.to_string();
    }
    if max_width == 0 {
        return String::new();
    }
    let budget = max_width - 1;
    let bytes = value.as_bytes();
    let mut result = String::new();
    let mut width = 0;
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == 0x1b && index + 1 < bytes.len() && bytes[index + 1] == b'[' {
            let start = index;
            index += 2;
            while index < bytes.len() && !(0x40..=0x7e).contains(&bytes[index]) {
                index += 1;
            }
            if index < bytes.len() {
                index += 1;
            }
            result.push_str(&value[start..index]);
            continue;
        }
        let character = value[index..].chars().next().unwrap_or(' ');
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if width + character_width > budget {
            break;
        }
        result.push(character);
        width += character_width;
        index += character.len_utf8();
    }
    result.push('…');
    if value.contains("\x1b[") {
        result.push_str("\x1b[0m");
    }
    result
}

fn visible_width(value: &str) -> usize {
    let bytes = value.as_bytes();
    let mut width = 0;
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == 0x1b && index + 1 < bytes.len() && bytes[index + 1] == b'[' {
            index += 2;
            while index < bytes.len() && !(0x40..=0x7e).contains(&bytes[index]) {
                index += 1;
            }
            if index < bytes.len() {
                index += 1;
            }
            continue;
        }
        let character = value[index..].chars().next().unwrap_or(' ');
        width += UnicodeWidthChar::width(character).unwrap_or(0);
        index += character.len_utf8();
    }
    width
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn footer_rows_match_the_plan() {
        assert_eq!(footer_height(24), 5);
        assert_eq!(region_bottom(24), 19);
        assert_eq!(status_row(24), 20);
        assert_eq!(hints_row(24), 21);
        assert_eq!(input_top(24), 22);
        assert_eq!(input_rows(24), 3);

        assert_eq!(footer_height(10), 2);
        assert_eq!(region_bottom(10), 8);
        assert_eq!(status_row(10), 9);
        assert_eq!(hints_row(10), 0);
        assert_eq!(input_top(10), 10);

        assert_eq!(footer_height(4), 1);
        assert_eq!(region_bottom(4), 3);
        assert_eq!(status_row(4), 0);
        assert_eq!(input_top(4), 4);

        assert_eq!(footer_height(2), 0);
    }

    #[test]
    fn layout_wraps_and_tracks_the_cursor() {
        let layout = layout_input("Build › ", "hello", 5, 20);
        assert_eq!(layout.rows[0], "Build › hello");
        assert_eq!(layout.cursor, (0, 13));

        let layout = layout_input("P › ", "abcdef", 6, 5);
        assert_eq!(layout.rows[0], "P › a");
        assert_eq!(layout.rows[1], "bcdef");
        // Cursor at the end of a full row wraps to the next row, matching MinAgent.
        assert_eq!(layout.cursor, (2, 0));

        let layout = layout_input("P › ", "", 0, 20);
        assert_eq!(layout.rows[0], "P › ");
        assert_eq!(layout.cursor, (0, 4));
    }

    #[test]
    fn layout_keeps_multiline_input() {
        let layout = layout_input("P › ", "one\ntwo\nthree", 13, 40);
        assert_eq!(layout.rows, vec!["P › one", "two", "three"]);
        assert_eq!(layout.cursor, (2, 5));
    }

    #[test]
    fn caret_is_reverse_video_at_the_cursor() {
        let rendered = insert_caret("abc", 1);
        assert_eq!(rendered, "a\x1b[7mb\x1b[0mc");
        let rendered = insert_caret("abc", 3);
        assert_eq!(rendered, "abc\x1b[7m \x1b[0m");
        let rendered = insert_caret("", 0);
        assert_eq!(rendered, "\x1b[7m \x1b[0m");
    }

    #[test]
    fn input_rows_stay_inside_the_footer() {
        let layout = layout_input("P › ", "1\n2\n3\n4\n5", 9, 40);
        assert_eq!(layout.rows.len(), 5);
        // The viewport follows the caret so the newest line stays visible.
        let height = input_rows(24) as usize;
        let first = layout
            .cursor
            .0
            .saturating_sub(height.saturating_sub(1));
        assert_eq!(first, 2);
        assert_eq!(first + height, 5);
    }

    #[test]
    fn truncation_counts_cells_and_keeps_ansi() {
        let plain = truncate_styled("abcdefghij", 5);
        assert_eq!(plain, "abcd…");
        let styled = truncate_styled("\x1b[38;2;1;2;3mabcdefghij\x1b[0m", 5);
        assert!(styled.starts_with("\x1b[38;2;1;2;3m"));
        assert!(styled.ends_with("\x1b[0m"));
        assert!(visible_width(&styled) <= 5);
        assert_eq!(truncate_styled("short", 10), "short");
    }

    #[test]
    fn clock_pauses_and_accumulates() {
        let clock = ProcessingClock::new();
        clock.start();
        std::thread::sleep(Duration::from_millis(20));
        clock.pause();
        let paused = clock.elapsed();
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(clock.elapsed(), paused);
        clock.start();
        std::thread::sleep(Duration::from_millis(10));
        clock.pause();
        assert!(clock.elapsed() > paused);
        clock.reset();
        assert_eq!(clock.elapsed(), Duration::ZERO);
    }
}
