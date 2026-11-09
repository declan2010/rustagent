//! Prompt input and the persistent bottom line editor.
//!
//! Raw-input mode is enabled for the whole session by the caller, so when
//! interactive the [`LineEditor`] handles every key itself and draws through the
//! [`Footer`]. Tab toggles Build/Plan, Ctrl+T cycles the think level, and Enter
//! submits. While the model works, submissions are queued instead of being lost;
//! Tab/Ctrl+T update a separate [`Steering`] snapshot that is applied after the
//! turn.
//!
//! Multi-line clipboard text is kept inside the prompt: bracketed paste
//! (`ESC[200~` … `ESC[201~`) and bulk input chunks insert line breaks instead of
//! submitting the prompt line by line.
//!
//! When not interactive (piped), [`read_prompt_line`] keeps the cooked fallback.

use std::collections::VecDeque;
use std::io::{IsTerminal, Write};

use crate::agent::{Mode, next_think_level};
use crate::footer::Footer;
use crate::input::{self, Input};
use crate::ui;

/// Which prompt-level key the user pressed (cooked fallback callback).
pub enum PromptAction {
    ToggleMode,
    CycleThink,
}

/// Result of a prompt-level key action: an optional new prompt label (Tab) and an
/// optional styled notice line (Ctrl+T).
pub struct PromptActionResult {
    pub prompt: Option<String>,
    pub notice: Option<String>,
}

/// One interpreted key sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditorEvent {
    None,
    Submit(String),
    Abort,
    Cancel,
    Exit,
    ToggleMode,
    CycleThink,
}

impl EditorEvent {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// A prompt captured while the model was working.
#[derive(Debug, Clone)]
pub struct QueuedPrompt {
    pub text: String,
    pub mode: Mode,
    pub think: Option<String>,
}

/// Pending Tab/Ctrl+T changes made during a turn.
#[derive(Debug, Default, Clone)]
pub struct Steering {
    pub mode: Option<Mode>,
    /// `Some(None)` means "set thinking off/unset"; `None` means no change.
    pub think: Option<Option<String>>,
}

const PASTE_END: &[u8] = b"\x1b[201~";

/// Stateful raw-mode line editor. The buffer is UTF-8 and the cursor is a byte
/// offset at a character boundary.
pub struct LineEditor {
    pub prompt: String,
    pub buffer: String,
    pub cursor: usize,
    utf8: Vec<u8>,
    escape: Vec<u8>,
    escape_active: bool,
    paste: bool,
    paste_tail: Vec<u8>,
    paste_skip_lf: bool,
}

impl Default for LineEditor {
    fn default() -> Self {
        Self::new()
    }
}

impl LineEditor {
    pub fn new() -> Self {
        Self {
            prompt: String::new(),
            buffer: String::new(),
            cursor: 0,
            utf8: Vec::new(),
            escape: Vec::new(),
            escape_active: false,
            paste: false,
            paste_tail: Vec::new(),
            paste_skip_lf: false,
        }
    }

    pub fn set_prompt(&mut self, prompt: String) {
        self.prompt = prompt;
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.trim().is_empty()
    }

    pub fn take_line(&mut self) -> String {
        self.cursor = 0;
        self.utf8.clear();
        std::mem::take(&mut self.buffer)
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.cursor = 0;
        self.utf8.clear();
    }

    /// Feeds one raw byte. `bulk` and `more` describe the input chunk so a
    /// pasted burst can keep line breaks and shortcut keys literal.
    pub fn feed(&mut self, byte: u8, bulk: bool, more: bool) -> EditorEvent {
        if self.paste {
            return self.feed_paste(byte);
        }
        if self.escape_active {
            return self.feed_escape(byte);
        }
        if byte == 0x1b {
            if !more && !bulk {
                return EditorEvent::Abort;
            }
            self.escape_active = true;
            self.escape.clear();
            self.escape.push(byte);
            return EditorEvent::None;
        }

        let literal = bulk && more;
        match byte {
            b'\n' | b'\r' => {
                if literal {
                    self.insert_char('\n');
                    EditorEvent::None
                } else {
                    EditorEvent::Submit(self.take_line())
                }
            }
            0x09 => {
                if literal {
                    self.insert_str("    ");
                    EditorEvent::None
                } else {
                    EditorEvent::ToggleMode
                }
            }
            0x14 => {
                if literal {
                    EditorEvent::None
                } else {
                    EditorEvent::CycleThink
                }
            }
            0x7f | 0x08 => {
                if !literal {
                    self.backspace();
                }
                EditorEvent::None
            }
            0x03 => {
                if literal {
                    EditorEvent::None
                } else {
                    EditorEvent::Cancel
                }
            }
            0x04 => {
                if literal {
                    EditorEvent::None
                } else if self.buffer.is_empty() {
                    EditorEvent::Exit
                } else {
                    self.delete();
                    EditorEvent::None
                }
            }
            0x01 => {
                self.move_home();
                EditorEvent::None
            }
            0x05 => {
                self.move_end();
                EditorEvent::None
            }
            0x0b => {
                self.delete_to_end();
                EditorEvent::None
            }
            0x15 => {
                self.clear();
                EditorEvent::None
            }
            0x17 => {
                self.delete_word();
                EditorEvent::None
            }
            0x00..=0x1f => EditorEvent::None,
            _ => {
                if byte < 0x80 {
                    self.insert_char(byte as char);
                } else {
                    self.utf8.push(byte);
                    self.flush_utf8(false);
                }
                EditorEvent::None
            }
        }
    }

    fn feed_paste(&mut self, byte: u8) -> EditorEvent {
        self.paste_tail.push(byte);
        if self.paste_tail.ends_with(PASTE_END) {
            let content_len = self.paste_tail.len() - PASTE_END.len();
            let content: Vec<u8> = self.paste_tail.drain(..content_len).collect();
            self.paste_tail.clear();
            self.paste = false;
            self.insert_paste_bytes(&content);
            return EditorEvent::None;
        }
        let keep = PASTE_END.len() - 1;
        if self.paste_tail.len() > keep {
            let flush_len = self.paste_tail.len() - keep;
            let content: Vec<u8> = self.paste_tail.drain(..flush_len).collect();
            self.insert_paste_bytes(&content);
        }
        EditorEvent::None
    }

    fn feed_escape(&mut self, byte: u8) -> EditorEvent {
        self.escape.push(byte);
        match self.escape.get(1).copied() {
            Some(b'[') => {
                if self.escape.len() >= 3 && (0x40..=0x7e).contains(&byte) {
                    let sequence = std::mem::take(&mut self.escape);
                    self.escape_active = false;
                    return self.handle_escape(&sequence);
                }
            }
            Some(b'O') => {
                if self.escape.len() >= 3 {
                    let sequence = std::mem::take(&mut self.escape);
                    self.escape_active = false;
                    return self.handle_escape(&sequence);
                }
            }
            Some(b']') => {
                if byte == 0x07 {
                    self.escape.clear();
                    self.escape_active = false;
                }
            }
            _ => {
                self.escape.clear();
                self.escape_active = false;
            }
        }
        EditorEvent::None
    }

    fn handle_escape(&mut self, sequence: &[u8]) -> EditorEvent {
        if sequence == b"\x1b[D" || sequence == b"\x1bOD" {
            self.move_left();
        } else if sequence == b"\x1b[C" || sequence == b"\x1bOC" {
            self.move_right();
        } else if sequence == b"\x1b[H"
            || sequence == b"\x1bOH"
            || sequence == b"\x1b[1~"
            || sequence == b"\x1b[7~"
        {
            self.move_home();
        } else if sequence == b"\x1b[F"
            || sequence == b"\x1bOF"
            || sequence == b"\x1b[4~"
            || sequence == b"\x1b[8~"
        {
            self.move_end();
        } else if sequence == b"\x1b[3~" {
            self.delete();
        } else if sequence == b"\x1b[200~" {
            self.paste = true;
            self.paste_tail.clear();
            self.paste_skip_lf = false;
        } else if sequence == b"\x1b[201~" {
            // End marker without a tracked start: ignore.
        } else if let Some(last) = sequence.last() {
            // Modified arrows (e.g. ESC[1;5D) keep the final movement byte.
            match last {
                b'D' => self.move_left(),
                b'C' => self.move_right(),
                b'H' => self.move_home(),
                b'F' => self.move_end(),
                _ => {}
            }
        }
        EditorEvent::None
    }

    fn insert_paste_bytes(&mut self, bytes: &[u8]) {
        self.utf8.extend_from_slice(bytes);
        self.flush_utf8(true);
    }

    fn flush_utf8(&mut self, paste: bool) {
        loop {
            match std::str::from_utf8(&self.utf8) {
                Ok(text) => {
                    let text = text.to_string();
                    self.utf8.clear();
                    self.insert_processed(&text, paste);
                    return;
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    if valid > 0 {
                        let text = String::from_utf8_lossy(&self.utf8[..valid]).into_owned();
                        self.utf8.drain(..valid);
                        self.insert_processed(&text, paste);
                        continue;
                    }
                    if let Some(length) = error.error_len() {
                        self.utf8.drain(..length);
                        continue;
                    }
                    return;
                }
            }
        }
    }

    fn insert_processed(&mut self, text: &str, paste: bool) {
        if !paste {
            for character in text.chars() {
                if !character.is_control() {
                    self.insert_char(character);
                }
            }
            return;
        }
        for character in text.chars() {
            if self.paste_skip_lf {
                self.paste_skip_lf = false;
                if character == '\n' {
                    continue;
                }
            }
            match character {
                '\r' => {
                    self.insert_char('\n');
                    self.paste_skip_lf = true;
                }
                '\n' => self.insert_char('\n'),
                '\t' => self.insert_str("    "),
                value if value.is_control() => {}
                value => self.insert_char(value),
            }
        }
    }

    fn insert_char(&mut self, character: char) {
        self.buffer.insert(self.cursor, character);
        self.cursor += character.len_utf8();
    }

    fn insert_str(&mut self, text: &str) {
        self.buffer.insert_str(self.cursor, text);
        self.cursor += text.len();
    }

    fn backspace(&mut self) {
        let previous = previous_boundary(&self.buffer, self.cursor);
        if previous < self.cursor {
            self.buffer.replace_range(previous..self.cursor, "");
            self.cursor = previous;
        }
    }

    fn delete(&mut self) {
        let next = next_boundary(&self.buffer, self.cursor);
        if next > self.cursor {
            self.buffer.replace_range(self.cursor..next, "");
        }
    }

    fn delete_to_end(&mut self) {
        self.buffer.truncate(self.cursor);
    }

    fn delete_word(&mut self) {
        let mut start = self.cursor;
        while start > 0 {
            let previous = previous_boundary(&self.buffer, start);
            let character = self.buffer[previous..start].chars().next().unwrap_or(' ');
            if !character.is_whitespace() {
                break;
            }
            start = previous;
        }
        while start > 0 {
            let previous = previous_boundary(&self.buffer, start);
            let character = self.buffer[previous..start].chars().next().unwrap_or(' ');
            if character.is_whitespace() {
                break;
            }
            start = previous;
        }
        self.buffer.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    fn move_left(&mut self) {
        self.cursor = previous_boundary(&self.buffer, self.cursor);
    }

    fn move_right(&mut self) {
        self.cursor = next_boundary(&self.buffer, self.cursor);
    }

    fn move_home(&mut self) {
        self.cursor = 0;
    }

    fn move_end(&mut self) {
        self.cursor = self.buffer.len();
    }
}

fn previous_boundary(value: &str, index: usize) -> usize {
    if index == 0 {
        return 0;
    }
    let mut result = index - 1;
    while result > 0 && !value.is_char_boundary(result) {
        result -= 1;
    }
    result
}

fn next_boundary(value: &str, index: usize) -> usize {
    if index >= value.len() {
        return value.len();
    }
    let mut result = index + 1;
    while result < value.len() && !value.is_char_boundary(result) {
        result += 1;
    }
    result
}

/// The interactive session's editor, queued prompts, and pending steering.
pub struct Session {
    editor: LineEditor,
    queue: VecDeque<QueuedPrompt>,
    steering: Steering,
    selected_mode: Mode,
    think: Option<String>,
    interactive: bool,
    exit_requested: bool,
}

impl Session {
    pub fn new(prompt: String, interactive: bool) -> Self {
        let mut editor = LineEditor::new();
        editor.set_prompt(prompt);
        Self {
            editor,
            queue: VecDeque::new(),
            steering: Steering::default(),
            selected_mode: Mode::Build,
            think: None,
            interactive,
            exit_requested: false,
        }
    }

    pub fn interactive(&self) -> bool {
        self.interactive
    }

    pub fn set_prompt(&mut self, prompt: String) {
        self.editor.set_prompt(prompt);
    }

    pub fn note_agent_state(&mut self, mode: Mode, think: Option<String>) {
        self.selected_mode = mode;
        self.think = think;
    }

    pub fn take_steering(&mut self) -> Steering {
        std::mem::take(&mut self.steering)
    }

    pub fn toggle_steering_mode(&mut self) -> Mode {
        let next = self.steering.mode.unwrap_or(self.selected_mode).toggle();
        self.steering.mode = Some(next);
        next
    }

    /// Cycles the queued-`think` override with `EditorEvent::CycleThink`.
    /// `allowed` limits the cycle to the states the running model supports;
    /// `None` falls back to the generic MinAgent cycle.
    pub fn cycle_steering_think(&mut self, allowed: Option<&[String]>) -> String {
        let current = self
            .steering
            .think
            .clone()
            .unwrap_or_else(|| self.think.clone());
        let next =
            next_think_level(current.as_deref(), allowed).unwrap_or_else(|| "false".to_string());
        self.steering.think = Some(Some(next.clone()));
        next
    }

    pub fn queue_prompt(&mut self, text: String) -> usize {
        let mode = self.steering.mode.unwrap_or(self.selected_mode);
        let think = self
            .steering
            .think
            .clone()
            .unwrap_or_else(|| self.think.clone());
        self.queue.push_back(QueuedPrompt { text, mode, think });
        self.queue.len()
    }

    /// Queues a submitted line and echoes it above the footer.
    pub fn queue_submission(&mut self, text: String) -> usize {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return self.queue.len();
        }
        let display = trimmed.to_string();
        let count = self.queue_prompt(text);
        ui::print_submitted(&display);
        ui::print_info(&ui::muted(&format!("Queued message · {count} waiting")));
        count
    }

    /// Queues a non-empty draft (used before switching to approval input).
    pub fn queue_draft(&mut self) -> Option<usize> {
        if self.editor.is_empty() {
            self.editor.clear();
            return None;
        }
        let text = self.editor.take_line();
        Some(self.queue_submission(text))
    }

    pub fn pop_queued(&mut self) -> Option<QueuedPrompt> {
        self.queue.pop_front()
    }

    pub fn clear_queue(&mut self) {
        self.queue.clear();
    }

    pub fn request_exit(&mut self) {
        self.exit_requested = true;
    }

    pub fn exit_requested(&self) -> bool {
        self.exit_requested
    }

    pub fn feed(&mut self, byte: u8, bulk: bool, more: bool) -> EditorEvent {
        self.editor.feed(byte, bulk, more)
    }

    pub fn render(&self, footer: &Footer) {
        footer.set_input(
            &self.editor.prompt,
            &self.editor.buffer,
            self.editor.cursor,
            self.queue.len(),
        );
    }

    /// Reads keys until an event. Interactive mode renders after each chunk.
    pub async fn next_event(&mut self, input: &mut Input, footer: &Footer) -> EditorEvent {
        if !self.interactive {
            let mut bytes = Vec::new();
            while let Some(byte) = input.read_byte().await {
                if byte == b'\n' || byte == b'\r' {
                    return EditorEvent::Submit(String::from_utf8_lossy(&bytes).into_owned());
                }
                bytes.push(byte);
            }
            return EditorEvent::Exit;
        }
        loop {
            let event = match input::read_event(input).await {
                Some(input::InputEvent::Byte(byte)) => {
                    let bulk = input.is_bulk();
                    let more = input.has_more_in_chunk();
                    let event = self.editor.feed(byte, bulk, more);
                    if !more {
                        self.render(footer);
                    }
                    event
                }
                Some(input::InputEvent::Resized) => {
                    // A terminal resize re-renders the whole footer (region,
                    // status, hints, input) at the new geometry.
                    footer.refresh();
                    EditorEvent::None
                }
                None => return EditorEvent::Exit,
            };
            if !event.is_none() {
                return event;
            }
        }
    }

    /// Switches the footer input to a `[y/N]` prompt for a terminal command.
    pub async fn read_approval(
        &mut self,
        input: &mut Input,
        footer: &Footer,
        command: &str,
    ) -> bool {
        self.read_permission(input, footer, "Allow this command? ", command)
            .await
    }

    /// Switches the footer input to a `[y/N]` prompt for any extension
    /// approval (terminal command, skill load) showing `question` + subject.
    pub async fn read_permission(
        &mut self,
        input: &mut Input,
        footer: &Footer,
        question: &str,
        subject: &str,
    ) -> bool {
        if !self.interactive {
            return read_permission_plain(input, question, subject).await;
        }
        self.queue_draft();
        let saved_prompt = self.editor.prompt.clone();
        self.editor.set_prompt("[y/N] ".to_string());
        self.editor.clear();
        let quoted = serde_json::to_string(subject).unwrap_or_else(|_| format!("{subject:?}"));
        println!("{question}{quoted}");
        self.render(footer);
        let approved = loop {
            match self.next_event(input, footer).await {
                EditorEvent::Submit(text) => {
                    break matches!(text.trim().to_lowercase().as_str(), "y" | "yes");
                }
                EditorEvent::Abort | EditorEvent::Cancel => break false,
                EditorEvent::Exit => {
                    self.request_exit();
                    break false;
                }
                _ => {}
            }
        };
        self.editor.set_prompt(saved_prompt);
        self.editor.clear();
        self.render(footer);
        approved
    }
}

async fn read_permission_plain(input: &mut Input, question: &str, subject: &str) -> bool {
    let mut stdout = std::io::stdout();
    let quoted = serde_json::to_string(subject).unwrap_or_else(|_| format!("{subject:?}"));
    let _ = write!(stdout, "{question}{quoted}\n[y/N] ");
    let _ = stdout.flush();
    let mut line = String::new();
    while let Some(value) = input.read_byte().await {
        match value {
            b'\n' | b'\r' => break,
            0x1b => {
                line.clear();
                let _ = write!(stdout, "\n");
                let _ = stdout.flush();
                return false;
            }
            0x7f | 0x08 => {
                if line.pop().is_some() {
                    let _ = write!(stdout, "\x08 \x08");
                    let _ = stdout.flush();
                }
            }
            0x20..=0x7e => {
                line.push(value as char);
                let _ = stdout.write_all(&[value]);
                let _ = stdout.flush();
            }
            _ => {}
        }
    }
    let _ = write!(stdout, "\n");
    let _ = stdout.flush();
    matches!(line.trim().to_lowercase().as_str(), "y" | "yes")
}

/// Reads one prompt line in the cooked (piped) fallback. Returns `None` on EOF,
/// Ctrl+C, or Ctrl+D on an empty line.
pub async fn read_prompt_line(
    input: &mut Input,
    mut prompt: String,
    mut on_action: impl FnMut(PromptAction) -> PromptActionResult,
) -> std::io::Result<Option<String>> {
    let mut stdout = std::io::stdout();
    let raw = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();

    write!(stdout, "{prompt}")?;
    stdout.flush()?;

    let mut buffer: Vec<u8> = Vec::new();
    let mut skipping_escape = false;
    let mut pending_escape = false;
    let mut csi: Vec<u8> = Vec::new();
    let mut paste_active = false;
    let mut paste_tail: Vec<u8> = Vec::new();
    let mut paste_skip_lf = false;
    loop {
        let Some(value) = input.read_byte().await else {
            if raw {
                write!(stdout, "\n")?;
                stdout.flush()?;
            }
            return Ok(None);
        };

        if paste_active {
            paste_tail.push(value);
            if paste_tail.ends_with(PASTE_END) {
                let content_len = paste_tail.len() - PASTE_END.len();
                let content: Vec<u8> = paste_tail.drain(..content_len).collect();
                write_paste(&mut buffer, &content, &mut stdout, &mut paste_skip_lf)?;
                paste_tail.clear();
                paste_active = false;
            } else {
                let keep = PASTE_END.len() - 1;
                if paste_tail.len() > keep {
                    let flush_len = paste_tail.len() - keep;
                    let flushed: Vec<u8> = paste_tail.drain(..flush_len).collect();
                    write_paste(&mut buffer, &flushed, &mut stdout, &mut paste_skip_lf)?;
                }
            }
            continue;
        }

        if !raw {
            if value == b'\n' || value == b'\r' {
                return Ok(Some(String::from_utf8_lossy(&buffer).into_owned()));
            }
            buffer.push(value);
            continue;
        }

        if skipping_escape {
            if (0x40..=0x7e).contains(&value) {
                skipping_escape = false;
                if csi == b"[200" && value == b'~' {
                    paste_active = true;
                    paste_tail.clear();
                    paste_skip_lf = false;
                }
                csi.clear();
            } else {
                csi.push(value);
            }
            continue;
        }
        if pending_escape {
            pending_escape = false;
            if value == b'[' || value == b'O' || value == b']' {
                skipping_escape = true;
                csi.clear();
                csi.push(value);
            } else {
                input.unread(&[value]);
            }
            continue;
        }

        let paste_literal = input.is_bulk() && input.has_more_in_chunk();

        if value == b'\n' || value == b'\r' {
            if paste_literal {
                write_paste(&mut buffer, &[value], &mut stdout, &mut paste_skip_lf)?;
                continue;
            }
            write!(stdout, "\n")?;
            stdout.flush()?;
            return Ok(Some(String::from_utf8_lossy(&buffer).into_owned()));
        }
        match value {
            0x1b => {
                pending_escape = true;
            }
            0x14 => {
                if paste_literal {
                    write_paste(&mut buffer, &[value], &mut stdout, &mut paste_skip_lf)?;
                    continue;
                }
                let result = on_action(PromptAction::CycleThink);
                write!(stdout, "\r\x1b[2K")?;
                if let Some(notice) = result.notice {
                    write!(stdout, "{notice}\n")?;
                }
                if let Some(new_prompt) = result.prompt {
                    prompt = new_prompt;
                }
                write!(stdout, "{prompt}{}", String::from_utf8_lossy(&buffer))?;
                stdout.flush()?;
            }
            0x09 => {
                if paste_literal {
                    write_paste(&mut buffer, &[value], &mut stdout, &mut paste_skip_lf)?;
                    continue;
                }
                let result = on_action(PromptAction::ToggleMode);
                write!(stdout, "\r\x1b[2K")?;
                if let Some(notice) = result.notice {
                    write!(stdout, "{notice}\n")?;
                }
                if let Some(new_prompt) = result.prompt {
                    prompt = new_prompt;
                }
                write!(stdout, "{prompt}{}", String::from_utf8_lossy(&buffer))?;
                stdout.flush()?;
            }
            0x7f | 0x08 => {
                if paste_literal {
                    write_paste(&mut buffer, &[value], &mut stdout, &mut paste_skip_lf)?;
                    continue;
                }
                if pop_utf8(&mut buffer) {
                    write!(stdout, "\x08 \x08")?;
                    stdout.flush()?;
                }
            }
            0x03 => {
                if paste_literal {
                    write_paste(&mut buffer, &[value], &mut stdout, &mut paste_skip_lf)?;
                    continue;
                }
                write!(stdout, "^C\n")?;
                stdout.flush()?;
                return Ok(None);
            }
            0x04 => {
                if paste_literal {
                    write_paste(&mut buffer, &[value], &mut stdout, &mut paste_skip_lf)?;
                    continue;
                }
                if buffer.is_empty() {
                    return Ok(None);
                }
            }
            0x00..=0x1f => {}
            _ => {
                buffer.push(value);
                stdout.write_all(&[value])?;
                stdout.flush()?;
            }
        }
    }
}

/// Inserts pasted bytes into the prompt buffer, normalizing CR/CRLF to a single
/// newline and ignoring terminal control bytes.
fn write_paste(
    buffer: &mut Vec<u8>,
    bytes: &[u8],
    stdout: &mut impl Write,
    skip_lf: &mut bool,
) -> std::io::Result<()> {
    for &byte in bytes {
        if *skip_lf && byte == b'\n' {
            *skip_lf = false;
            continue;
        }
        *skip_lf = false;
        match byte {
            b'\r' => {
                buffer.push(b'\n');
                stdout.write_all(b"\n")?;
                *skip_lf = true;
            }
            b'\n' => {
                buffer.push(b'\n');
                stdout.write_all(b"\n")?;
            }
            0x09 => {
                buffer.push(b'\t');
                stdout.write_all(b"\t")?;
            }
            0x00..=0x1f | 0x7f => {}
            _ => {
                buffer.push(byte);
                stdout.write_all(&[byte])?;
            }
        }
    }
    stdout.flush()
}

fn pop_utf8(buffer: &mut Vec<u8>) -> bool {
    if buffer.is_empty() {
        return false;
    }
    let mut index = buffer.len() - 1;
    while index > 0 && (buffer[index] & 0xC0) == 0x80 {
        index -= 1;
    }
    buffer.truncate(index);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn type_text(editor: &mut LineEditor, text: &str) {
        for byte in text.bytes() {
            editor.feed(byte, false, false);
        }
    }

    #[test]
    fn typing_and_submitting_returns_the_line() {
        let mut editor = LineEditor::new();
        type_text(&mut editor, "hello");
        assert_eq!(editor.buffer, "hello");
        assert_eq!(editor.cursor, 5);
        assert_eq!(
            editor.feed(b'\r', false, false),
            EditorEvent::Submit("hello".into())
        );
        assert_eq!(editor.buffer, "");
        assert_eq!(editor.cursor, 0);
    }

    #[test]
    fn arrows_and_backspace_edit_at_the_cursor() {
        let mut editor = LineEditor::new();
        type_text(&mut editor, "abc");
        editor.feed(0x1b, false, true);
        editor.feed(b'[', false, true);
        editor.feed(b'D', false, false);
        assert_eq!(editor.cursor, 2);
        editor.feed(0x7f, false, false);
        assert_eq!(editor.buffer, "ac");
        assert_eq!(editor.cursor, 1);
        editor.feed(0x1b, false, true);
        editor.feed(b'[', false, true);
        editor.feed(b'C', false, false);
        assert_eq!(editor.cursor, 2);
    }

    #[test]
    fn bracketed_paste_keeps_newlines_and_does_not_submit() {
        let mut editor = LineEditor::new();
        let paste = b"\x1b[200~one\ntwo\nthree\x1b[201~";
        for (index, byte) in paste.iter().enumerate() {
            let more = index + 1 < paste.len();
            let event = editor.feed(*byte, false, more);
            assert_eq!(event, EditorEvent::None);
        }
        assert_eq!(editor.buffer, "one\ntwo\nthree");
        assert_eq!(
            editor.feed(b'\r', false, false),
            EditorEvent::Submit("one\ntwo\nthree".into())
        );
    }

    #[test]
    fn bulk_paste_fallback_keeps_inner_newlines() {
        let mut editor = LineEditor::new();
        let text = b"a\nb\nc\n";
        for (index, byte) in text.iter().enumerate() {
            let more = index + 1 < text.len();
            let event = editor.feed(*byte, true, more);
            if index + 1 == text.len() {
                assert_eq!(event, EditorEvent::Submit("a\nb\nc".into()));
            } else {
                assert_eq!(event, EditorEvent::None);
            }
        }
    }

    #[test]
    fn tab_and_ctrl_t_are_shortcuts_only_outside_paste() {
        let mut editor = LineEditor::new();
        assert_eq!(editor.feed(0x09, false, false), EditorEvent::ToggleMode);
        assert_eq!(editor.feed(0x14, false, false), EditorEvent::CycleThink);
        assert_eq!(editor.feed(0x09, true, true), EditorEvent::None);
        assert_eq!(editor.buffer, "    ");
        assert_eq!(editor.feed(0x14, true, true), EditorEvent::None);
    }

    #[test]
    fn ctrl_u_clears_and_ctrl_w_deletes_a_word() {
        let mut editor = LineEditor::new();
        type_text(&mut editor, "one two three");
        editor.feed(0x17, false, false);
        assert_eq!(editor.buffer, "one two ");
        editor.feed(0x15, false, false);
        assert_eq!(editor.buffer, "");
        assert_eq!(editor.cursor, 0);
    }

    #[test]
    fn utf8_is_inserted_and_deleted_by_character() {
        let mut editor = LineEditor::new();
        type_text(&mut editor, "aé");
        assert_eq!(editor.buffer, "aé");
        editor.feed(0x7f, false, false);
        assert_eq!(editor.buffer, "a");
        editor.feed(0x1b, false, true);
        editor.feed(b'[', false, true);
        editor.feed(b'3', false, true);
        editor.feed(b'~', false, false);
        assert_eq!(editor.buffer, "a");
    }

    #[test]
    fn session_queue_captures_steering() {
        let mut session = Session::new("Build › ".into(), true);
        session.note_agent_state(Mode::Build, None);
        let mode = session.toggle_steering_mode();
        assert_eq!(mode, Mode::Plan);
        let think = session.cycle_steering_think(None);
        assert_eq!(think, "false");
        session.queue_submission("hello".into());
        let queued = session.pop_queued().unwrap();
        assert_eq!(queued.text, "hello");
        assert_eq!(queued.mode, Mode::Plan);
        assert_eq!(queued.think.as_deref(), Some("false"));
    }
}
