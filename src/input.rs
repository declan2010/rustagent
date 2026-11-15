//! Session input: one background stdin reader feeding a small buffer.
//!
//! The terminal is switched to "raw input" mode for the whole session: ICANON
//! and ECHO are off (bytes arrive per keystroke, we echo ourselves) but OPOST
//! stays on so `println!`/`\n` still render normally, and ISIG stays on so
//! Ctrl+C still interrupts the process. Esc (0x1b) is delivered as a byte,
//! which the agent uses to abort the current turn, matching MinAgent.
//!
//! Reads are grouped into chunks so the prompt can tell a pasted burst from a
//! single keypress: a chunk with more than one byte (and not an escape
//! sequence) is treated as bulk input, so pasted line breaks are inserted
//! instead of submitting the prompt.

use std::collections::VecDeque;
use std::io::IsTerminal;
use std::time::Duration;

use terminal_size::{Height, Width, terminal_size};
use tokio::sync::mpsc;

/// One item on the input stream: stdin bytes (with their paste
/// classification) or a notice that the terminal size changed.
enum StreamItem {
    Bytes { bytes: Vec<u8>, bulk: bool },
    Resized,
}

/// One input item for event-driven consumers: a keystroke byte or a notice
/// that the terminal was resized.
pub enum InputEvent {
    Byte(u8),
    Resized,
}

/// How often the resize watcher polls the terminal size.
const RESIZE_POLL_INTERVAL: Duration = Duration::from_millis(250);

pub struct Input {
    rx: mpsc::UnboundedReceiver<StreamItem>,
    pending: VecDeque<u8>,
    current: VecDeque<u8>,
    current_bulk: bool,
}

impl Input {
    /// Spawns a background reader that pumps stdin bytes into a channel. Uses an
    /// epoll-integrated `AsyncFd` so the task cancels cleanly on shutdown instead
    /// of leaving a blocking read thread behind.
    pub fn spawn() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel::<StreamItem>();
        // Reports terminal-size changes so the UI re-renders at the new
        // geometry; MinAgent leaves resizes to readline, but here the notice
        // rides the same input stream the editor already selects on.
        spawn_resize_watcher(sender.clone());
        #[cfg(unix)]
        let reader = async_fd_stdin();
        #[cfg(unix)]
        if let Some(async_fd) = reader {
            tokio::spawn(async move {
                use std::io::Read;
                let mut buffer = [0u8; 4096];
                loop {
                    let mut guard = match async_fd.readable().await {
                        Ok(guard) => guard,
                        Err(_) => break,
                    };
                    match guard.try_io(|io| io.get_ref().read(&mut buffer)) {
                        Ok(Ok(0)) => break,
                        Ok(Ok(count)) => {
                            let bytes = buffer[..count].to_vec();
                            // An escape-prefixed chunk is a key sequence, not a paste.
                            let bulk = count > 1 && bytes.first() != Some(&0x1b);
                            if sender.send(StreamItem::Bytes { bytes, bulk }).is_err() {
                                return;
                            }
                        }
                        Ok(Err(ref error)) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            continue;
                        }
                        Ok(Err(_)) => break,
                        Err(_) => continue,
                    }
                }
            });
        } else {
            // Fallback (very unusual): blocking stdin via tokio's async stdin.
            spawn_blocking_stdin(sender);
        }
        #[cfg(not(unix))]
        spawn_blocking_stdin(sender);

        Self {
            rx: receiver,
            pending: VecDeque::new(),
            current: VecDeque::new(),
            current_bulk: false,
        }
    }

    /// Next byte: any pushed-back bytes first, then the stream. Resize
    /// notices are skipped here; [`read_event`] surfaces them instead.
    pub async fn read_byte(&mut self) -> Option<u8> {
        if let Some(byte) = self.pending.pop_front() {
            return Some(byte);
        }
        loop {
            if let Some(byte) = self.current.pop_front() {
                return Some(byte);
            }
            match self.rx.recv().await? {
                StreamItem::Bytes { bytes, bulk } => {
                    self.current = bytes.into();
                    self.current_bulk = bulk;
                }
                StreamItem::Resized => continue,
            }
        }
    }

    /// Whether the byte most recently read belongs to a bulk (pasted) chunk.
    /// Pushback bytes are always treated as normal input.
    pub fn is_bulk(&self) -> bool {
        self.pending.is_empty() && self.current_bulk
    }

    /// Whether the current chunk has more bytes after the one just read. A
    /// newline followed by more bytes is inside a paste; a trailing newline is
    /// treated as a normal submit. Bytes returned by [`Input::unread`] also
    /// count, so a swallow-and-replay escape sequence is still recognized as
    /// one keypress instead of a lone Esc.
    pub fn has_more_in_chunk(&self) -> bool {
        !self.pending.is_empty() || !self.current.is_empty()
    }

    /// Returns bytes to the front of the stream so they are not lost.
    pub fn unread(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.pending.push_back(*byte);
        }
    }
}

/// Reads one byte together with the chunk flags that classify it. Returning the
/// flags in the same future lets callers feed the editor without holding a
/// mutable borrow of [`Input`] across the handler.
pub async fn read_chunk(input: &mut Input) -> Option<(u8, bool, bool)> {
    let byte = input.read_byte().await?;
    Some((byte, input.is_bulk(), input.has_more_in_chunk()))
}

/// Next byte or resize notice: unlike [`read_chunk`], a resize wakes the
/// waiter instead of waiting behind the next keystroke, so an idle prompt
/// re-renders immediately after a terminal resize.
pub async fn read_event(input: &mut Input) -> Option<InputEvent> {
    if let Some(byte) = input.pending.pop_front() {
        return Some(InputEvent::Byte(byte));
    }
    loop {
        if let Some(byte) = input.current.pop_front() {
            return Some(InputEvent::Byte(byte));
        }
        match input.rx.recv().await? {
            StreamItem::Bytes { bytes, bulk } => {
                input.current = bytes.into();
                input.current_bulk = bulk;
            }
            StreamItem::Resized => return Some(InputEvent::Resized),
        }
    }
}

/// Polls the terminal size and reports every change on the input stream. The
/// thread exits once the channel closes (application shutdown).
fn spawn_resize_watcher(sender: mpsc::UnboundedSender<StreamItem>) {
    std::thread::Builder::new()
        .name("rustagent-resize".into())
        .spawn(move || {
            let mut last = terminal_dimensions();
            loop {
                std::thread::sleep(RESIZE_POLL_INTERVAL);
                let current = terminal_dimensions();
                if current == last {
                    continue;
                }
                last = current;
                if sender.send(StreamItem::Resized).is_err() {
                    return;
                }
            }
        })
        .expect("spawn resize watcher");
}

fn terminal_dimensions() -> Option<(u16, u16)> {
    terminal_size().map(|(Width(width), Height(height))| (width, height))
}

fn spawn_blocking_stdin(sender: mpsc::UnboundedSender<StreamItem>) {
    tokio::spawn(async move {
        use tokio::io::AsyncReadExt;
        let mut stdin = tokio::io::stdin();
        let mut byte = [0u8; 1];
        loop {
            match stdin.read(&mut byte).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let chunk = StreamItem::Bytes {
                        bytes: vec![byte[0]],
                        bulk: false,
                    };
                    if sender.send(chunk).is_err() {
                        break;
                    }
                }
            }
        }
    });
}

/// Enables raw-input mode for the session when stdin and stdout are both TTYs.
/// Restore happens on drop.
#[cfg(unix)]
pub fn enable_raw_mode() -> Option<raw_mode::Guard> {
    if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        raw_mode::enable().ok()
    } else {
        None
    }
}

#[cfg(not(unix))]
pub fn enable_raw_mode() -> Option<()> {
    None
}

#[cfg(unix)]
fn async_fd_stdin() -> Option<tokio::io::unix::AsyncFd<std::fs::File>> {
    use std::os::fd::AsRawFd;
    use std::os::fd::FromRawFd;
    // On an interactive terminal fds 0/1/2 share one open file description,
    // so setting O_NONBLOCK on a dup of stdin leaks it to stdout and stderr
    // too. With stdout nonblocking, a backed-up terminal (a long paste keeps
    // the UI redrawing faster than the terminal drains) makes every println!
    // fail with EAGAIN, and the std print macros panic on that error: the
    // session dies mid-paste with no visible message, because the panic text
    // itself goes to the same blocked stderr. Reopening the controlling
    // terminal instead yields a fresh description whose O_NONBLOCK stays
    // private to this reader, leaving stdout blocking. Piped input keeps the
    // dup: there stdin and stdout are separate pipes, so the flag cannot
    // leak, and /dev/tty would be the wrong source anyway.
    let file = if std::io::stdin().is_terminal() {
        std::fs::OpenOptions::new()
            .read(true)
            .open("/dev/tty")
            .ok()
    } else {
        None
    };
    let file = match file {
        Some(file) => file,
        None => {
            // SAFETY: `dup` produces a new fd referring to the same open file
            // description; the O_NONBLOCK below then applies to it. When
            // stdin is not a terminal that description is a pipe, which
            // cannot leak to stdout, and this reader is its only consumer.
            unsafe {
                let fd = libc::dup(libc::STDIN_FILENO);
                if fd < 0 {
                    return None;
                }
                std::fs::File::from_raw_fd(fd)
            }
        }
    };
    // SAFETY: `fcntl` on a valid descriptor and `AsyncFd::register` with the
    // required O_NONBLOCK already set; both calls only touch `file`.
    unsafe {
        let flags = libc::fcntl(file.as_raw_fd(), libc::F_GETFL);
        if flags < 0 {
            // Dropping `file` closes the descriptor on all return paths below.
            return None;
        }
        if libc::fcntl(file.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) < 0 {
            return None;
        }
        match tokio::io::unix::AsyncFd::register(file) {
            Ok(async_fd) => Some(async_fd),
            Err(_) => None,
        }
    }
}

#[cfg(unix)]
pub mod raw_mode {
    use libc::{
        ECHO, ECHOE, ICANON, ICRNL, INLCR, ISTRIP, IXON, STDIN_FILENO, TCSANOW, VMIN, VTIME,
        tcgetattr, tcsetattr, termios,
    };
    use std::io::{self, Write};

    const BRACKETED_PASTE_ENABLE: &[u8] = b"\x1b[?2004h";
    const BRACKETED_PASTE_DISABLE: &[u8] = b"\x1b[?2004l";

    pub struct Guard {
        original: termios,
    }

    pub fn enable() -> io::Result<Guard> {
        // SAFETY: `tcgetattr` reads into a POD `termios` we own.
        unsafe {
            let mut termios: termios = std::mem::zeroed();
            if tcgetattr(STDIN_FILENO, &mut termios) != 0 {
                return Err(io::Error::last_os_error());
            }
            let original = termios;
            // Raw input, but keep output post-processing (OPOST) and signals (ISIG).
            termios.c_lflag &= !(ICANON | ECHO | ECHOE);
            termios.c_iflag &= !(IXON | ICRNL | INLCR | ISTRIP);
            termios.c_cc[VMIN] = 1;
            termios.c_cc[VTIME] = 0;
            if tcsetattr(STDIN_FILENO, TCSANOW, &termios) != 0 {
                return Err(io::Error::last_os_error());
            }
            // Bracketed paste lets multi-line clipboard text stay inside the prompt.
            let mut stdout = io::stdout();
            let _ = stdout.write_all(BRACKETED_PASTE_ENABLE);
            let _ = stdout.flush();
            Ok(Guard { original })
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            let mut stdout = io::stdout();
            let _ = stdout.write_all(BRACKETED_PASTE_DISABLE);
            let _ = stdout.flush();
            // SAFETY: restores the previously saved settings.
            unsafe {
                tcsetattr(STDIN_FILENO, TCSANOW, &self.original);
            }
        }
    }
}
