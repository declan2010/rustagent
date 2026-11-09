# RustAgent

A Rust clone of [MinAgent](../MinAgent): a terminal coding agent for OpenAI-compatible
endpoints and Ollama. Same configuration variables, same streaming semantics (SSE and
Ollama NDJSON), same workspace tools, and the same `proc` / `t/s` status line.

The version from `Cargo.toml` surfaces in the session banner (`RustAgent <version> ·
SESSION`), in `--version` / `--help`, and drives the update checker, so every
release identifies itself on startup.

This is **milestone 1** of the port: the core chat loop, streaming parsers, tool loop,
workspace tools, permissions, manual compaction, the persistent bottom footer with a
raw-mode line editor and message queueing, Ctrl+T think cycling, Esc interruption,
gpt-oss action-tag parsing, and unit tests are in place. Skills, MCP, `/init`,
and image input are not ported yet (see [Parity status](#parity-status)).

## Build and run

```sh
cargo build --release
./target/release/rustagent
```

Configuration is read from the process environment and an optional `.env` in the
current directory, `~/.config/rustagent/.env`, or next to the binary. Process
environment wins over `.env`, and a project `.env` wins over the user config, as in
MinAgent.

```sh
cp .env.example .env
# edit .env, then:
cargo run
```

`OPENAI_MODEL` is required. Point `OPENAI_BASE_URL` at any OpenAI-compatible
`/v1` endpoint (llama.cpp, Ollama, vLLM, LM Studio, a cloud bridge, ...). The endpoint
is normalized to `.../chat/completions`; a trailing `/chat/completions` is accepted.

### Running on macOS

RustAgent is portable Rust and builds on Apple silicon and Intel Macs the same way
as on Linux — the raw-input (`termios`), stdin reader (`tokio::io::unix::AsyncFd`,
backed by kqueue on macOS), and `run_terminal` (`sh -c`) code paths are all under
`#[cfg(unix)]`, so no Linux-specific syscall is used.

```sh
# 1. Install Rust (stable) if you don't have it:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Build:
cd rustagent
cargo build --release
./target/release/rustagent
```

Then configure a backend. The easiest is local Ollama:

```sh
# Install Ollama from https://ollama.com (or `brew install ollama`) and start it,
# then pull a model, e.g.:
ollama pull qwen3:0.6b

# RustAgent reads config from ~/.config/rustagent/.env (or a project .env):
mkdir -p ~/.config/rustagent
cat > ~/.config/rustagent/.env <<'EOF'
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_MODEL=qwen3:0.6b
OPENAI_CONTEXT_WINDOW=65536
OPENAI_OLLAMA_THINK=low
TERMINAL_MODE=auto
EOF
```

`OPENAI_OLLAMA_THINK` (any of `false`/`low`/`medium`/`high`) routes requests to
Ollama's native `/api/chat` and is required for the Ctrl+T think-level cycle. Leave
it unset to use the OpenAI-compatible `/v1/chat/completions` path (needed for cloud
endpoints without a native Ollama API). See [Configuration](#configuration) for the
full variable list.

Notes:
- Verified on Linux in this repo; macOS compiles on the same Unix paths but has not
  been run end-to-end here. If you hit a macOS-specific issue, report it and it can be
  fixed in the `#[cfg(unix)]` code.
- Apple Terminal and iTerm2 both support the truecolor footer and bracketed paste.


### Releases

Every push to `master` publishes a version automatically (`.github/workflows/release.yml`):
conventional commits decide the bump — `feat:` (and, before 1.0, breaking changes) bump
the minor, anything else bumps the patch — then the workflow updates `Cargo.toml` and
`Cargo.lock`, tags `v<version>`, and creates a GitHub Release listing the commits.
Releases feed the built-in update checker, which prints a notice when a newer release
exists.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `OPENAI_API_KEY` | unset | Bearer token |
| `OPENAI_MODEL` | required | Model identifier |
| `OPENAI_CONTEXT_WINDOW` | `262144` | Context window cap; local Ollama models also clamp to their declared maximum |
| `OPENAI_TIMEOUT_MS` | `0` | Request timeout; `0` disables it |
| `OPENAI_OLLAMA_THINK` | unset | Enables native `/api/chat` with a think level |
| `OPENAI_INPUT` | `text,image` | Input modalities; local Ollama models override them with their declared capabilities (`vision` → image, `audio` → audio) |
| `TERMINAL_MODE` | `ask` | `auto`, `ask`, or `off`; `off` hides `run_terminal` |
| `SKILLS_MODE` | `off` | `auto`, `ask`, or `off`; discovered `SKILL.md` catalogs enable the `load_skill` tool |
| `MCP_MODE` | `off` | `auto`, `ask`, or `off`; connects the servers in `.rustagent/mcp.json` |

On the native Ollama path (`OPENAI_OLLAMA_THINK` set), RustAgent sends
`options.num_ctx = OPENAI_CONTEXT_WINDOW` so long conversations are not truncated
to Ollama's small default context. Raising the window increases VRAM use; on an
unstable GPU a smaller window may be safer.

The configured window is a **cap**: for local Ollama models the model's declared
maximum (from `/api/show` metadata, fetched with the same cached snapshot as
the think states) clamps it in both directions — a model that only supports
32k never receives a 65k `num_ctx`, and a model that declares 262k keeps the
configured cap as a hardware budget. The banner, the token meter, and the
auto-compaction budget all follow the effective window, and a muted notice
explains the clamp at startup and after `/model` switches. Endpoints without
the snapshot (cloud, non-Ollama) keep the configured value unchanged.

### Commands

- `/new` — start a new conversation
- `/model` — list models; `/model <number|name>` switches
- `/compact` — summarize earlier history and keep the checkpoint
- `/help` — command list
- `/exit` — quit

## What is implemented

- OpenAI-compatible SSE streaming: content, reasoning deltas, tool-call assembly, usage,
  `[DONE]`, `STREAM_ERROR`/`STREAM_ENDED`/`INVALID_TOOL_CALL` checks.
- Ollama native NDJSON streaming via `OPENAI_OLLAMA_THINK`: `message.thinking`,
  `message.tool_calls`, `done`/`eval_count`, and the repeated-call replacement rules.
- Tool-call id unification across rounds (`unique_tool_call_ids`), so cloud upstreams
  cannot attribute a tool result to the wrong round.
- Tool loop with schema validation, "Did you mean" suggestions, tool-result metadata,
  a 16-calls-per-round limit, and a 32-round safety stop.
- Workspace tools: `read_file`, `list_directory`, `edit_file`, `write_file`,
  `delete_file`, `delete_directory`, `search_files`, `run_terminal`.
- Path sandboxing: writes/edits/deletes/listings/search stay inside the workspace;
  `read_file` accepts an absolute path outside it (explicit user-named reads).
- `run_terminal` permission modes: `auto`, `ask` (y/N prompt), `off` (tool hidden),
  with a 120 s timeout and a 64 KiB output cap.
- Token estimate (UTF-8 bytes ÷ 3) and the per-reading `t/s` meter: the window starts at
  the first streamed delta (excluding prefill) and ends at completion; live values use
  the byte estimate, the final value uses exact `usage.completion_tokens`.
- Manual `/compact` using the MinAgent summary checkpoint format.
- Auto-compaction before the context limit: before each model request, if usage
  crosses `context_window - reserve` (reserve = `min(16384, window/4)`), earlier
  history is summarized with the same checkpoint format and the recent
  `keep_recent` tokens (`min(20000, window/4)`) are kept verbatim. Mirrors
  MinAgent's `compactAutomaticallyIfNeeded` and `findCompactionCutPoint`.
- Build/Plan modes like MinAgent: **Tab** at the prompt toggles between Build
  (all tools) and Plan (read-only: `read_file`, `list_directory`, `search_files`).
  Plan mode advertises only those tools and rejects write/command tool calls with
  `Tool 'X' is unavailable in Plan mode. Switch to Build.`; the system prompt and
  prompt label (`Plan ›` orange / `Build ›` green) switch accordingly, context usage
  is reset, and the footer shows a `Running: Plan/Build` segment when a turn runs in
  a different mode than the selected one.
- `/model` works like MinAgent's: it fetches the API catalog (`GET /models`), and with
  no argument opens an **interactive selector** in the footer — a titled list
  (`┌─ MODELS · N matches`) with a `›` highlight, **↑/↓** to move, **Enter** to
  switch, **Esc** to cancel (mirroring MinAgent's `formatAutocompletePanel`).
  `/model <number|name|prefix>` switches directly. Selection applies the
  catalog entry's context window and input modalities when the API supplies them
  (falling back to the configured defaults), resets context usage, prints fallback
  notices, and **persists `OPENAI_MODEL=<id>` to `~/.config/rustagent/.env` so
  the chosen model becomes the default for future sessions. Prefix matching
  mirrors MinAgent: exact id, else a case-insensitive prefix (`qwen3` → ambiguous
  list), else error. Non-interactive runs fall back to the printed numbered list.
  On a **local Ollama endpoint with an empty catalog**, `/model` pulls a small
  default model (`qwen3:0.6b`) through `POST /api/pull` — sparse progress lines
  (event kinds + every 10% step) — and re-opens the selector with the refreshed
  catalog.
- Persistent bottom footer (ported from MinAgent's scroll-region renderer): truecolor
  status row with model, think level, token usage and a 10-cell meter, activity,
  cumulative `proc` time, and live/final `t/s`; a hints row ending with the workspace
  root path. It clears the screen on start, pins the transcript to a scrolling region,
  redraws every ~100 ms while streaming and every second during silent tools, falls
  back to a plain status line when the output is not a TTY, and resets the region on
  exit.
- GFM table rendering (ported from MinAgent's `markdown-terminal.mjs`): when the model
  emits a Markdown table (`|` header, separator, body rows), it is buffered and
  rendered with aligned columns, `│` cell borders, a `─┼─` separator, and a bold
  cyan header row. Inline Markdown inside cells is stripped. Non-table text still
  streams raw.
- Ctrl+T at the prompt cycles the think states the selected model actually declares.
  The model's Ollama `/api/show` snapshot (cached per model) reports the `thinking`
  capability and the effort levels its chat template declares (`reasoning_effort`,
  e.g. qwen3.8: `xhigh → medium → low`). Models that cannot think keep thinking off
  and Ctrl+T says so; models with a plain template cycle `off ↔ on`; models with
  unknown snapshots use the generic MinAgent cycle (`off → low → medium → high`).
  The level applies to the next request. Only active for local Ollama endpoints;
  cooked REPL otherwise. The `/model` selector marks models without the thinking
  capability as `(no think)`, and switching models prints which levels are valid.
  Because current Ollama builds may not map think levels for native GGUF models
  (the template default applies instead), a one-line warning is printed the first
  time such a level is selected.
- Esc interrupts the current turn: it cancels the in-flight model stream (or the
  running terminal command, which is killed via `kill_on_drop`), prints
  "Response stopped. You can send a new message.", and returns to `Ready`. Typed-ahead
  keys pressed during a turn are preserved for the next prompt.

## Chat bubbles

Conversation turns render like MinAgent's chat bubbles:

- Your submitted message echoes as a right-aligned **YOU bubble** (magenta
  frame, teal content) — same title, wrap width (66 content columns), and
  padding as MinAgent's `printUserBubble`.
- The model's streamed reply renders inside a left-aligned bubble per round
  (cyan frame, navy background): a `╭─ Model · <name>` title, word-wrapped
  markdown output, and a closing `╰─ complete`/`╰─ interrupted`/`╰─ incomplete`
  status line. Rounds that carry only tool calls open no bubble and tool
  results print between the frames.
- Reasoning streams into its own muted frame (`╭─ Thinking · <name>`) that
  wraps the thinking at word boundaries and closes with the round status —
  complete, interrupted, or incomplete — before the answer bubble opens.
  MinAgent prints thinking as plain muted text; the frame keeps the wall of
  thinking separated and readable, and reasoning that arrives after the
  answer started falls back to plain muted output.

## Terminal resize

MinAgent leaves resizes to readline; RustAgent re-renders on its own:

- A resize watcher reports terminal-size changes on the input stream, so the
  footer (scroll region, status, hints, input line) redraws at the new
  geometry immediately — even while idle, without waiting for a keystroke.
- Frames never exceed the live terminal: every bubble row clamps to the
  current width, so a resize mid-stream adapts the frame instead of
  wrapping its rows off-screen. Already-printed history keeps its width
  (terminals cannot reflow past output), and each new frame picks up the
  current terminal.

## Skills

When `SKILLS_MODE` is `auto` or `ask`, rustagent discovers `SKILL.md` files in
the same roots as MinAgent:

- The application directory's `skills/<skill-name>/` and `.agents/skills/<skill-name>/`
- The workspace's `skills/<skill-name>/` and `.agents/skills/<skill-name>/`

Each manifest needs YAML frontmatter with `name` and `description`; the name
must match its directory (lowercase letters, digits, hyphens). Up to 24 valid
skills load (`Feature setup` warnings report the rest and any parsing issues).
The Build system prompt receives the skill catalog bounded to 8 KiB with
descriptions capped at 160 characters, and the model uses one `load_skill`
tool: omit `path` to load the instructions, or set it to read a bundled text
resource. Reads are hardened like MinAgent: no symlinks or hard links, no
escapes outside the skill directory, 64 KiB per skill file and 32 KiB per
resource, UTF-8 only. In `ask`, each load waits for a `[y/N]` approval in the
footer; in `auto` it runs directly; `off` (the default) skips discovery and
hides the tool.

## MCP

When `MCP_MODE` is `auto` or `ask`, rustagent connects the servers configured
in `~/.config/rustagent/mcp.json` (next to rustagent's `.env`; MinAgent keeps
its own next to the app under `.minagent/mcp.json`):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "env": { },
      "cwd": "."
    },
    "remote": {
      "url": "https://example.invalid/mcp",
      "headers": { "Authorization": "Bearer token" }
    }
  }
}
```

Both transports from MinAgent work: stdio servers (spawned with the configured
command/args/env/cwd) and streamable-HTTP servers (JSON responses or SSE event
frames, `Mcp-Session-Id` and `MCP-Protocol-Version` headers, session DELETE on
exit). Each server goes through the `initialize` handshake (protocol versions
2024-11-05 through 2025-11-25), pages `tools/list` (up to 32 tools per
connection), and its instructions join the Build system prompt bounded to
8 KiB. Advertised functions are named `mcp_<server>_<tool>` with a 64-character
cap, keep their input schema (8 KiB each, 64 KiB combined), and run through
`tools/call` with results bounded to 48,000 characters. Limits mirror
MinAgent: 32 servers, 32 tools, 1 MiB config (regular, unlinked file). In
`ask`, every call waits for a `[y/N]` approval showing
`serverName/toolName`; in `auto` calls run directly; `off` (the default) skips
connection entirely. Plan mode never advertises MCP tools. Server images come
back as a one-line omission note because rustagent does not attach image parts
to messages yet.

## Parity status

| Area | Status |
| --- | --- |
| Config variables and precedence | Ported |
| SSE + Ollama NDJSON streaming | Ported |
| Tool loop, validation, unique ids | Ported |
| Workspace tools and permissions | Ported |
| `proc` + `t/s` persistent footer | Ported (truecolor status + hints; non-TTY plain line) |
| Manual `/compact` | Ported (simple checkpoint; no keep-recent cut tuning) |
| Auto-compaction before the context limit | Ported (MinAgent budget + cut point) |
| `/model` catalog, interactive selector, default persistence | Ported (footer panel with ↑/↓/Enter/Esc; prefix match; persists to `~/.config/rustagent/.env`) |
| Markdown terminal rendering | Partial: GFM tables are aligned with `│` borders and a bold header (ported from MinAgent); other Markdown stays raw |
| Persistent footer / line editor / message queue | Ported (truecolor footer, raw-mode editor, queued prompts) |
| Esc interruption | Ported (cancels streaming and terminal commands) |
| Build/Plan modes (Tab) | Ported |
| gpt-oss action-tag tool calls | Ported (parses `<|START_ACTION|>`/`<|START_THINKING|>` text tags) |
| Skills discovery and `load_skill` | Ported (`SKILL.md` roots like MinAgent; 24-skill/8 KiB catalog budget; `auto` runs loads, `ask` asks `[y/N]`) |
| MCP client (stdio + HTTP) | Ported (`mcp.json` servers; initialize handshake, paged `tools/list`, `tools/call`; 32-server/32-tool budgets; session DELETE on exit) |
| `/init`, image input | Not ported |
| AGENTS.md loading, evidence ledger | Not ported |

## Tests

```sh
cargo test
```

64 unit tests cover config parsing, endpoint normalization, token estimates, the `t/s`
meter, the footer (scroll-region rows, styled truncation, the processing clock),
the raw-mode line editor and message queue, model-default persistence, compaction
budget/cut point, SSE/NDJSON parsing, gpt-oss action-tag parsing, unique tool-call
ids, tool validation, workspace operations, search, terminal execution, and a full
streaming round trip against a mock HTTP server (no model required).

## Notes

- The project is intentionally a separate crate so it can evolve without touching the
  Node implementation.
- `run_terminal` executes with the user's permissions; the workspace sandbox applies to
  file tools, not to commands. Use `TERMINAL_MODE=off` or `ask` when in doubt.
