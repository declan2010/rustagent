# RustAgent

A Rust clone of [MinAgent](../MinAgent): a terminal coding agent for OpenAI-compatible
endpoints and Ollama. Same configuration variables, same streaming semantics (SSE and
Ollama NDJSON), same workspace tools, and the same `proc` / `t/s` status line.

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


### Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `OPENAI_API_KEY` | unset | Bearer token |
| `OPENAI_MODEL` | required | Model identifier |
| `OPENAI_CONTEXT_WINDOW` | `262144` | Context window used by the status line |
| `OPENAI_TIMEOUT_MS` | `0` | Request timeout; `0` disables it |
| `OPENAI_OLLAMA_THINK` | unset | Enables native `/api/chat` with a think level |
| `OPENAI_INPUT` | `text,image` | Input modalities |
| `TERMINAL_MODE` | `ask` | `auto`, `ask`, or `off`; `off` hides `run_terminal` |
| `SKILLS_MODE` | `off` | Reserved; skills are not ported yet |
| `MCP_MODE` | `off` | Reserved; MCP is not ported yet |

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
- `/model` works like MinAgent's: it fetches the API catalog (`GET /models`), lists
  identifiers (marking the active one, with context size when the API reports it), and
  `/model <number|name|prefix>` switches. Selection applies the catalog entry's context
  window and input modalities when the API supplies them (falling back to the configured
  defaults), resets context usage, prints fallback notices, and **persists
  `OPENAI_MODEL=<id>` to `~/.config/rustagent/.env` so the chosen model becomes the
  default for future sessions. Prefix matching mirrors MinAgent: exact id, else a
  case-insensitive prefix (`qwen3` → ambiguous list), else error.
- Persistent bottom footer (ported from MinAgent's scroll-region renderer): truecolor
  status row with model, think level, token usage and a 10-cell meter, activity,
  cumulative `proc` time, and live/final `t/s`; a hints row ending with the workspace
  root path. It clears the screen on start, pins the transcript to a scrolling region,
  redraws every ~100 ms while streaming and every second during silent tools, falls
  back to a plain status line when the output is not a TTY, and resets the region on
  exit.
- Ctrl+T at the prompt cycles the Ollama think level (`off → low → medium → high`),
  mirroring MinAgent; the footer updates immediately and the level applies to the next
  request. Only active for local Ollama endpoints; cooked REPL otherwise.
- Esc interrupts the current turn: it cancels the in-flight model stream (or the
  running terminal command, which is killed via `kill_on_drop`), prints
  "Response stopped. You can send a new message.", and returns to `Ready`. Typed-ahead
  keys pressed during a turn are preserved for the next prompt.

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
| `/model` catalog, settings, default persistence | Ported (prefix match; persists to `~/.config/rustagent/.env`) |
| Markdown terminal rendering | Not ported (raw streamed text) |
| Persistent footer / line editor / message queue | Ported (truecolor footer, raw-mode editor, queued prompts) |
| Esc interruption | Ported (cancels streaming and terminal commands) |
| Build/Plan modes (Tab) | Ported |
| gpt-oss action-tag tool calls | Ported (parses `<|START_ACTION|>`/`<|START_THINKING|>` text tags) |
| Skills, MCP, `/init`, image input | Not ported |
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
