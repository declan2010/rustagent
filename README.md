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
| `SKILLS_MODE` | `off` | `auto`, `ask`, or `off`; discovered `SKILL.md` catalogs enable the `load_skill` tool |
| `MCP_MODE` | `off` | `auto`, `ask`, or `off`; connects the servers in `.rustagent/mcp.json` |

On the native Ollama path (`OPENAI_OLLAMA_THINK` set), RustAgent sends
`options.num_ctx = OPENAI_CONTEXT_WINDOW` so long conversations are not truncated
to Ollama's small default context. Raising the window increases VRAM use; on an
unstable GPU a smaller window may be safer.

## Chat bubbles

Conversation turns render like MinAgent's chat bubbles:

- Your submitted message echoes as a right-aligned **YOU bubble** (magenta
  frame, teal content) — same title, wrap width (66 content columns), and
  padding as MinAgent's `printUserBubble`.
- The model's streamed reply renders inside a left-aligned bubble per round
  (cyan frame, navy background): a `╭─ Model · <name>` title, word-wrapped
  markdown output, and a closing `╰─ complete`/`╰─ interrupted`/`╰─ incomplete`
  status line. Rounds that carry only tool calls open no bubble, reasoning
  stays muted, and tool results print between the frames.

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
