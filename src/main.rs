mod agent;
mod config;
mod context;
mod footer;
mod genrate;
mod input;
mod openai;
mod prompt;
mod tools;
mod types;
mod ui;

use anyhow::Result;
use std::io::IsTerminal;

use agent::Agent;
use footer::Footer;
use input::Input;
use prompt::{EditorEvent, Session};

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        ui::print_error(&error.to_string());
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!(
            "RustAgent {} — a Rust clone of MinAgent",
            env!("CARGO_PKG_VERSION")
        );
        println!();
        println!("Usage: rustagent");
        println!();
        println!("Configuration is read from the environment and an optional .env file");
        println!("(current directory, ~/.config/rustagent/.env, or next to the binary):");
        println!(
            "  OPENAI_BASE_URL      OpenAI-compatible endpoint (default: https://api.openai.com/v1)"
        );
        println!("  OPENAI_API_KEY       Bearer token, when the endpoint requires one");
        println!("  OPENAI_MODEL         Model identifier (required)");
        println!("  OPENAI_CONTEXT_WINDOW Context window in tokens (default: 262144)");
        println!("  OPENAI_TIMEOUT_MS    Request timeout in milliseconds (0 disables it)");
        println!(
            "  OPENAI_OLLAMA_THINK  Enables Ollama native /api/chat with the given think level"
        );
        println!("  OPENAI_INPUT         Input modalities: text,image");
        println!("  TERMINAL_MODE        auto, ask, or off (default: ask)");
        println!("  SKILLS_MODE          auto, ask, or off (default: off)");
        println!("  MCP_MODE             auto, ask, or off (default: off)");
        println!();
        println!("In an interactive terminal, Ctrl+T cycles the Ollama think level and Tab toggles Build/Plan mode.");
        return Ok(());
    }
    if arguments
        .iter()
        .any(|argument| argument == "--version" || argument == "-V")
    {
        println!("rustagent {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let config = config::load()?;
    let mut agent = Agent::new(config)?;
    let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    let footer = Footer::new();
    footer.init();
    // Raw-input mode for the session (keeps OPOST so println! still renders).
    let _raw_guard = input::enable_raw_mode();
    let mut input = Input::spawn();
    let mut session = Session::new(ui::mode_prompt(agent.selected_mode()), interactive);
    session.note_agent_state(agent.selected_mode(), agent.config.ollama_think.clone());
    ui::print_banner(&agent.config, agent.estimated_tokens());
    agent.render_ready(&footer);
    session.render(&footer);

    loop {
        if session.exit_requested() {
            break;
        }
        let (line, from_queue) = if let Some(queued) = session.pop_queued() {
            agent.apply_queued(&footer, queued.mode, queued.think);
            session.note_agent_state(agent.selected_mode(), agent.config.ollama_think.clone());
            session.set_prompt(ui::mode_prompt(agent.selected_mode()));
            session.render(&footer);
            (queued.text, true)
        } else {
            match next_event(&mut agent, &footer, &mut input, &mut session).await {
                EditorEvent::Submit(text) => (text, false),
                EditorEvent::ToggleMode => {
                    agent.toggle_mode(&footer);
                    session.note_agent_state(
                        agent.selected_mode(),
                        agent.config.ollama_think.clone(),
                    );
                    session.set_prompt(ui::mode_prompt(agent.selected_mode()));
                    session.render(&footer);
                    continue;
                }
                EditorEvent::CycleThink => {
                    cycle_think(&mut agent, &footer);
                    session.note_agent_state(
                        agent.selected_mode(),
                        agent.config.ollama_think.clone(),
                    );
                    session.render(&footer);
                    continue;
                }
                EditorEvent::Abort | EditorEvent::Cancel | EditorEvent::None => continue,
                EditorEvent::Exit => break,
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            session.set_prompt(ui::mode_prompt(agent.selected_mode()));
            session.render(&footer);
            continue;
        }
        if !from_queue {
            ui::print_submitted(session.prompt(), trimmed);
        }
        let exited = handle_input(&mut agent, &footer, trimmed, &mut input, &mut session).await;
        agent.apply_steering(&footer, &session.take_steering());
        session.note_agent_state(agent.selected_mode(), agent.config.ollama_think.clone());
        session.set_prompt(ui::mode_prompt(agent.selected_mode()));
        session.render(&footer);
        if exited || session.exit_requested() {
            break;
        }
    }
    drop(_raw_guard);
    footer.finish();
    println!();
    Ok(())
}

async fn next_event(
    agent: &mut Agent,
    footer: &Footer,
    input: &mut Input,
    session: &mut Session,
) -> EditorEvent {
    if session.interactive() {
        return session.next_event(input, footer).await;
    }
    match prompt::read_prompt_line(input, ui::mode_prompt(agent.selected_mode()), |action| {
        on_prompt_action(agent, footer, action)
    })
    .await
    {
        Ok(Some(line)) => EditorEvent::Submit(line),
        Ok(None) => EditorEvent::Exit,
        Err(error) => {
            ui::print_error(&error.to_string());
            EditorEvent::Exit
        }
    }
}

fn cycle_think(agent: &mut Agent, footer: &Footer) {
    if !agent.is_local_endpoint() {
        ui::print_info(&ui::warning(
            "· Ctrl+T only controls thinking with a local Ollama endpoint.",
        ));
        return;
    }
    let next = agent::next_think_level(agent.config.ollama_think.as_deref());
    agent.config.ollama_think = next.map(|level| level.to_string());
    agent.client.ollama_think = agent.config.ollama_think.clone();
    agent.set_activity(footer, "Ready", agent.last_rate);
    ui::print_info(&ui::warning(&format!(
        "· Thinking: {} · applies to the next request.",
        agent.think_label_string()
    )));
}

/// Prompt-level key handler: Tab toggles Build/Plan mode, Ctrl+T cycles the think
/// level. Returns the new prompt label and/or a styled notice.
fn on_prompt_action(
    agent: &mut Agent,
    footer: &Footer,
    action: prompt::PromptAction,
) -> prompt::PromptActionResult {
    match action {
        prompt::PromptAction::ToggleMode => {
            agent.toggle_mode(footer);
            prompt::PromptActionResult {
                prompt: Some(ui::mode_prompt(agent.selected_mode())),
                notice: None,
            }
        }
        prompt::PromptAction::CycleThink => {
            let notice = if !agent.is_local_endpoint() {
                ui::warning("· Ctrl+T only controls thinking with a local Ollama endpoint.")
            } else {
                let next = agent::next_think_level(agent.config.ollama_think.as_deref());
                agent.config.ollama_think = next.map(|level| level.to_string());
                agent.client.ollama_think = agent.config.ollama_think.clone();
                agent.set_activity(footer, "Ready", agent.last_rate);
                ui::warning(&format!(
                    "· Thinking: {} · applies to the next request.",
                    agent.think_label_string()
                ))
            };
            prompt::PromptActionResult {
                prompt: None,
                notice: Some(notice),
            }
        }
    }
}

/// Returns `true` to exit the REPL.
async fn handle_input(
    agent: &mut Agent,
    footer: &Footer,
    input: &str,
    stdin: &mut Input,
    session: &mut Session,
) -> bool {
    match input {
        "/exit" | "/quit" => {
            session.clear_queue();
            session.request_exit();
            return true;
        }
        "/help" => {
            ui::print_help();
            println!();
            return false;
        }
        "/new" => {
            agent.reset();
            ui::clear_screen();
            ui::print_banner(&agent.config, agent.estimated_tokens());
            agent.render_ready(footer);
            return false;
        }
        "/compact" => {
            match agent.compact(footer).await {
                Ok((before, after)) => ui::print_info(&format!(
                    "Compaction complete · context ~{before} → ~{after} tokens"
                )),
                Err(error) => ui::print_error(&error.to_string()),
            }
            println!();
            return false;
        }
        "/model" => {
            handle_model(agent, footer, None).await;
            println!();
            return false;
        }
        _ => {}
    }
    if let Some(selection) = input.strip_prefix("/model ") {
        handle_model(agent, footer, Some(selection.trim())).await;
        println!();
        return false;
    }
    if input.starts_with('/') {
        ui::print_error(&format!("Unknown command: {input}. Try /help."));
        return false;
    }

    match agent.turn(input, footer, stdin, session).await {
        Ok(()) => {
            println!();
            false
        }
        Err(error) => {
            ui::print_error(&error.to_string());
            println!();
            false
        }
    }
}

async fn handle_model(agent: &mut Agent, footer: &Footer, selection: Option<&str>) {
    agent.set_activity(footer, "Loading models", agent.last_rate);
    let catalog = agent.client.list_models().await;
    agent.set_activity(footer, "Ready", agent.last_rate);
    match selection {
        None => match catalog {
            Ok(models) => {
                ui::print_info(&format!("Models ({})", models.len()));
                for (index, entry) in models.iter().enumerate() {
                    let marker = if entry.id == agent.client.model {
                        "*"
                    } else {
                        " "
                    };
                    let context = entry
                        .context_window
                        .map(|value| format!("  ~{} ctx", ui::format_number(value)))
                        .unwrap_or_default();
                    println!(
                        " {marker} {:>3}. {}{}",
                        index + 1,
                        entry.id,
                        ui::muted(&context)
                    );
                }
                println!("{}", ui::dim("Use /model <number|name> to switch."));
            }
            Err(error) => ui::print_error(&error.to_string()),
        },
        Some(selection) => {
            let catalog = match catalog {
                Ok(catalog) => catalog,
                Err(error) => {
                    ui::print_error(&error.to_string());
                    return;
                }
            };
            let entry = match agent::resolve_selection(&catalog, selection) {
                Ok(entry) => entry,
                Err(error) => {
                    ui::print_error(&error.to_string());
                    return;
                }
            };
            if entry.id == agent.client.model {
                ui::print_info(&format!("Model {} is already selected.", entry.id));
                return;
            }
            match agent.switch_model(&entry) {
                Ok((known_context, known_input)) => {
                    println!("{}", ui::success_bold(&format!("Model switched to {}.", entry.id)));
                    match config::persist_model_default(&entry.id) {
                        Ok(path) => println!(
                            "{}",
                            ui::muted(&format!("· saved as default in {}", path.display()))
                        ),
                        Err(error) => println!(
                            "{}",
                            ui::warning(&format!("· could not save default: {error}"))
                        ),
                    }
                    if !known_context {
                        println!(
                            "{}",
                            ui::muted(&format!(
                                "Context size not supplied by the API; using configured {} tokens.",
                                ui::format_number(agent.config.context_window)
                            ))
                        );
                    }
                    if !known_input {
                        println!(
                            "{}",
                            ui::muted(&format!(
                                "Input capabilities not supplied by the API; using configured {}.",
                                agent.config.input_modalities.join(", ")
                            ))
                        );
                    }
                    agent.render_ready(footer);
                }
                Err(error) => ui::print_error(&error.to_string()),
            }
        }
    }
}
