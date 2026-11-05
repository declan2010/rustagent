use crate::types::Message;

/// Token estimate used across RustAgent: UTF-8 bytes divided by 3, rounded up.
pub fn estimate_text_tokens(value: &str) -> u64 {
    ((value.len() as u64) + 2) / 3
}

pub fn estimate_message_tokens(message: &Message) -> u64 {
    let mut text = message.text();
    if let Some(calls) = &message.tool_calls {
        if let Ok(serialized) = serde_json::to_string(calls) {
            text.push_str(&serialized);
        }
    }
    estimate_text_tokens(&text)
}

pub fn estimate_messages_tokens(messages: &[Message], tools_json: &str) -> u64 {
    messages.iter().map(estimate_message_tokens).sum::<u64>() + estimate_text_tokens(tools_json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn estimates_bytes_over_three_rounded_up() {
        assert_eq!(estimate_text_tokens(""), 0);
        assert_eq!(estimate_text_tokens("abc"), 1);
        assert_eq!(estimate_text_tokens("abcd"), 2);
        assert_eq!(estimate_text_tokens("ááá"), 2);
    }

    #[test]
    fn message_estimates_include_tool_calls() {
        let plain = Message::user("hello");
        let with_call = Message::assistant(
            Some(json!(null)),
            Some(vec![crate::types::ToolCall {
                id: "call_0".into(),
                kind: "function".into(),
                function: crate::types::FunctionCall {
                    name: "read_file".into(),
                    arguments: "{\"path\":\"a\"}".into(),
                },
            }]),
        );
        assert!(estimate_message_tokens(&with_call) > estimate_message_tokens(&plain));
    }
}
