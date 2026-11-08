//! Update notification: compares `CARGO_PKG_VERSION` against the latest tag
//! published in this repository's GitHub releases. The check runs once per
//! run, never blocks (the notice is handed to the main loop through a
//! channel), and stays silent on any network or format failure.

use anyhow::Result;

const RELEASES_API: &str = "https://api.github.com/repos/declan2010/rustagent/releases/latest";

/// Fetches the latest release tag ("vX.Y.Z"), or `None` on any failure
/// (no releases published, rate limit, offline).
pub async fn latest_release_tag() -> Option<String> {
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(4))
        .build()
        .ok()?;
    let response = http
        .get(RELEASES_API)
        .header("User-Agent", "rustagent")
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    let payload: serde_json::Value = response.json().await.ok()?;
    payload
        .get("tag_name")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
        .map(str::to_string)
}

/// True when `remote` is a strict numeric upgrade of `current`. Both accept an
/// optional leading `v`; non-numeric or garbage tags never count as newer.
pub fn is_newer(remote: &str, current: &str) -> bool {
    let (Some(remote), Some(current)) = (semver_numbers(remote), semver_numbers(current)) else {
        return false;
    };
    let width = remote.len().max(current.len());
    for index in 0..width {
        let left = remote.get(index).copied().unwrap_or(0);
        let right = current.get(index).copied().unwrap_or(0);
        if left != right {
            return left > right;
        }
    }
    false
}

/// Numeric components of a semver-ish version: "v0.40.13-rc1" → [0, 40, 13]
/// (suffixes after a dash are ignored, since a pre-release cannot outrank its
/// own release). Returns `None` when nothing numeric is parseable.
fn semver_numbers(version: &str) -> Option<Vec<u64>> {
    let without_v = version.strip_prefix('v').unwrap_or(version);
    let numeric_head = without_v.split(['-', '+']).next().unwrap_or("");
    let numbers: Vec<u64> = numeric_head
        .split('.')
        .map(|part| part.trim().parse::<u64>())
        .collect::<Result<_, _>>()
        .ok()?;
    if numbers.is_empty() {
        return None;
    }
    Some(numbers)
}

/// The one-line notice to print when an update is available.
pub fn notice(remote: &str, current: &str) -> Result<String> {
    Ok(crate::ui::muted(&format!(
        "· Update available: {} (this build is {}). See https://github.com/declan2010/rustagent/releases",
        remote, current
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newer_remotes_are_detected_once_each_component_grows() {
        assert!(is_newer("v0.2.0", "v0.1.0"));
        assert!(is_newer("0.40.15", "0.40.11"));
        assert!(is_newer("v1.0", "v0.9.9"));
        assert!(is_newer("v0.2", "v0.1.10"));
    }

    #[test]
    fn equal_or_older_remotes_are_ignored() {
        assert!(!is_newer("v0.1.0", "v0.1.0"));
        assert!(!is_newer("0.40.10", "0.40.11"));
        assert!(!is_newer("v0.1", "v0.1.0"));
    }

    #[test]
    fn garbage_or_prerelease_versions_never_count() {
        assert!(!is_newer("", "0.1.0"));
        assert!(!is_newer("latest", "0.1.0"));
        assert!(
            !is_newer("v0.2.0-rc1", "v0.2.0"),
            "pre-release of the same build"
        );
        assert!(!is_newer("", ""));
    }

    #[test]
    fn notice_mentions_both_versions() {
        let text = notice("v0.2.0", "0.1.0").expect("notice");
        assert!(text.contains("v0.2.0"));
        assert!(text.contains("0.1.0"));
        assert!(text.contains("releases"));
    }
}
