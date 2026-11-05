//! Model reading speed: average output tokens per second.
//! The window starts with the first streamed delta (excluding prefill/prompt eval)
//! and ends when the response completes. During the stream, tokens are estimated from
//! bytes (÷3); on close, exact usage is used when provided.

use std::time::{Duration, Instant};

const MIN_WINDOW: Duration = Duration::from_secs(1);

pub struct GenerationRate {
    active: bool,
    first_delta: Option<Instant>,
    output_bytes: u64,
    last_rate: Option<f64>,
}

impl Default for GenerationRate {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerationRate {
    pub fn new() -> Self {
        Self {
            active: false,
            first_delta: None,
            output_bytes: 0,
            last_rate: None,
        }
    }

    pub fn begin(&mut self) {
        self.active = true;
        self.first_delta = None;
        self.output_bytes = 0;
        self.last_rate = None;
    }

    pub fn note_delta(&mut self, text: &str) {
        if !self.active || text.is_empty() {
            return;
        }
        self.output_bytes += text.len() as u64;
        if self.first_delta.is_none() {
            self.first_delta = Some(Instant::now());
        }
    }

    pub fn live(&self) -> Option<f64> {
        if !self.active {
            return None;
        }
        let first = self.first_delta?;
        let elapsed = first.elapsed();
        if elapsed < MIN_WINDOW {
            return None;
        }
        rate(elapsed, self.output_bytes.div_ceil(3))
    }

    /// Closes the reading; prefers exact completion tokens from the endpoint usage.
    pub fn finish(&mut self, completion_tokens: Option<u64>) -> Option<f64> {
        if !self.active {
            return self.last_rate;
        }
        self.active = false;
        let Some(first) = self.first_delta else {
            self.last_rate = None;
            return None;
        };
        let tokens = completion_tokens
            .filter(|tokens| *tokens > 0)
            .unwrap_or_else(|| self.output_bytes.div_ceil(3));
        self.last_rate = rate(first.elapsed(), tokens);
        self.last_rate
    }

    /// Speed to display: the current reading if measurable; otherwise the last completed one.
    pub fn current(&self) -> Option<f64> {
        self.live().or(self.last_rate)
    }
}

fn rate(elapsed: Duration, tokens: u64) -> Option<f64> {
    if elapsed < MIN_WINDOW || tokens == 0 {
        return None;
    }
    Some(tokens as f64 / elapsed.as_secs_f64())
}

pub fn format_tokens_per_second(rate: Option<f64>) -> Option<String> {
    let rate = rate?;
    if !rate.is_finite() || rate <= 0.0 {
        return None;
    }
    let rounded = (rate * 10.0).round() / 10.0;
    Some(if rounded >= 100.0 {
        format!("{} t/s", rounded.round() as u64)
    } else {
        format!("{rounded:.1} t/s")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn live_rate_requires_a_one_second_window() {
        let mut tracker = GenerationRate::new();
        tracker.begin();
        assert_eq!(tracker.live(), None);
        tracker.note_delta("abc");
        assert_eq!(tracker.live(), None);
        sleep(Duration::from_millis(1100));
        let rate = tracker.live().expect("rate after one second");
        assert!(rate > 0.0 && rate < 3.0, "unexpected rate {rate}");
    }

    #[test]
    fn finish_prefers_exact_usage_tokens() {
        let mut tracker = GenerationRate::new();
        tracker.begin();
        tracker.note_delta(&"x".repeat(300));
        sleep(Duration::from_millis(1100));
        let rate = tracker.finish(Some(80)).expect("rate with usage");
        assert!(rate > 40.0 && rate < 100.0, "unexpected rate {rate}");
        assert_eq!(tracker.current(), Some(rate));
    }

    #[test]
    fn new_reading_clears_previous_rate() {
        let mut tracker = GenerationRate::new();
        tracker.begin();
        tracker.note_delta(&"a".repeat(60));
        sleep(Duration::from_millis(1100));
        assert!(tracker.finish(None).is_some());
        assert!(tracker.current().is_some());
        tracker.begin();
        assert_eq!(tracker.current(), None);
    }

    #[test]
    fn short_or_empty_readings_report_nothing() {
        let mut tracker = GenerationRate::new();
        tracker.begin();
        assert_eq!(tracker.finish(None), None);
        tracker.begin();
        tracker.note_delta("hi");
        sleep(Duration::from_millis(100));
        assert_eq!(tracker.finish(None), None);
    }

    #[test]
    fn formatting_rounds_large_rates() {
        assert_eq!(format_tokens_per_second(Some(18.44)).unwrap(), "18.4 t/s");
        assert_eq!(format_tokens_per_second(Some(99.96)).unwrap(), "100 t/s");
        assert_eq!(format_tokens_per_second(Some(120.6)).unwrap(), "121 t/s");
        assert_eq!(format_tokens_per_second(Some(0.0)), None);
        assert_eq!(format_tokens_per_second(None), None);
    }
}
