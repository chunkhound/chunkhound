//! Repository-name resolution: `git remote get-url origin`, parsed to a repo
//! name, falling back to the directory basename. Never panics, and never
//! blocks past `GIT_TIMEOUT` — a shelled-out `git` that's missing, fails,
//! hangs (unreachable network mount, an interactive credential-helper
//! prompt), or returns garbage must never break analytics or stall the host
//! command that's constructing an `AnalyticsRecorder`.

use std::path::Path;
use std::process::{Command, Output};
use std::sync::mpsc;
use std::time::Duration;

/// Mirrors `chunkhound/utils/git_safe.py`'s default `run_git()` timeout. This
/// is the only git shell-out in the Rust crate, so a shared cross-language
/// helper isn't worth it, but it must carry the same "never hang the caller"
/// guarantee as the Python side's isolated `run_git()`.
const GIT_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) fn resolve_repository_name(dir: &Path) -> String {
    if let Some(name) = remote_repo_name(dir) {
        return name;
    }
    dir.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn remote_repo_name(dir: &Path) -> Option<String> {
    let mut cmd = Command::new("git");
    cmd.args(["remote", "get-url", "origin"])
        .current_dir(dir)
        // Isolation mirrors `git_safe.py::_build_git_env` — avoids an
        // interactive credential-helper prompt or an unusual global/system
        // git config turning this into the exact kind of hang `GIT_TIMEOUT`
        // exists to bound.
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", git_config_null())
        .env("GIT_CONFIG_SYSTEM", git_config_null());
    let output = run_with_timeout(cmd, GIT_TIMEOUT)?;
    if !output.status.success() {
        return None;
    }
    let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
    parse_repo_name(&url)
}

fn git_config_null() -> &'static str {
    if cfg!(windows) {
        "NUL"
    } else {
        "/dev/null"
    }
}

/// Runs `cmd` on a detached thread and waits at most `timeout` for it,
/// returning `None` on failure to launch, a non-zero-effort timeout, or any
/// other error. Mirrors `shutdown_blocking`'s thread + `mpsc::recv_timeout`
/// pattern elsewhere in this module: on timeout the orphaned process keeps
/// running in the background (best-effort, not killed) while this call
/// returns anyway — exactly the "never block the caller past the bound"
/// property needed here.
fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Option<Output> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(cmd.output());
    });
    rx.recv_timeout(timeout).ok()?.ok()
}

fn parse_repo_name(url: &str) -> Option<String> {
    let trimmed = url.trim().trim_end_matches('/').trim_end_matches(".git");
    trimmed
        .rsplit(['/', ':'])
        .next()
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command as ProcessCommand;
    use tempfile::tempdir;

    #[test]
    fn parses_https_remote_url() {
        assert_eq!(
            parse_repo_name("https://github.com/chunkhound/chunkhound.git"),
            Some("chunkhound".to_string())
        );
    }

    #[test]
    fn parses_ssh_remote_url() {
        assert_eq!(
            parse_repo_name("git@github.com:chunkhound/chunkhound.git"),
            Some("chunkhound".to_string())
        );
    }

    #[test]
    fn falls_back_to_directory_basename_without_git() {
        let dir = tempdir().unwrap();
        let project = dir.path().join("my-project");
        std::fs::create_dir(&project).unwrap();
        assert_eq!(resolve_repository_name(&project), "my-project");
    }

    #[test]
    fn resolves_from_a_real_git_remote() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        let git = |args: &[&str]| {
            assert!(ProcessCommand::new("git")
                .args(args)
                .current_dir(repo)
                .status()
                .expect("git must be on PATH for this test")
                .success());
        };
        git(&["init", "-q"]);
        git(&[
            "remote",
            "add",
            "origin",
            "https://example.com/org/some-repo.git",
        ]);
        assert_eq!(resolve_repository_name(repo), "some-repo");
    }

    // Unix-only: exercises `run_with_timeout` with a genuinely slow child
    // process (`sh -c 'sleep ...'`) rather than the real `git` binary, so it
    // doesn't depend on being able to make git itself hang. Regression test
    // for the bug this fix addresses — a hung `git` used to block
    // `AnalyticsRecorder::new()` indefinitely with no bound.
    #[test]
    #[cfg(unix)]
    fn run_with_timeout_returns_none_promptly_instead_of_blocking() {
        let mut cmd = ProcessCommand::new("sh");
        cmd.args(["-c", "sleep 10"]);
        let start = std::time::Instant::now();
        let result = run_with_timeout(cmd, Duration::from_millis(200));
        assert!(result.is_none());
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "run_with_timeout must return near its timeout, not wait for the child to exit"
        );
    }

    #[test]
    fn run_with_timeout_returns_output_of_a_fast_command() {
        let mut cmd = ProcessCommand::new("git");
        cmd.arg("--version");
        let result = run_with_timeout(cmd, GIT_TIMEOUT);
        assert!(result.is_some_and(|o| o.status.success()));
    }
}
