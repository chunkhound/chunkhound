"""Unit tests for chunkhound.core.git_diff.runner."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from chunkhound.core.git_diff.parser import stream_diff_to_chunks
from chunkhound.core.git_diff.runner import (
    _MAX_STDERR_BYTES,
    stream_git_diff_fragments,
)


async def _diff_text(commit_range: str, cwd: Path) -> str:
    return "".join(
        [fragment async for fragment in stream_git_diff_fragments(commit_range, cwd)]
    )


class _Stdout:
    """Fake git stdout; each ``read`` call returns the next queued payload."""

    def __init__(
        self, reads: list[bytes], hang: bool = False, error: Exception | None = None
    ) -> None:
        self._reads = list(reads)
        self._hang = hang
        self._error = error

    async def read(self, n: int = -1) -> bytes:
        if self._error is not None:
            raise self._error
        if self._reads:
            return self._reads.pop(0)
        if self._hang:
            await asyncio.Event().wait()
        return b""


class _Stderr:
    def __init__(self, payload: bytes = b"", hang: bool = False) -> None:
        self._payload = payload
        self._hang = hang
        self.read_calls = 0

    async def read(self, n: int = -1) -> bytes:
        self.read_calls += 1
        if self._hang:
            await asyncio.Event().wait()
        if n is None or n < 0:
            chunk, self._payload = self._payload, b""
            return chunk
        chunk, self._payload = self._payload[:n], self._payload[n:]
        return chunk


class StreamProcess:
    def __init__(
        self,
        reads: list[bytes],
        hang_stdout: bool = False,
        hang_stderr: bool = False,
        error: Exception | None = None,
        exit_code: int = 0,
        stderr: bytes = b"",
    ) -> None:
        self.stdout = _Stdout(reads, hang=hang_stdout, error=error)
        self.stderr = _Stderr(payload=stderr, hang=hang_stderr)
        self.returncode: int | None = None
        self.killed = False
        self._exit_code = exit_code

    def kill(self) -> None:
        self.killed = True
        self.returncode = -1

    async def wait(self) -> None:
        if self.returncode is None:
            self.returncode = self._exit_code


@pytest.mark.asyncio
async def test_real_git_diff_yields_new_content_per_changed_file(
    tmp_path: Path,
) -> None:
    proc = await asyncio.create_subprocess_exec(
        "git", "init", cwd=str(tmp_path), stdout=asyncio.subprocess.PIPE
    )
    await proc.communicate()
    for name in ("one.py", "two.py"):
        (tmp_path / name).write_text(f"{name} = 1\n", encoding="utf-8")
    proc = await asyncio.create_subprocess_exec("git", "add", ".", cwd=str(tmp_path))
    await proc.communicate()
    proc = await asyncio.create_subprocess_exec(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "initial",
        cwd=str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    for name in ("one.py", "two.py"):
        (tmp_path / name).write_text(f"{name} = 2\n", encoding="utf-8")

    chunks = [
        chunk
        async for chunk in stream_diff_to_chunks(
            stream_git_diff_fragments("HEAD", cwd=tmp_path)
        )
    ]

    assert [str(chunk.file_path) for chunk in chunks] == ["one.py", "two.py"]
    assert "one.py = 2" in chunks[0].code
    assert "two.py = 2" in chunks[1].code


@pytest.mark.asyncio
async def test_real_git_preserves_exact_paths_despite_noprefix_config(
    tmp_path: Path,
) -> None:
    async def git(*args: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(tmp_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        assert proc.returncode == 0, stderr.decode("utf-8", errors="replace")

    await git("init")
    await git("config", "diff.noprefix", "true")
    paths = ("space name.py", "café.py", "b/nested.py")
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("value = 1\n", encoding="utf-8")
    await git("add", ".")
    await git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "initial",
    )
    for path in paths:
        (tmp_path / path).write_text("value = 2\n", encoding="utf-8")

    chunks = [
        chunk
        async for chunk in stream_diff_to_chunks(
            stream_git_diff_fragments("HEAD", cwd=tmp_path)
        )
    ]

    assert {str(chunk.file_path) for chunk in chunks} == set(paths)


@pytest.mark.asyncio
async def test_closing_stream_reaps_real_git_with_full_stdout(tmp_path: Path) -> None:
    async def git(*args: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(tmp_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        assert proc.returncode == 0

    await git("init")
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "b.py").write_text("b = 1\n")
    await git("add", ".")
    await git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "initial",
    )
    (tmp_path / "a.py").write_text("a = 2\n")
    (tmp_path / "b.py").write_text("b = 2\n" * 1_000_000)

    processes = []
    real_exec = asyncio.create_subprocess_exec

    async def tracked_exec(*args, **kwargs):
        proc = await real_exec(*args, **kwargs)
        processes.append(proc)
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=tracked_exec):
        stream = stream_diff_to_chunks(stream_git_diff_fragments("HEAD", cwd=tmp_path))
        assert str((await anext(stream)).file_path) == "a.py"
        # Allow git to fill the pipe while the consumer is paused at yield.
        await asyncio.sleep(0.2)
        await asyncio.wait_for(stream.aclose(), timeout=2)

    assert len(processes) == 1
    assert processes[0].returncode is not None
    assert processes[0].stdout.at_eof()


@pytest.mark.asyncio
async def test_stream_reassembles_line_longer_than_reader_limit(
    tmp_path: Path,
) -> None:
    """A single diff line longer than asyncio's default 64 KiB reader limit
    must not truncate the stream or raise -- diffs can legitimately contain
    very long single lines (minified bundles, generated lockfiles)."""
    proc = await asyncio.create_subprocess_exec(
        "git", "init", cwd=str(tmp_path), stdout=asyncio.subprocess.PIPE
    )
    await proc.communicate()
    (tmp_path / "bundle.js").write_text("var x = 1;\n", encoding="utf-8")
    proc = await asyncio.create_subprocess_exec("git", "add", ".", cwd=str(tmp_path))
    await proc.communicate()
    proc = await asyncio.create_subprocess_exec(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "initial",
        cwd=str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
    )
    await proc.communicate()

    long_line = "x" * 200_000  # comfortably over the 64 KiB default reader limit
    (tmp_path / "bundle.js").write_text(f"var x = 1;\n{long_line}\n", encoding="utf-8")

    diff = await _diff_text("HEAD", tmp_path)

    assert diff.count("diff --git ") == 1
    assert f"+{long_line}\n" in diff


@pytest.mark.asyncio
@pytest.mark.parametrize("object_format", ["sha1", "sha256"])
async def test_stream_root_commit_uses_repository_empty_tree(
    tmp_path: Path, object_format: str
) -> None:
    proc = await asyncio.create_subprocess_exec(
        "git",
        "init",
        f"--object-format={object_format}",
        cwd=str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 and object_format == "sha256":
        message = stderr.decode("utf-8", errors="replace")
        if "unknown option" in message or "unknown hash algorithm" in message:
            pytest.skip("installed Git does not support SHA-256 repositories")
    assert proc.returncode == 0, stderr.decode("utf-8", errors="replace")
    (tmp_path / "root.py").write_text("root = True\n", encoding="utf-8")
    proc = await asyncio.create_subprocess_exec("git", "add", ".", cwd=str(tmp_path))
    await proc.communicate()
    proc = await asyncio.create_subprocess_exec(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "root",
        cwd=str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    proc = await asyncio.create_subprocess_exec(
        "git",
        "rev-parse",
        "HEAD",
        cwd=str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    root_hash = stdout.decode().strip()
    assert len(root_hash) == (40 if object_format == "sha1" else 64)

    diff = await _diff_text(f"{root_hash}^..{root_hash}", tmp_path)

    assert diff.startswith("diff --git a/root.py b/root.py\n")
    assert "+root = True\n" in diff


@pytest.mark.asyncio
async def test_unsafe_ref_rejected() -> None:
    with pytest.raises(ValueError, match="Unsafe git ref rejected"):
        async for _ in stream_git_diff_fragments("--output=/tmp/x", cwd=Path("/tmp")):
            pass


@pytest.mark.asyncio
async def test_option_injection_rejected() -> None:
    """Git options must be rejected even though they pass the char regex."""
    for bad_ref in ("--cached", "--staged", "-p", "--no-index"):
        with pytest.raises(ValueError, match="Unsafe git ref rejected"):
            async for _ in stream_git_diff_fragments(bad_ref, cwd=Path("/tmp")):
                pass


@pytest.mark.asyncio
async def test_root_commit_retry_also_fails(tmp_path: Path) -> None:
    """If the empty-tree retry also fails, the error from the retry is raised."""
    commit_hash = "b" * 40
    procs = [
        StreamProcess(reads=[], exit_code=128, stderr=b"fatal: unknown revision bbbb^"),
        StreamProcess(reads=[], exit_code=128, stderr=b"fatal: not a git repository"),
    ]
    call_count = 0

    async def fake_exec(*args: object, **kwargs: object) -> StreamProcess:
        nonlocal call_count
        call_count += 1
        return procs.pop(0)

    with (
        patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
        patch(
            "chunkhound.core.git_diff.runner._empty_tree_oid",
            return_value="4b825dc642cb6eb9a060e54bf8d69288fbee4904",
        ),
    ):
        with pytest.raises(ValueError, match="not a git repository"):
            async for _ in stream_git_diff_fragments(
                f"{commit_hash}^..{commit_hash}", cwd=tmp_path
            ):
                pass

    assert call_count == 2


@pytest.mark.asyncio
async def test_non_root_failure_not_retried(tmp_path: Path) -> None:
    """Unrelated git failures (no 'unknown revision') are not retried."""
    commit_hash = "c" * 40
    call_count = 0

    async def fake_exec(*args: object, **kwargs: object) -> StreamProcess:
        nonlocal call_count
        call_count += 1
        return StreamProcess(
            reads=[], exit_code=128, stderr=b"fatal: bad object HEAD~999"
        )

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        with pytest.raises(ValueError, match="bad object"):
            async for _ in stream_git_diff_fragments(
                f"{commit_hash}^..{commit_hash}", cwd=tmp_path
            ):
                pass

    assert call_count == 1


@pytest.mark.asyncio
async def test_stream_caps_stderr_retained_in_memory(tmp_path: Path) -> None:
    """A pathologically large stderr payload must not be fully buffered."""
    huge_stderr = b"fatal: bad object HEAD~999 " + b"x" * (10 * _MAX_STDERR_BYTES)
    proc = StreamProcess(reads=[], exit_code=128, stderr=huge_stderr)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with pytest.raises(ValueError) as exc_info:
            async for _ in stream_git_diff_fragments("HEAD", cwd=tmp_path):
                pass

    message = str(exc_info.value)
    assert "fatal: bad object" in message
    assert len(message) <= _MAX_STDERR_BYTES + len("git diff failed: ")


@pytest.mark.asyncio
async def test_stream_stderr_drain_reaches_eof_past_the_cap(
    tmp_path: Path,
) -> None:
    """Draining must continue past the cap so git never blocks on a full pipe."""
    huge_stderr = b"x" * (10 * _MAX_STDERR_BYTES)
    proc = StreamProcess(reads=[], exit_code=128, stderr=huge_stderr)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with pytest.raises(ValueError):
            async for _ in stream_git_diff_fragments("HEAD", cwd=tmp_path):
                pass

    assert proc.stderr._payload == b""
    assert proc.stderr.read_calls > 1


@pytest.mark.asyncio
async def test_stream_buffered_reads_avoid_per_line_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Many ready lines should not spend the liveness budget on scheduling."""
    import chunkhound.core.git_diff.runner as runner_module

    monkeypatch.setattr(runner_module, "_GIT_DIFF_TIMEOUT_SECONDS", 0.05)
    proc = StreamProcess(reads=[b"diff --git a/a.py b/a.py\n" + b"+line\n" * 10_000])
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        diff = await _diff_text("HEAD", tmp_path)

    assert diff.count("+line\n") == 10_000


@pytest.mark.asyncio
async def test_stream_timeout_kills_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chunkhound.core.git_diff.runner as runner_module

    monkeypatch.setattr(runner_module, "_GIT_DIFF_TIMEOUT_SECONDS", 0.05)
    proc = StreamProcess(reads=[], hang_stdout=True, hang_stderr=True)
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with pytest.raises(TimeoutError, match="timed out"):
            async for _ in stream_git_diff_fragments("HEAD", cwd=tmp_path):
                pass
    assert proc.killed is True


@pytest.mark.asyncio
async def test_stream_cancel_kills_process(tmp_path: Path) -> None:
    proc = StreamProcess(reads=[], hang_stdout=True, hang_stderr=True)

    async def consume() -> None:
        async for _ in stream_git_diff_fragments("HEAD", cwd=tmp_path):
            pass

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert proc.killed is True


@pytest.mark.asyncio
async def test_stream_unexpected_io_kills_process(tmp_path: Path) -> None:
    proc = StreamProcess(reads=[], error=OSError("pipe closed"), hang_stderr=True)
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with pytest.raises(OSError, match="pipe closed"):
            async for _ in stream_git_diff_fragments("HEAD", cwd=tmp_path):
                pass
    assert proc.killed is True


@pytest.mark.asyncio
async def test_stream_budget_excludes_consumer_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow consumer (embedding a batch) must not trip git's liveness guard.

    The generator pauses at each yield while the caller embeds that fragment.
    That time is not spent waiting on git, so it must not count against the
    timeout — otherwise a large commit range kills its own git process.
    """
    import chunkhound.core.git_diff.runner as runner_module

    monkeypatch.setattr(runner_module, "_GIT_DIFF_TIMEOUT_SECONDS", 0.2)
    proc = StreamProcess(
        reads=[
            b"diff --git a/a.py b/a.py\n+first\n",
            b"diff --git a/b.py b/b.py\n+second\n",
        ]
    )
    fragments: list[str] = []
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        async for fragment in stream_git_diff_fragments("HEAD~1..HEAD", cwd=tmp_path):
            fragments.append(fragment)
            await asyncio.sleep(0.15)  # 2 x 0.15s = 0.3s > the 0.2s git budget

    assert "".join(fragments) == (
        "diff --git a/a.py b/a.py\n+first\ndiff --git a/b.py b/b.py\n+second\n"
    )


def _git(cwd: Path, *args: str) -> None:
    import subprocess

    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.mark.asyncio
async def test_real_git_ignores_user_diff_rewrites(tmp_path: Path) -> None:
    """color.ui / diff.external / textconv must not empty or rewrite git-history diffs (#470)."""
    import shutil
    import subprocess

    from chunkhound.core.git_diff.parser import parse_diff_to_chunks

    if shutil.which("git") is None:
        pytest.skip("git not available")

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "a@b")
    _git(repo, "config", "user.name", "a")
    (repo / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "1")
    (repo / "a.py").write_text("def a():\n    return 2\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "2")

    # Hostile local config that previously broke the parser / runner.
    _git(repo, "config", "color.ui", "always")
    _git(repo, "config", "diff.external", "/bin/false")
    attributes = repo / ".gitattributes"
    attributes.write_text("*.py diff=hostile\n", encoding="utf-8")
    converter = tmp_path / "hostile-textconv.sh"
    converter.write_text("#!/bin/sh\necho CONVERTED\n", encoding="utf-8")
    converter.chmod(0o755)
    _git(repo, "config", "diff.hostile.textconv", str(converter))

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    parent = subprocess.check_output(
        ["git", "rev-parse", "HEAD~1"], cwd=repo, text=True
    ).strip()
    raw = await _diff_text(f"{parent}..{head}", repo)

    assert "\x1b[" not in raw
    assert "CONVERTED" not in raw
    assert "diff --git" in raw
    assert "return 2" in raw

    chunks = parse_diff_to_chunks(raw)
    paths = {str(c.file_path) for c in chunks}
    assert "a.py" in paths
    assert any("return 2" in c.code for c in chunks)
