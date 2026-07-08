from pathlib import Path

from chunkhound.services import batch_processor as bp


def test_process_file_batch_timeout(monkeypatch, tmp_path: Path):
    # Create a small file
    f = tmp_path / "x.py"
    f.write_text("print('x')\n")

    # Force timeout path to be used regardless of size by setting min size 0
    # Note: code treats 0 as falsy; use -1 to ensure timeout path applies
    cfg = {"per_file_timeout_seconds": 0.01, "per_file_timeout_min_size_kb": -1}

    # Monkeypatch the worker timeout function to simulate timeout
    monkeypatch.setattr(
        bp, "_parse_file_with_timeout", lambda *a, **k: ("timeout", None)
    )

    results = bp.process_file_batch([(f, None)], cfg)
    assert results and results[0].status == "skipped"
    assert results[0].error == "timeout"
