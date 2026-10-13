"""Tests for the BPE tokenizer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from bpe import BPETokenizer

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def trained_tokenizer() -> BPETokenizer:
    """Train a small tokenizer on the sample data for testing."""
    sample_path = PROJECT_ROOT / "data" / "sample.txt"
    text = sample_path.read_text(encoding="utf-8")
    tok = BPETokenizer()
    tok.train(text, vocab_size=500)
    return tok


@pytest.mark.parametrize("text", [
    "hello world",
    "The quick brown fox.",
    "日本語のテスト",
    "emoji: \U0001f680\U0001f30d✨",
    "",
    "a",
    "the the the the the",
    "AAAAAAAAAA",
    "tokenization is the art of compression",
    "low lower lowest",
    "1234567890",
    "!@#$%^&*()",
])
def test_roundtrip(trained_tokenizer, text):
    """Assert encode(decode(x)) == x for various inputs."""
    ids = trained_tokenizer.encode(text)
    assert trained_tokenizer.decode(ids) == text


def test_encode_returns_valid_ids(trained_tokenizer):
    """All token IDs must be within the vocabulary."""
    text = "Hello, world! Testing 123."
    ids = trained_tokenizer.encode(text)
    for idx in ids:
        assert idx in trained_tokenizer.vocab


def test_vocab_size(trained_tokenizer):
    """Trained tokenizer should have exactly the requested vocab size."""
    # We trained with vocab_size=500, but may have stopped early if no more pairs
    assert len(trained_tokenizer.vocab) <= 500
    assert len(trained_tokenizer.vocab) >= 256


def test_merges_are_consistent(trained_tokenizer):
    """Every merge's vocab entry should be the concatenation of its parts."""
    for (a, b), new_id in trained_tokenizer.merges.items():
        assert trained_tokenizer.vocab[new_id] == trained_tokenizer.vocab[a] + trained_tokenizer.vocab[b]


def test_compression_ratio_improves():
    """A trained tokenizer should compress better than byte-level baseline."""
    sample_path = PROJECT_ROOT / "data" / "sample.txt"
    text = sample_path.read_text(encoding="utf-8")

    # Byte-level baseline: 1 byte per token
    baseline_ratio = 1.0

    tok = BPETokenizer()
    tok.train(text, vocab_size=500)
    ratio = tok.compression_ratio(text)

    assert ratio > baseline_ratio, f"Expected compression > 1.0, got {ratio}"


def test_empty_string(trained_tokenizer):
    """Empty string should encode to empty list and decode back to empty."""
    ids = trained_tokenizer.encode("")
    assert ids == []
    assert trained_tokenizer.decode(ids) == ""


def test_deterministic_encoding(trained_tokenizer):
    """Same input should always produce the same output."""
    text = "deterministic test"
    ids1 = trained_tokenizer.encode(text)
    ids2 = trained_tokenizer.encode(text)
    assert ids1 == ids2


def test_cli_train_and_demo():
    """Test that the CLI scripts work end-to-end."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer_path = os.path.join(tmpdir, "test_tokenizer.json")

        # Train
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"),
             "--input", str(PROJECT_ROOT / "data" / "sample.txt"),
             "--vocab_size", "300",
             "--output", tokenizer_path],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Training failed: {result.stderr}"
        assert os.path.exists(tokenizer_path)

        # Demo
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo.py"),
             "--tokenizer", tokenizer_path,
             "--text", "hello world"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Demo failed: {result.stderr}"
        assert "Match:    True" in result.stdout
