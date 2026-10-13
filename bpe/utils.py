"""Utility helpers for byte<->str conversion and tokenizer persistence."""

from __future__ import annotations

import json
from pathlib import Path

from bpe.tokenizer import BPETokenizer


def bytes_to_str(byte_ids: list[int]) -> str:
    """Decode a list of byte IDs (0-255) into a string."""
    return bytes(byte_ids).decode("utf-8", errors="replace")


def str_to_bytes(text: str) -> list[int]:
    """Encode a string into a list of byte IDs (0-255)."""
    return list(text.encode("utf-8"))


def save_tokenizer(tokenizer: BPETokenizer, path: str | Path) -> None:
    """Save tokenizer vocab and merges to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "vocab_size": len(tokenizer.vocab),
        "merges": {f"{a},{b}": idx for (a, b), idx in tokenizer.merges.items()},
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_tokenizer(path: str | Path) -> BPETokenizer:
    """Load a tokenizer from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    tokenizer = BPETokenizer()
    # Base vocab: bytes 0-255
    for i in range(256):
        tokenizer.vocab[i] = bytes([i])

    # Rebuild merges in order of their assigned IDs
    merges_raw = []
    for key, new_id in data["merges"].items():
        a, b = key.split(",")
        merges_raw.append((int(a), int(b), new_id))
    merges_raw.sort(key=lambda x: x[2])

    for a, b, new_id in merges_raw:
        tokenizer.merges[(a, b)] = new_id
        tokenizer.vocab[new_id] = tokenizer.vocab[a] + tokenizer.vocab[b]

    return tokenizer
