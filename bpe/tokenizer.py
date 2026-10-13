"""Byte-level BPE Tokenizer — train, encode, decode."""

from __future__ import annotations

import json
import time
from pathlib import Path

from bpe.trainer import merge_pair, train_bpe


class BPETokenizer:
    """A byte-level BPE tokenizer.

    Base vocabulary is the 256 single bytes. Training learns merge rules
    that combine frequent adjacent pairs into new tokens.
    """

    def __init__(self):
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: dict[tuple[int, int], int] = {}

    def train(self, text: str, vocab_size: int) -> dict[tuple[int, int], int]:
        """Train the tokenizer on `text` to reach `vocab_size` tokens."""
        ids = list(text.encode("utf-8"))
        self.merges, self.vocab = train_bpe(ids, vocab_size)
        return self.merges

    def train_file(self, path: str | Path, vocab_size: int) -> dict[tuple[int, int], int]:
        """Train on the contents of a text file."""
        text = Path(path).read_text(encoding="utf-8")
        return self.train(text, vocab_size)

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            # Find the pair with the lowest merge index (learned earliest)
            best_pair = None
            best_idx = float("inf")
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                idx = self.merges.get(pair, float("inf"))
                if idx < best_idx:
                    best_idx = idx
                    best_pair = pair

            if best_pair is None or best_idx == float("inf"):
                break

            ids = merge_pair(ids, best_pair, self.merges[best_pair])

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        """Return the byte string for a given token ID."""
        return self.vocab[token_id]

    def token_str(self, token_id: int) -> str:
        """Return the decoded string for a given token ID."""
        return self.vocab[token_id].decode("utf-8", errors="replace")

    def compression_ratio(self, text: str) -> float:
        """Return bytes-per-token ratio for the given text."""
        ids = self.encode(text)
        if not ids:
            return 0.0
        return len(text.encode("utf-8")) / len(ids)

    def save(self, path: str | Path) -> None:
        """Save tokenizer vocab and merges to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab_size": len(self.vocab),
            "merges": {f"{a},{b}": idx for (a, b), idx in self.merges.items()},
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "BPETokenizer":
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
