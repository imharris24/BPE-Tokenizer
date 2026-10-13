"""Merge-counting and vocabulary-building logic for BPE training."""

from __future__ import annotations

from collections import Counter


def count_pairs(ids: list[int]) -> Counter:
    """Count adjacent token pairs in a sequence."""
    return Counter(zip(ids, ids[1:]))


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace all occurrences of `pair` with `new_id` in `ids`."""
    result = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result


def train_bpe(ids: list[int], vocab_size: int) -> tuple[dict[tuple[int, int], int], dict[int, bytes]]:
    """
    Train BPE merges on a list of byte IDs.

    Returns:
        merges: dict mapping (token_a, token_b) -> new_token_id
        vocab:  dict mapping token_id -> byte string
    """
    if vocab_size < 256:
        raise ValueError(f"vocab_size must be >= 256, got {vocab_size}")

    # Base vocabulary: all 256 single bytes
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: dict[tuple[int, int], int] = {}

    # Work on a copy so the caller's list isn't mutated
    ids = list(ids)

    while len(vocab) < vocab_size:
        counts = count_pairs(ids)
        if not counts:
            break

        # Find the most frequent pair; break ties by lowest token IDs for determinism
        pair = min(counts, key=lambda p: (-counts[p], p[0], p[1]))
        new_id = len(vocab)

        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        ids = merge_pair(ids, pair, new_id)

    return merges, vocab
