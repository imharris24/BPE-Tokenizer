"""Utility helpers for byte<->str conversion."""

from __future__ import annotations


def bytes_to_str(byte_ids: list[int]) -> str:
    """Decode a list of byte IDs (0-255) into a string."""
    return bytes(byte_ids).decode("utf-8", errors="replace")


def str_to_bytes(text: str) -> list[int]:
    """Encode a string into a list of byte IDs (0-255)."""
    return list(text.encode("utf-8"))
