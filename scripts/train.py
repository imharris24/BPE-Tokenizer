"""CLI: train a BPE tokenizer on a text file."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from bpe import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
    parser.add_argument("--input", required=True, help="Path to training text file")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Target vocabulary size")
    parser.add_argument("--output", default="tokenizer.json", help="Output path for trained tokenizer")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 1

    text = input_path.read_text(encoding="utf-8")
    print(f"Training on {input_path} ({len(text):,} chars, {len(text.encode('utf-8')):,} bytes)")
    print(f"Target vocab size: {args.vocab_size}")

    tokenizer = BPETokenizer()

    start = time.time()
    merges = tokenizer.train(text, args.vocab_size)
    elapsed = time.time() - start

    tokenizer.save(args.output)

    print(f"\nTraining complete in {elapsed:.2f}s")
    print(f"Final vocab size: {len(tokenizer.vocab)}")
    print(f"Merges learned: {len(merges)}")
    print(f"Compression ratio: {tokenizer.compression_ratio(text):.2f} bytes/token")
    print(f"Saved to: {args.output}")

    # Show top 10 most frequent merges (by merge order = frequency rank)
    print("\nTop 10 merges learned:")
    for i, (pair, new_id) in enumerate(list(merges.items())[:10]):
        a_str = tokenizer.vocab[pair[0]].decode("utf-8", errors="replace")
        b_str = tokenizer.vocab[pair[1]].decode("utf-8", errors="replace")
        merged = tokenizer.vocab[new_id].decode("utf-8", errors="replace")
        print(f"  {i+1}. '{a_str}' + '{b_str}' -> '{merged}'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
