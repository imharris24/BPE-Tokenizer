"""CLI: encode/decode a sample string with a trained tokenizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bpe import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Demo: encode and decode text with a trained BPE tokenizer")
    parser.add_argument("--tokenizer", default="tokenizer.json", help="Path to trained tokenizer JSON")
    parser.add_argument("--text", default="Hello, world! 🌍", help="Text to encode/decode")
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"Error: tokenizer not found: {tokenizer_path}")
        print("Train one first: python scripts/train.py --input data/sample.txt --output tokenizer.json")
        return 1

    tok = BPETokenizer.load(tokenizer_path)

    print(f"Tokenizer: {tokenizer_path}")
    print(f"Vocab size: {len(tok.vocab)}")
    print(f"Merges: {len(tok.merges)}")
    print()

    ids = tok.encode(args.text)
    decoded = tok.decode(ids)

    print(f"Input:    {args.text!r}")
    print(f"IDs:      {ids}")
    print(f"Tokens:   {[tok.token_str(i) for i in ids]}")
    print(f"Decoded:  {decoded!r}")
    print(f"Match:    {decoded == args.text}")
    print(f"Compression: {tok.compression_ratio(args.text):.2f} bytes/token")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
