
# BPE Tokenizer from Scratch

A from-scratch implementation of **Byte-Pair Encoding (BPE)** tokenization — the same algorithm used by GPT-2, GPT-3, GPT-4, and most modern LLMs to convert raw text into token IDs.

This project implements the full pipeline: training the vocabulary, encoding text, and decoding it back — with no tokenizer libraries (no HuggingFace `tokenizers`, no `tiktoken`). Just Python + NumPy.

---

## 📌 Why this project exists

Most people learn about LLMs starting at the transformer. But the transformer never sees text — it sees **integers**. The tokenizer is the bridge, and it's one of the most under-appreciated parts of the pipeline.

I built this to understand:
- How subword vocabularies are learned from raw text
- Why "tokenization" causes weird LLM behaviors (e.g., why models struggle with counting letters)
- How byte-level BPE handles arbitrary Unicode without an `<UNK>` token

---

## 🧠 What is BPE?

Byte-Pair Encoding is a compression algorithm adapted for tokenization. The core idea:

1. Start with a base vocabulary (here: the **256 raw bytes**).
2. Count the most frequent adjacent pair of tokens in the training corpus.
3. Merge that pair into a new token.
4. Repeat until you reach the target vocabulary size.

**Example** — training on the word `"lower"` appearing frequently:

```
l o w e r        (initial: 5 tokens)
l o w er         (merge 'e','r' -> 'er')
l ow er          (merge 'o','w' -> 'ow')
low er           (merge 'l','ow' -> 'low')
lower            (merge 'low','er' -> 'lower')
```

Frequent words become single tokens; rare words stay split into subwords. This is why the model can handle any word — even ones it's never seen.

---

## ✨ Features

- **Byte-level BPE** — operates on raw UTF-8 bytes, so it can encode *any* string (including emoji and non-Latin scripts) with zero unknown tokens.
- **Trainable** — learns a vocabulary from any text corpus.
- **Encode / decode** — round-trips text ↔ token IDs losslessly.
- **Save / load** — persists the learned merges and vocab to disk.
- **Visualization** — plots the most frequent merges and token length distribution.
- **No dependencies** beyond NumPy and Matplotlib.

---

## 📂 Project structure

```
bpe-tokenizer/
├── bpe/
│   ├── __init__.py
│   ├── tokenizer.py       # BPETokenizer class (train, encode, decode)
│   ├── trainer.py         # merge-counting and vocab-building logic
│   └── utils.py           # byte<->str helpers, save/load
├── scripts/
│   ├── train.py           # CLI: train a tokenizer on a text file
│   └── demo.py            # CLI: encode/decode a sample string
├── data/
│   └── sample.txt         # small corpus for testing
├── notebooks/
│   └── visualize_merges.ipynb
├── tests/
│   └── test_roundtrip.py  # asserts encode(decode(x)) == x
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### Install

```bash
git clone https://github.com/<your-username>/bpe-tokenizer.git
cd bpe-tokenizer
pip install -r requirements.txt
```

### Train a tokenizer

```bash
python scripts/train.py \
  --input data/sample.txt \
  --vocab_size 1000 \
  --output tokenizer.json
```

### Use it in Python

```python
from bpe import BPETokenizer

tok = BPETokenizer.load("tokenizer.json")

text = "Hello, world! 🌍"
ids = tok.encode(text)
print(ids)             # [72, 101, 108, 108, 111, ...]

print(tok.decode(ids)) # "Hello, world! 🌍"
```

---

## 🔧 How it works

### 1. Training

```python
def train(self, text: str, vocab_size: int):
    # 1. Convert text to a list of byte IDs (0-255)
    ids = list(text.encode("utf-8"))

    # 2. Repeatedly find and merge the most frequent pair
    merges = {}
    while len(self.vocab) < vocab_size:
        counts = count_pairs(ids)
        pair = max(counts, key=counts.get)
        new_id = len(self.vocab)

        # record the merge and apply it everywhere
        merges[pair] = new_id
        self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        ids = merge_pair(ids, pair, new_id)

    return merges
```

### 2. Encoding

Encoding applies the learned merges **in the order they were learned** (this is critical — BPE merges are order-dependent):

```python
def encode(self, text: str) -> list[int]:
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        # find the pair with the lowest merge index (i.e., learned earliest)
        pair = min(
            zip(ids, ids[1:]),
            key=lambda p: self.merges.get(p, float("inf")),
            default=None,
        )
        if pair not in self.merges:
            break
        ids = merge_pair(ids, pair, self.merges[pair])
    return ids
```

### 3. Decoding

Trivial — every token maps back to a byte string:

```python
def decode(self, ids: list[int]) -> str:
    return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
```

---

## 📊 Results

Trained on `<your corpus>` with `vocab_size = 1000`:

| Metric | Value |
|---|---|
| Training time | `<X>` seconds |
| Final vocab size | 1000 |
| Compression ratio (bytes/token) | `<Y>` |
| Round-trip accuracy | 100% (see `tests/`) |

**Most frequent merges learned:**

```
1. ('e', 'r')     -> 'er'
2. ('t', 'h')     -> 'th'
3. ('i', 'n')     -> 'in'
4. ('o', 'u')     -> 'ou'
5. ('a', 'n')     -> 'an'
...
```

**Token length distribution:**

![token length histogram](docs/token_lengths.png)

*Most tokens are 1–4 bytes; only a handful of long tokens represent common words.*

---

## 🧪 Tests

```bash
pytest tests/
```

The key invariant tested is **lossless round-tripping**:

```python
@pytest.mark.parametrize("text", [
    "hello world",
    "The quick brown fox.",
    "日本語のテスト",
    "emoji: 🚀🌍✨",
    "",
])
def test_roundtrip(text):
    tok = BPETokenizer.load("tokenizer.json")
    assert tok.decode(tok.encode(text)) == text
```

---

## 📈 What I learned

- **Byte-level BPE never needs `<UNK>`.** Because the base vocab is all 256 bytes, every possible input is representable. This is why GPT-2+ uses it.
- **Merges are order-dependent.** Encoding must apply merges in the same order they were learned, or you get different tokenizations.
- **Tokenization explains a lot of LLM weirdness.** Ask GPT why "strawberry" has two r's and watch it struggle — it's because the tokenizer probably has `"straw"` and `"berry"` as single tokens, so the model never sees the individual letters.
- **Compression = understanding.** A good tokenizer compresses common patterns into single tokens, which means the model spends its capacity on *meaning*, not on spelling.

---

## 🔭 Future work

- [ ] Add a `regex`-based pre-tokenizer (GPT-2 uses a specific pattern to split text before BPE)
- [ ] Support training on multiple files / streaming
- [ ] Add a fast Rust or Cython implementation for large corpora
- [ ] Compare my tokenizer's output to GPT-2's tokenizer on the same text
- [ ] Add special tokens (`<|endoftext|>`, `<|pad|>`, etc.)

---

## 📚 References

- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (2016) — the paper that introduced BPE to NLP
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019) — introduced byte-level BPE
- Karpathy, *Let's build the GPT Tokenizer* (video) — excellent walkthrough of the same ideas
- HuggingFace `tokenizers` library — production reference implementation

---

## 📄 License

MIT — use it, learn from it, fork it.
