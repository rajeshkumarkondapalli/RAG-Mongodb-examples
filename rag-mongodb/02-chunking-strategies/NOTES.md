# Notes — Before You Run Example 2 (Chunking Strategies)

---

## What This Example Does

Takes the same 6 sample documents and splits them into chunks using 4 different
strategies. Shows you the difference in chunk count, size, and readability.
Then ingests the chunks into MongoDB so you can compare retrieval quality.

---

## Prerequisites

- [ ] Completed Example 1 setup (`.env` file, Atlas cluster, OpenAI key)
- [ ] `pip install -r requirements.txt` already run
- [ ] You understand what embeddings are (read Example 1 NOTES first)

> `compare.py` does **not** need MongoDB or OpenAI — you can run it offline.
> `ingest_with_chunks.py` and `main.py` do need both.

---

## The Core Problem This Solves

Imagine you have a 10-page PDF. You cannot embed the whole thing as one vector —
it would exceed the model's token limit and the resulting vector would be too
broad to be useful for retrieval.

So you split it into chunks first:

```
Full document (10 pages, ~8000 words)
         │
         ▼  chunking
 ┌──────┬──────┬──────┬──────┬──────┐
 │chunk1│chunk2│chunk3│chunk4│chunk5│  ...more chunks
 └──────┴──────┴──────┴──────┴──────┘
         │
         ▼  embed each chunk separately
 [vec1] [vec2] [vec3] [vec4] [vec5]
         │
         ▼  store in MongoDB
   {text: chunk1, embedding: vec1}
   {text: chunk2, embedding: vec2}
   ...
```

At query time you search for chunks, not whole documents.

---

## The Overlap Concept (Important)

Chunks are not cut at exact boundaries — they **overlap** with their neighbors:

```
chunk_size=500, overlap=100

│◄─────────── 500 chars ────────────►│
                    │◄────── 500 chars ──────────────►│
                    │◄100►│  ← shared content (overlap)
```

**Why overlap?** A sentence at the end of chunk 1 that continues at the start
of chunk 2 would lose its context without overlap. Overlap keeps those boundary
sentences in both chunks.

**Rule of thumb:** overlap = 10–20% of chunk_size

---

## What Each Strategy Does (Plain English)

### Strategy 1 — Fixed-size (`fixed_size_chunks`)

Cuts every N characters, no matter what.

```
Input:  "MongoDB Atlas supports cosine. It also supports euclidean distance."
         ^─────────────────── 50 chars ──────────────────^
Output: "MongoDB Atlas supports cosine. It also suppo"   ← cuts mid-word!
        "rts euclidean distance."
```

- Simple, fast
- Does not care about words, sentences, or paragraphs
- **Best for:** uniform data like logs or CSV rows where structure doesn't matter

### Strategy 2 — Recursive (`recursive_chunks`) ← recommended default

Tries to split on `\n\n` first, then `\n`, then `. `, then spaces.
Only falls back to character splitting if nothing else fits.

```
Input:  Paragraph 1\n\nParagraph 2\n\nParagraph 3
Output: ["Paragraph 1", "Paragraph 2", "Paragraph 3"]   ← respects paragraphs
```

- Preserves natural boundaries
- Chunk sizes vary but stay close to target
- **Best for:** most documents (docs, articles, web pages, PDFs)

### Strategy 3 — Sentence-based (`sentence_chunks`)

Groups sentences into sliding windows:

```
sentences_per_chunk=3, overlap_sentences=1

S1 S2 S3 | S3 S4 S5 | S5 S6 S7 ...
          ↑ overlap  ↑ overlap
```

- Clean, readable chunks — each chunk is always a complete thought
- Chunk sizes vary significantly if sentence lengths vary
- **Best for:** narrative text, articles, customer support transcripts

### Strategy 4 — Token-based (`token_chunks`)

Counts model tokens (not characters) per chunk.

```
"Hello world"  →  ["Hello", "world"]  →  2 tokens (not 11 characters)
```

- Guarantees you never exceed the embedding model's token limit
- Slower (must tokenize first)
- **Best for:** when you're hitting token limit errors with other strategies

---

## How to Read the Comparison Output

```
Strategy: Recursive (500 chars, 100 overlap)
  Chunks produced  : 6        ← more chunks = finer granularity
  Avg size (chars) : 437      ← average characters per chunk
  Min / Max (chars): 98 / 500 ← smallest and largest chunk
  Approx avg tokens: 109      ← ~chars/4 (rough estimate for English)
```

**What to look for:**
- High min/max variance = inconsistent chunks (may hurt retrieval)
- Very small min = tiny orphan chunks that won't retrieve well
- Very large max = chunks that may exceed token limits

---

## Chunk Size Decision Guide

Answer these questions to pick your chunk size:

**1. What's my embedding model's token limit?**
- `text-embedding-3-small`: 8191 tokens
- OpenAI recommends keeping chunks well under this (~512 tokens / ~2000 chars)

**2. How long are my documents?**
- Short FAQs / Q&A → 150–300 chars
- Technical docs → 300–500 chars
- Long articles / books → 500–800 chars

**3. Do I need precision or context?**
- Need precise answers? → smaller chunks (300 chars)
- Need surrounding context? → larger chunks (600 chars)

---

## What Changes When You Switch Strategies

Run `compare.py`, then change the strategy in `ingest_with_chunks.py` and
re-run. Then ask the same question and compare the retrieved chunks.

```python
# In ingest_with_chunks.py, try changing this line:
from chunkers import recursive_chunks   # default
# to:
from chunkers import sentence_chunks    # try this
# or:
from chunkers import fixed_size_chunks  # try this
```

Then run the query from Example 1 and notice:
- Which chunks get retrieved?
- Are the retrieved chunks complete thoughts or cut mid-sentence?
- Does the LLM give a better or worse answer?

---

## Things That Can Go Wrong

| Error | Cause | Fix |
|---|---|---|
| `ImportError: nltk` | NLTK not installed | `pip install nltk` then `python -c "import nltk; nltk.download('punkt')"` |
| `ImportError: sentence_transformers` | Missing package | `pip install sentence-transformers` |
| Very few chunks produced | chunk_size too large for the text | Reduce `chunk_size` |
| Single massive chunk | Splitter found no split points | Check that `separators` match your text format |

---

## Key Insight After Running This

The number that matters most is **not** how many chunks you produce —
it's whether the **retrieved chunks** contain the information needed to answer
the question. Run the same question against different strategies and compare
the retrieved chunk content. That is the real measure of chunking quality.
