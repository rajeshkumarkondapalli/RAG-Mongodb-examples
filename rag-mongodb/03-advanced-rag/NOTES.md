# Notes — Before You Run Example 3 (Advanced RAG)

---

## Hard Requirements Before Running Anything Here

- [ ] You have run Example 1 (`01-basic-rag/ingest.py`) at least once
- [ ] The Atlas vector index `vector_index` shows **Status: Active** in Atlas UI
- [ ] Your `.env` file is filled in with a working `MONGODB_URI` and `OPENAI_API_KEY`
- [ ] You understand the basic RAG chain from Example 1

If you skip these, every script here will fail.

---

## What Makes This "Advanced"

Example 1 is a straight line: question → search → answer.

These scripts solve the problems you hit in production:

| Problem | Script That Solves It |
|---|---|
| "I only want to search docs from one department" | `filtered_retrieval.py` |
| "My question is ambiguous and vector search misses relevant docs" | `multi_query.py` |
| "Keywords in my query aren't found by semantic search" | `hybrid_search.py` |
| "Users ask follow-up questions that need conversation context" | `conversational_rag.py` |

---

## Script 1 — `filtered_retrieval.py`

### What it does
Adds a MongoDB filter (`pre_filter`) to the vector search so it only looks inside
a subset of documents before ranking by similarity.

### The key concept: pre_filter vs post-filter

```
pre_filter  (runs IN Atlas, BEFORE similarity scoring)
    → Atlas only considers matching documents when ranking
    → Fast. Runs on the server. Doesn't waste similarity budget.

post-filter (runs in Python, AFTER similarity scoring)
    → You retrieve k results, then discard ones that don't match
    → Wastes k slots. You might return fewer than k results.
```

Always use `pre_filter` when you can.

### What you'll see
```
--- No filter (all documents) ---
  [ai | ai_concepts] Introduction to RAG
  [database | mongodb_docs] MongoDB Atlas Vector Search Overview

--- AI category only ---
  [ai | ai_concepts] Introduction to RAG
  [ai | ai_concepts] Prompt Engineering for RAG

--- Database category only ---
  [database | mongodb_docs] MongoDB Atlas Vector Search Overview
  [database | mongodb_docs] Hybrid Search in MongoDB
```

### Before running
The `metadata.category` and `metadata.source` fields must be indexed as
**filter** fields in your Atlas vector index definition. The index was created
with these in `shared/mongodb_utils.py`:

```python
{"type": "filter", "path": "metadata.category"},
{"type": "filter", "path": "metadata.source"},
```

If you created your index manually without these fields, filtering will silently
return zero results. Delete the index in Atlas UI and re-run `ingest.py`.

---

## Script 2 — `multi_query.py`

### What it does
Before searching, it asks the LLM to rephrase your question 3 different ways.
Then runs 3 separate vector searches and deduplicates the results.

### Why this matters
Vector search is sensitive to wording. The same concept phrased differently can
return completely different documents.

```
Your question:
  "How do I improve search relevance?"

Vector search alone might miss: "best practices for retrieval quality"

Multi-query generates and searches all of these:
  1. "How do I improve search relevance?"
  2. "What techniques enhance document retrieval quality?"
  3. "Best practices for vector search ranking?"
  4. "How to get better results from semantic search?"

→ Finds more relevant documents → better answers
```

### Cost warning
This script makes **4 LLM calls per question** (1 to generate variants + 3 to
embed and search). It's ~3–4x more expensive than a single query.

Use it when answer quality matters more than cost. For simple lookups, stick
with basic retrieval from Example 1.

### What you'll see
```
Query: How can I improve search relevance in my application?

Multi-query retrieved 5 unique documents:

  - [database] MongoDB Atlas Vector Search Overview
    MongoDB Atlas Vector Search allows you to store vector embeddings...

  - [ai] Introduction to RAG
    Retrieval-Augmented Generation (RAG) is an AI technique...
```

---

## Script 3 — `hybrid_search.py`

### What it does
Runs **two searches in parallel**:
1. Vector search (semantic similarity)
2. Full-text Atlas Search (BM25 keyword matching)

Then merges both ranked lists using **Reciprocal Rank Fusion (RRF)**.

### You need TWO indexes for this

| Index | Type | Purpose |
|---|---|---|
| `vector_index` | vectorSearch | Semantic similarity |
| `fulltext_index` | search | BM25 keyword matching |

The vector index was created in Example 1. The full-text index is created
by calling `create_fulltext_index(collection)` inside this script.

**Wait for both indexes to be Active before querying.**

### What RRF does (the math, simply)

Each result list is ranked 1, 2, 3... A document's RRF score is:

```
score = 1/(rank_in_vector_list + 60) + 1/(rank_in_text_list + 60)
```

A document that appears #1 in both lists gets the highest possible combined score.
A document only in one list gets a lower score.

The 60 is a constant that prevents top-ranked documents from dominating too much.

### When to use hybrid vs pure vector

```
Query: "What is semantic search?"
→ Use vector search. It's conceptual. Keywords won't help.

Query: "Does Atlas support $vectorSearch in aggregation pipelines?"
→ Use hybrid. The exact token "$vectorSearch" won't match semantically.

Query: "List the distance metrics: cosine, euclidean, dot product"
→ Use hybrid. Exact terms matter here.
```

### Atlas tier requirement
Atlas Search (full-text) requires **M10 or higher** cluster in production.
Free tier (M0) does support Atlas Search but with limitations.
If `create_fulltext_index` fails, check your cluster tier in Atlas UI.

---

## Script 4 — `conversational_rag.py`

### What it does
Maintains a multi-turn conversation. Each question is aware of what was asked before.

### The two-step architecture

```
Step 1 — Condense:
  History: "Q: What is Atlas? A: Atlas is a cloud database..."
  Current: "What distance metrics does it support?"
         ↓ LLM condenses these into:
  "What distance metrics does MongoDB Atlas Vector Search support?"

Step 2 — Retrieve + Answer:
  The condensed question is used for vector search (not the raw follow-up)
  Retrieved docs + full history → LLM → grounded answer
```

### Why the condensation step is necessary

Without condensing, the follow-up question "What distance metrics does it support?"
would be sent to the vector store as-is. "It" has no meaning to the vector store —
it doesn't know "it" refers to Atlas from the previous turn. The condensation step
resolves pronouns and references before retrieval.

### Cost warning
Every turn makes **2 LLM calls**: one to condense, one to answer. A 10-turn
conversation = 20 LLM calls. Budget accordingly.

On the first turn with no history, the condensation step is skipped (the code
checks `if inputs.get("chat_history")`), so you only pay for one call.

### The session_id
Each conversation needs a unique `session_id`. In the demo it's hardcoded as
`"workshop-demo"`. In a real app this would be a user ID or chat ID.

```python
chain.invoke(
    {"question": "What is Atlas?"},
    config={"configurable": {"session_id": "user-42"}}   ← unique per user
)
```

### Memory is in-process only
The `ChatMessageHistory` store in this example lives in a Python `dict`. It is
lost when the process restarts. For production, replace it with a
MongoDB-backed history store so conversations survive restarts.

---

## Running All Scripts Together (`main.py`)

`main.py` runs all four patterns in sequence. Before running it:

1. Make sure documents are ingested: `cd ../01-basic-rag && python ingest.py`
2. Wait for `vector_index` to show Active in Atlas UI (or wait 60s)
3. For hybrid search: `hybrid_search.py` will call `create_fulltext_index()` automatically — wait another 60s after it runs before hybrid search queries work

---

## Approximate Cost for Running All Scripts

| Script | LLM Calls | Approx Cost |
|---|---|---|
| `filtered_retrieval.py` | 4 (embed only, no LLM) | < $0.001 |
| `multi_query.py` | ~4 (1 generate + 3 embed) | ~$0.01 |
| `hybrid_search.py` | 1 embed call | < $0.001 |
| `conversational_rag.py` | 7 (4 turns × ~2 minus first) | ~$0.02 |
| **Total** | | **~$0.03** |

---

## Concepts You Should Understand After Running This

- Why pre-filtering is better than post-filtering for scoped retrieval
- How RRF combines two ranked lists into one without needing a shared score scale
- Why multi-query costs more and when that trade-off is worth it
- What "condensing" does in a conversational chain and why it's needed
- The difference between in-memory session history and persistent history
