# Notes — Before You Run Example 1 (Basic RAG)

Read this top to bottom before running any script in this folder.

---

## What This Example Does

1. Takes 6 sample documents from `shared/sample_data.py`
2. Sends each document to OpenAI to get a 1536-number embedding vector
3. Stores the text + vector + metadata in your MongoDB Atlas collection
4. Creates a vector search index in Atlas
5. Answers questions by finding the closest documents to your question vector

---

## Checklist Before Running

- [ ] Python 3.10 or higher installed (`python --version`)
- [ ] Dependencies installed (`pip install -r requirements.txt` from `rag-mongodb/`)
- [ ] `.env` file created from `.env.example` with real values filled in
- [ ] MongoDB Atlas cluster is running and reachable
- [ ] OpenAI API key is active and has available credits

---

## What You Need to Set Up First

### MongoDB Atlas (free tier is fine)

1. Go to [cloud.mongodb.com](https://cloud.mongodb.com) and create a free cluster
2. Under **Security → Database Access**: create a user with read/write access
3. Under **Security → Network Access**: add your IP address (or `0.0.0.0/0` for testing)
4. Click **Connect → Drivers** and copy the connection string
5. Paste it into your `.env` as `MONGODB_URI`, replacing `<username>` and `<password>`

Your `.env` should look like:
```
MONGODB_URI=mongodb+srv://myuser:mypassword@cluster0.abc123.mongodb.net/?retryWrites=true&w=majority
OPENAI_API_KEY=sk-proj-...
DB_NAME=rag_workshop
COLLECTION_NAME=documents
```

### OpenAI API Key

- Get one from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- This example calls:
  - `text-embedding-3-small` (embeddings) — very cheap, ~$0.0001 per 1K tokens
  - `gpt-4o-mini` (answers) — cheap, ~$0.0002 per 1K input tokens
- Running `main.py` costs roughly **$0.01–0.05** total

---

## Important: The 60-Second Wait

After `ingest.py` runs, it tells Atlas to build the vector index.
**Atlas builds the index asynchronously** — it is not instant.

```
ingest.py finishes → Atlas starts building → ~60 seconds → index is ready → you can query
```

If you query before the index is built you will get an error like:
```
OperationFailure: Index 'vector_index' not found
```

`main.py` already includes a `time.sleep(60)` for this.
If you run `ingest.py` and `rag_chain.py` manually, wait 60 seconds between them.

You can check index status in Atlas UI:
**Atlas → Your Cluster → Search → vector_index → Status: Active**

---

## What Each File Does (Plain English)

### `ingest.py`
- Reads the 6 sample documents
- Sends each to OpenAI: `"MongoDB Atlas is a cloud database..."` → `[0.012, -0.045, ...]`
- Saves `{ text, embedding, metadata }` into MongoDB
- Tells Atlas to build a vector search index on the `embedding` field

You only need to run this **once**. Re-running it wipes and re-creates the data.

### `rag_chain.py`
- Connects to the vector store (doesn't store anything, just reads)
- Builds a chain: `question → embed question → find similar docs → fill prompt → ask LLM → answer`
- Returns the chain object — you invoke it with a question string

### `main.py`
- Calls `ingest.py` → waits 60s → calls `rag_chain.py` → asks 4 sample questions
- This is the easiest entry point; run this first

---

## Expected Output

```
=== Step 1: Ingesting documents ===

Deleted 0 documents from 'documents'.
Prepared 6 documents for ingestion.
Embedding and storing documents in MongoDB Atlas...
Successfully stored 6 documents.

Vector search index 'vector_index' creation triggered.
Wait ~60 seconds for Atlas to build the index before querying.

Waiting 60 seconds for Atlas to finish building the vector index...

=== Step 2: Building RAG chain ===

=== Step 3: Answering questions ===

Q: What is MongoDB Atlas Vector Search and what distance metrics does it support?
  Retrieved sources:
    - MongoDB Atlas Vector Search Overview
    - LangChain and MongoDB Integration
    - Hybrid Search in MongoDB
A: MongoDB Atlas Vector Search allows you to store vector embeddings ...
----------------------------------------------------------------------
```

---

## Things That Can Go Wrong

| Error | Cause | Fix |
|---|---|---|
| `Authentication failed` | Wrong username/password in URI | Double-check `.env` MONGODB_URI |
| `connection timed out` | IP not whitelisted | Add your IP in Atlas Network Access |
| `Index 'vector_index' not found` | Querying before index is ready | Wait 60 seconds, check Atlas UI |
| `openai.AuthenticationError` | Invalid API key | Check OPENAI_API_KEY in `.env` |
| `openai.RateLimitError` | Too many requests / no credits | Check your OpenAI account billing |
| `ModuleNotFoundError` | Missing dependency | Run `pip install -r requirements.txt` |

---

## Concepts You Should Understand After Running This

- What an embedding vector is and why similar texts have similar vectors
- Why we need a vector *index* (without it, Atlas scans every document — slow)
- What `k=3` means: retrieve the 3 closest documents to the question
- Why the prompt says "answer only from context" — without this, the LLM ignores your data
- What LCEL is: the `|` pipe syntax for chaining LangChain components
