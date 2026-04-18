# Retrieval-Augmented Generation (RAG)

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances Large Language Models (LLMs) by giving them access to external knowledge at inference time. Instead of relying solely on knowledge baked into model weights during training, RAG retrieves relevant documents from a knowledge base and feeds them as context to the LLM before generating a response.

**Core idea:** `Answer = LLM(Query + Retrieved Context)`

---

## Why RAG?

LLMs have two fundamental limitations:

| Problem | Description |
|---|---|
| **Knowledge cutoff** | Model weights freeze at training time — the model doesn't know about recent events |
| **Hallucination** | LLMs confidently generate plausible but factually wrong answers |
| **No private data** | Pre-trained models have no access to your internal documents, databases, or wikis |

RAG solves all three by grounding the LLM's response in retrieved, up-to-date, verifiable documents.

---

## RAG Architecture

A RAG system has two phases:

### Phase 1 — Indexing (Offline)

```
Raw Documents
     │
     ▼
[Chunking]           Split documents into smaller passages (e.g., 512 tokens)
     │
     ▼
[Embedding Model]    Convert each chunk → dense vector (e.g., 768-dim float array)
     │
     ▼
[Vector Store]       Store vectors + metadata (e.g., Pinecone, Chroma, FAISS, pgvector)
```

### Phase 2 — Retrieval + Generation (Online)

```
User Query
     │
     ▼
[Embedding Model]    Embed the query → query vector
     │
     ▼
[Vector Search]      Find top-k most similar chunks (cosine / dot-product similarity)
     │
     ▼
[Context Assembly]   Concatenate retrieved chunks into a prompt
     │
     ▼
[LLM]                Generate an answer grounded in the retrieved context
     │
     ▼
Response
```

---

## Key Components

### 1. Document Chunking

How you split documents matters enormously.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,       # overlap prevents losing context at boundaries
    separators=["\n\n", "\n", ".", " "],
)
chunks = splitter.split_text(document_text)
```

**Chunking strategies:**
- **Fixed-size** — simple, consistent; may cut sentences mid-way
- **Sentence-aware** — respects sentence boundaries
- **Semantic** — groups semantically related sentences (more expensive)
- **Document-structure-aware** — respects headers, paragraphs (best for structured docs)

### 2. Embedding Models

Embeddings map text to a point in high-dimensional space where semantically similar texts are close together.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embeddings = model.encode(chunks)   # shape: (n_chunks, 768)
```

Popular embedding models:
| Model | Dims | Notes |
|---|---|---|
| `text-embedding-3-small` (OpenAI) | 1536 | Good balance of cost/quality |
| `text-embedding-3-large` (OpenAI) | 3072 | Highest quality from OpenAI |
| `BAAI/bge-base-en-v1.5` | 768 | Strong open-source option |
| `nomic-embed-text` | 768 | Open-source, long context |

### 3. Vector Stores

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))],
)

# Retrieve top-3 most relevant chunks
results = collection.query(query_embeddings=[query_embedding], n_results=3)
```

Popular vector stores:
| Store | Type | Best for |
|---|---|---|
| **FAISS** | In-memory library | Local / research |
| **Chroma** | Embedded DB | Prototyping |
| **Pinecone** | Managed cloud | Production at scale |
| **pgvector** | Postgres extension | If you already use Postgres |
| **Weaviate** | Managed / self-hosted | Hybrid search |

### 4. Retrieval

**Similarity search** — find chunks whose embedding is closest to the query embedding.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# In practice, vector stores handle this for you
top_k_chunks = vector_store.similarity_search(query, k=5)
```

**Retrieval strategies:**
- **Dense retrieval** — embedding similarity (semantic)
- **Sparse retrieval (BM25)** — keyword matching (lexical)
- **Hybrid** — combine dense + sparse scores (often best in practice)
- **Re-ranking** — use a cross-encoder to re-score the top-k results

### 5. Prompt Assembly & Generation

```python
def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(retrieved_chunks)
    return f"""You are a helpful assistant. Answer the question using only the provided context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {query}

Answer:"""

prompt = build_prompt(user_query, top_k_chunks)
response = llm.generate(prompt)
```

---

## End-to-End Example (Python)

```python
from sentence_transformers import SentenceTransformer
import chromadb
import anthropic

# --- Setup ---
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
client = chromadb.Client()
collection = client.create_collection("docs")
llm = anthropic.Anthropic()

# --- Index documents ---
docs = [
    "RAG stands for Retrieval-Augmented Generation.",
    "Vector databases store high-dimensional embeddings for fast similarity search.",
    "Chunking splits documents into smaller pieces before embedding.",
]
embeddings = embedder.encode(docs).tolist()
collection.add(documents=docs, embeddings=embeddings, ids=[f"d{i}" for i in range(len(docs))])

# --- Query ---
query = "What does RAG stand for?"
query_embedding = embedder.encode([query]).tolist()

results = collection.query(query_embeddings=query_embedding, n_results=2)
context_chunks = results["documents"][0]

prompt = f"""Answer using only this context:
{chr(10).join(context_chunks)}

Question: {query}"""

response = llm.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    messages=[{"role": "user", "content": prompt}],
)
print(response.content[0].text)
```

---

## Advanced RAG Techniques

### Hypothetical Document Embeddings (HyDE)
Instead of embedding the raw query, ask the LLM to generate a hypothetical answer first, then embed that. The hypothetical answer is closer in embedding space to real answer documents.

```python
hypothetical = llm.generate(f"Write a short passage that answers: {query}")
query_embedding = embedder.encode([hypothetical])
```

### Multi-Query Retrieval
Generate multiple paraphrased versions of the query, retrieve for each, then merge and deduplicate results.

### Re-Ranking
After dense retrieval, use a cross-encoder to re-score and re-order results before feeding to the LLM.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, chunk) for chunk in top_k_chunks])
reranked = [chunk for _, chunk in sorted(zip(scores, top_k_chunks), reverse=True)]
```

### Parent Document Retrieval
Index small child chunks for high-precision retrieval, but return the larger parent document as context for the LLM.

### Contextual Compression
Ask the LLM to compress each retrieved chunk to only the part relevant to the query before assembling the final prompt.

---

## RAG vs Fine-Tuning

| | RAG | Fine-Tuning |
|---|---|---|
| **Knowledge update** | Real-time (update the index) | Requires re-training |
| **Cost** | Inference + retrieval | Training cost (GPU hours) |
| **Factual grounding** | Strong (cites sources) | Weaker |
| **Latency** | Higher (retrieval step) | Lower |
| **Best for** | Dynamic, private knowledge | Style/format/behavior changes |

In practice, **RAG + Fine-Tuning** together often works best: fine-tune for format and behavior, use RAG for factual grounding.

---

## Common Failure Modes

| Issue | Cause | Fix |
|---|---|---|
| Wrong chunks retrieved | Poor chunking or embedding model | Tune chunk size; try a better embedder |
| Answer ignores context | LLM "forgets" the context | Rewrite prompt; reduce chunk count |
| Too much irrelevant context | k is too large | Lower k; use re-ranking |
| Hallucination despite retrieval | LLM ignores grounding instruction | Stronger system prompt; citations |
| Slow retrieval | Large index, no ANN index | Use HNSW or IVF index in vector store |

---

## Summary

1. **Chunk** your documents into small, overlapping passages
2. **Embed** each chunk with an embedding model → store in a vector DB
3. At query time, **embed the query** and retrieve the top-k most similar chunks
4. **Assemble a prompt** with the retrieved chunks as context
5. Pass to an **LLM** to generate a grounded answer
