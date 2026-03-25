# 📘 DocQ — Complete Documentation

> **DocQ** is an Intelligent Document Q&A system powered by **RAG** (Retrieval-Augmented Generation).
> This documentation covers every concept, every file, and every decision behind the project.

---

## Table of Contents

1. [The Problem We're Solving](#1-the-problem-were-solving)
2. [What is RAG?](#2-what-is-rag-retrieval-augmented-generation)
3. [Core Concepts](#3-core-concepts)
   - [Document Ingestion](#-31--document-ingestion)
   - [Chunking](#-32--chunking)
   - [Embeddings](#-33--embeddings-turning-text-into-numbers)
   - [Vector Store (ChromaDB)](#-34--vector-store-chromadb)
   - [BM25 Keyword Search](#-35--bm25-keyword-search)
   - [Hybrid Search](#-36--hybrid-search-semantic--bm25)
   - [Reciprocal Rank Fusion](#-37--reciprocal-rank-fusion-rrf)
   - [Re-Ranking](#-38--re-ranking)
   - [LLM Generation](#-39--llm-generation-gemini)
   - [Confidence Scoring](#-310--confidence-scoring)
   - [Conversation Memory](#-311--conversation-memory)
4. [System Architecture](#4-system-architecture)
5. [File-by-File Breakdown](#5-file-by-file-breakdown)
   - [config.py](#️-configpy--the-control-center)
   - [ingest.py](#-ingestpy--the-ingestion-engine)
   - [rag_chain.py](#-rag_chainpy--the-brain)
   - [main.py](#-mainpy--the-cli-interface)
   - [app.py](#-apppy--the-web-dashboard)
6. [Technology Stack](#6-technology-stack)
7. [How to Use](#7-how-to-use)
8. [Configuration Reference](#8-configuration-reference)
9. [Glossary](#9-glossary)

---

## 1. The Problem We're Solving

Imagine you have 50 PDF reports, some Word docs, and a bunch of text files sitting on your computer. You want to **ask questions in plain English** and get answers **from those documents**, like:

- *"What were the key findings in Q3?"*
- *"Summarize the risk factors mentioned in the policy doc"*
- *"What did the report say about customer churn?"*

**The problem:** LLMs (Large Language Models) like Google Gemini are incredibly smart, but they **haven't read your private documents**. They can only answer based on their general training data. If you ask Gemini about your company's internal policy, it simply doesn't know.

**The solution:** **RAG** — automatically find the relevant parts of your documents and feed them to the LLM right when it's answering your question.

---

## 2. What is RAG? (Retrieval-Augmented Generation)

RAG is a technique that combines **information retrieval** (searching) with **text generation** (AI writing). It works in two steps:

```
Step 1: RETRIEVE  →  Find the most relevant chunks from your documents
Step 2: GENERATE  →  Give those chunks to the LLM and let it write the answer
```

### The Open-Book Exam Analogy

Think of it like an **open-book exam**:

| Component | Analogy |
|-----------|---------|
| **LLM (Gemini)** | The student taking the exam |
| **Your documents** | The textbook |
| **RAG system** | The ability to flip to the right page before answering |
| **Without RAG** | A closed-book exam — the student relies only on memory |
| **With RAG** | An open-book exam — the student can reference the textbook |

### Why Not Just Send the Entire Document to the AI?

LLMs have a **context window limit** — a maximum number of words (tokens) they can process at once. If you have 500 pages of documents, they won't all fit. Even if they did, the AI would struggle to find the relevant needle in that haystack.

RAG solves this by **retrieving only the relevant pieces** (usually 3-4 short chunks), so the LLM gets focused, useful context instead of being overwhelmed.

---

## 3. Core Concepts

These are the building blocks of the RAG pipeline. Each concept builds on the previous one.

---

### 📄 3.1 — Document Ingestion

> *"Loading your raw documents into the system"*

Before you can search anything, the system needs to **read your files**. This project supports five formats:

| Format | Loader Used | Example |
|--------|-------------|---------|
| `.pdf` | `PyPDFLoader` | Research papers, reports |
| `.txt` | `TextLoader` | Plain text notes |
| `.docx` | `Docx2txtLoader` | Word documents |
| `.md` | `UnstructuredMarkdownLoader` | Markdown notes |
| `.csv` | `CSVLoader` | Spreadsheet data |

Each loader reads the file and converts it into a standardized `Document` object with two fields:
- `page_content` — the actual text
- `metadata` — info like filename, file type, size, and when it was ingested

**Duplicate detection:** The system computes a SHA-256 hash of each file. If you run ingestion again and a file hasn't changed, it's automatically skipped to save time and resources.

---

### ✂️ 3.2 — Chunking

> *"Breaking documents into small, searchable pieces"*

A 20-page PDF is too big to use as search context. So we **split it into chunks** — small, overlapping segments of text.

**How it works in this project:**

| Setting | Value | Meaning |
|---------|-------|---------|
| `CHUNK_SIZE` | 1000 characters | Maximum length of each chunk |
| `CHUNK_OVERLAP` | 200 characters | How much consecutive chunks overlap |

```
Original Document (5000 characters)
   ↓ Split with overlap
┌─────────────────────┐
│ Chunk 1: chars 0–1000│
└──────┬──────────────┘
       │  ← 200 char overlap
┌──────┴──────────────┐
│ Chunk 2: chars 800–1800│
└──────┬──────────────┘
       │  ← 200 char overlap
┌──────┴──────────────┐
│ Chunk 3: chars 1600–2600│
└─────────────────────┘
```

**Why overlap?** Without it, an important sentence that falls right at the boundary would be split in half across two chunks, losing its meaning. The 200-character overlap ensures continuity.

**Smart splitting:** The `RecursiveCharacterTextSplitter` tries to split at natural boundaries first:
1. Double newlines (`\n\n`) — paragraph breaks
2. Single newlines (`\n`) — line breaks
3. Sentences (`. `)
4. Spaces (` `)
5. Anywhere (last resort)

This means chunks usually start and end at logical boundaries rather than cutting words in half.

---

### 🔢 3.3 — Embeddings (Turning Text into Numbers)

> *"Converting text into mathematical vectors so a computer can understand similarity"*

Computers don't understand words — they understand numbers. An **embedding model** converts a piece of text into a long list of numbers called a **vector**. The magic is that **similar text gets similar vectors**.

```
"The cat sat on the mat"    → [0.12, -0.45, 0.78, 0.33, ...]  (768 numbers)
"A kitten rested on a rug"  → [0.11, -0.44, 0.77, 0.34, ...]  (very similar!)
"Stock market crashed today" → [0.89, 0.22, -0.56, 0.01, ...]  (very different!)
```

**In this project:** We use Google's `text-embedding-004` model. Every chunk of text gets embedded into a 768-dimensional vector. These vectors are what make semantic search possible.

**Key insight:** Embeddings capture **meaning**, not just words. So "automobile" and "car" would have very similar embeddings, even though they share no letters.

---

### 🗄️ 3.4 — Vector Store (ChromaDB)

> *"A database optimized for storing and searching embeddings"*

After embedding all your document chunks, we need somewhere to store them. **ChromaDB** is a vector database — think of it as a regular database, but instead of searching by exact column values, it searches by **vector similarity**.

**How searching works:**

```
Your question: "What is the refund policy?"
         ↓ embed
   [0.15, -0.42, 0.80, ...]
         ↓ find closest vectors in the database

ChromaDB scans all stored vectors:
   Chunk 47: [0.14, -0.43, 0.79, ...]  →  Distance: 0.02  ← Very close! ✅
   Chunk 12: [0.88, 0.21, -0.55, ...]  →  Distance: 0.91  ← Far away   ❌
   Chunk 33: [0.16, -0.40, 0.78, ...]  →  Distance: 0.05  ← Close!     ✅
```

The database returns the chunks with the **smallest distance** (most similar vectors) to your question.

**Persistence:** ChromaDB stores everything on disk in the `vectorstore/` directory. Once you've ingested documents, you don't need to do it again unless you add new files.

---

### 🔑 3.5 — BM25 (Keyword Search)

> *"Traditional keyword matching — because sometimes exact words matter more than meaning"*

Semantic search (embeddings) is great at understanding **meaning**, but sometimes you need exact **keyword matches**. For example:

- Searching for a product code like `"SKU-4829"`
- Looking for a specific person's name
- Finding an exact error message

**BM25** (Best Matching 25) is a classic text search algorithm — the same family of algorithm Google used before neural networks. It works by:

1. **Tokenizing** — splitting the text into individual words
2. **Scoring** — ranking documents based on how often your search words appear, weighted by how rare those words are across all documents

**Example:**
If the word "policy" appears in almost every document, it's not very useful for ranking. But if "refund" appears in only 2 documents, those 2 documents get a big score boost.

**In this project:** The BM25 index is built during ingestion and saved as a JSON file in the `bm25_index/` directory.

---

### ⚖️ 3.6 — Hybrid Search (Semantic + BM25)

> *"Combining the best of both worlds"*

This project doesn't rely on just one search method. It runs **both** semantic search and BM25 keyword search in parallel, then **merges** the results.

**Why hybrid?**

| Search Type | Good at | Bad at |
|-------------|---------|--------|
| **Semantic** | Understanding meaning, paraphrasing, synonyms | Exact terms, codes, names |
| **BM25 Keyword** | Exact matches, specific terms, IDs | Understanding meaning, synonyms |

By combining both, the system handles a much wider range of queries accurately.

**Weight control:** The `HYBRID_SEARCH_ALPHA` setting (default `0.7`) controls the balance:
- `1.0` = 100% semantic, 0% keyword (pure meaning)
- `0.7` = 70% semantic, 30% keyword (default — favors meaning but respects keywords)
- `0.0` = 0% semantic, 100% keyword (pure keyword matching)

---

### 🔀 3.7 — Reciprocal Rank Fusion (RRF)

> *"A smart way to merge rankings from different search methods"*

After running both semantic and BM25 search, we have two separate ranked lists. How do we combine them into one? That's where **RRF** comes in.

**The idea is simple:** A document's score is based on its **rank position**, not its raw score. Documents that appear near the top of **both** lists get the highest combined score.

**Formula:**
```
RRF_score(doc) = α × 1/(k + semantic_rank) + (1-α) × 1/(k + bm25_rank)
```
Where `k = 60` (a standard constant) and `α = 0.7` (the alpha weight).

**Example walkthrough:**

```
Semantic Search results:  [Chunk A (rank 1), Chunk B (rank 2), Chunk C (rank 3)]
BM25 Keyword results:     [Chunk B (rank 1), Chunk D (rank 2), Chunk A (rank 3)]

RRF scores:
  Chunk A: 0.7 × 1/61 + 0.3 × 1/63 = 0.0115 + 0.0048 = 0.0163
  Chunk B: 0.7 × 1/62 + 0.3 × 1/61 = 0.0113 + 0.0049 = 0.0162
  Chunk C: 0.7 × 1/63 + 0.3 × 0    = 0.0111
  Chunk D: 0.7 × 0    + 0.3 × 1/62 = 0.0048

Final ranking: [Chunk A, Chunk B, Chunk C, Chunk D]
```

Chunk A wins because it appeared in **both** lists (rank 1 semantic, rank 3 keyword).

---

### 📊 3.8 — Re-Ranking

> *"A second, smarter pass to pick the very best results"*

After hybrid search returns the top 6 candidates, the system does a **second pass** using the LLM itself to judge relevance.

**How it works:**
1. For each of the 6 candidate chunks, the system sends a mini-prompt to Gemini:
   > *"Rate the relevance of this passage to the query on a scale of 0–10."*
2. Gemini scores each chunk (e.g., 9, 7, 3, 8, 2, 6)
3. The chunks are sorted by score, and only the **top 4** are kept

**Why re-rank?** Initial retrieval (semantic + BM25) is fast but approximate. Re-ranking with the LLM is slower (requires one LLM call per chunk) but much more accurate because the LLM deeply understands the relationship between the query and the passage.

**Trade-off:**
```
Retrieval (fast, approximate) → 6 candidates
Re-ranking (slow, precise)   → 4 best candidates
```

This is configured by:
- `RETRIEVAL_TOP_K = 6` — how many to initially retrieve
- `RETRIEVAL_RERANK_TOP_K = 4` — how many to keep after re-ranking

---

### 🤖 3.9 — LLM Generation (Gemini)

> *"The AI writes the final answer based on the retrieved context"*

Once we have the top 4 most relevant chunks, they're assembled into a **prompt** along with:

1. **The retrieved context** — the actual text from the top-ranked chunks
2. **The user's question** — what they asked
3. **Conversation history** — previous Q&A turns (for follow-up questions)
4. **Instructions** — rules telling the LLM how to behave

The prompt template (defined in `config.py`) tells Gemini to:
- Answer using **ONLY** the provided context (no hallucination)
- If the information isn't there, say so explicitly
- Use `[Source: filename]` citations
- Provide a confidence rating at the end

**The LLM used:** `gemini-2.0-flash` — a fast, capable model from Google with low latency and strong reasoning.

**Key settings:**
- `LLM_TEMPERATURE = 0.2` — low temperature means more deterministic/factual answers (less creativity)
- `LLM_MAX_OUTPUT_TOKENS = 4096` — maximum length of the generated answer

---

### 🎯 3.10 — Confidence Scoring

> *"How sure is the AI about its answer?"*

The prompt instructs Gemini to self-rate its confidence at the end of every answer:

```
CONFIDENCE: HIGH    → 🟢  Score: 0.9 — Strong evidence in the documents
CONFIDENCE: MEDIUM  → 🟡  Score: 0.6 — Partial evidence or some inference
CONFIDENCE: LOW     → 🔴  Score: 0.3 — Weak evidence, answer may be unreliable
```

The system parses this line from the response, strips it from the displayed answer, and shows it as a visual badge in both the CLI and Web UI.

**This helps you know when to double-check:** A HIGH confidence answer from a clearly relevant chunk is probably reliable. A LOW confidence answer might need manual verification.

---

### 💬 3.11 — Conversation Memory

> *"Remembering what you talked about so you can ask follow-ups"*

Without memory, every question is isolated — the AI forgets you existed between messages. With conversation memory, you can have natural flows:

```
You:  "What is the remote work policy?"
AI:   "The policy states employees can work remotely up to 3 days..."

You:  "What about international employees?"         ← follow-up!
AI:   "For international employees, the policy requires..."
      (AI knows "the policy" = remote work policy from previous turn)
```

**How it works:**
- The system stores the last **5 Q&A turns** (configurable via `MEMORY_WINDOW_SIZE`)
- These are formatted and included in every prompt as "CONVERSATION HISTORY"
- The LLM uses this history to resolve pronouns like "it", "that", "the policy"
- Memory is **per-session** — different sessions (CLI vs Web UI, different browser tabs) have independent memories

---

## 4. System Architecture

### The Complete Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    INGESTION PHASE (one-time)                │
│                                                              │
│  📄 Documents  →  📥 Load  →  ✂️ Chunk  →  🔢 Embed  →  🗄️ Store  │
│  (PDF,TXT,                     (1000     (text-       (ChromaDB) │
│   DOCX,MD,                      chars)    embedding-            │
│   CSV)                                    004)        🔑 BM25   │
└──────────────────────────────────────────────────────────────┘
                              ↕ (persisted on disk)
┌──────────────────────────────────────────────────────────────┐
│                    QUERY PHASE (every question)              │
│                                                              │
│  ❓ Question                                                  │
│     │                                                        │
│     ├──→ 🔮 Semantic Search (ChromaDB)  ──┐                   │
│     │                                     ├→ ⚖️ RRF Fusion    │
│     └──→ 🔑 BM25 Keyword Search ─────────┘                   │
│                                                              │
│          ⚖️ RRF Fusion (top 6)                                │
│              ↓                                               │
│          📊 Re-Rank with LLM (top 4)                          │
│              ↓                                               │
│          📝 Build Prompt (context + question + history)        │
│              ↓                                               │
│          🤖 Gemini 2.0 Flash                                  │
│              ↓                                               │
│          ✅ Answer + Sources + Confidence                      │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow Across Files

```
config.py  ──────────→  (imported by every other file)
                │
ingest.py  ────┤──→  Reads docs → chunks → embeddings → ChromaDB + BM25
                │
rag_chain.py ──┤──→  Searches ChromaDB + BM25 → re-ranks → prompts Gemini
                │
main.py  ──────┤──→  CLI commands: ingest, query, stats, clear
                │
app.py  ───────┘──→  Streamlit web UI (same pipeline, visual interface)
```

---

## 5. File-by-File Breakdown

### 📁 Project Structure

```
document-qa/
├── config.py              ← All settings and constants
├── ingest.py              ← Document loading, chunking, embedding, indexing
├── rag_chain.py           ← Question answering (search → rerank → generate)
├── main.py                ← CLI interface (terminal commands)
├── app.py                 ← Web UI (Streamlit dashboard)
├── requirements.txt       ← Python dependencies
├── .env.example           ← API key template
├── .env                   ← Your actual API keys (not in git)
├── documents/             ← Put your documents here
├── vectorstore/           ← ChromaDB data (auto-created after ingestion)
├── bm25_index/            ← BM25 keyword index (auto-created after ingestion)
└── logs/                  ← Application logs (auto-created)
```

---

### ⚙️ `config.py` — The Control Center

**Purpose:** A single place that holds every tunable setting. Other files import from here.

**What it does:**
1. Loads environment variables from `.env` using `python-dotenv`
2. Defines all file paths (documents, vectorstore, BM25 index, logs)
3. Sets model names, chunking parameters, retrieval settings
4. Contains the full prompt template sent to Gemini
5. Configures structured logging

**All settings at a glance:**

| Setting | Value | What it controls |
|---------|-------|------------------|
| `EMBEDDING_MODEL` | `text-embedding-004` | Google model for converting text → vectors |
| `LLM_MODEL` | `gemini-2.0-flash` | The AI brain that generates answers |
| `LLM_TEMPERATURE` | `0.2` | Creativity level (low = more factual) |
| `LLM_MAX_OUTPUT_TOKENS` | `4096` | Maximum answer length |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `RETRIEVAL_TOP_K` | `6` | Chunks retrieved by hybrid search |
| `RETRIEVAL_RERANK_TOP_K` | `4` | Chunks kept after re-ranking |
| `HYBRID_SEARCH_ALPHA` | `0.7` | Weight: 70% semantic, 30% keyword |
| `MEMORY_WINDOW_SIZE` | `5` | Past Q&A turns to remember |
| `COLLECTION_NAME` | `document_qa` | ChromaDB collection name |
| `SUPPORTED_EXTENSIONS` | `.pdf .txt .docx .md .csv` | File types accepted |

**The Prompt Template** (lines 86–106) is the instruction set sent to Gemini with every question. It tells the LLM to:
- Answer from context only
- Cite sources with `[Source: filename]`
- Handle follow-up questions using conversation history
- Rate confidence as HIGH/MEDIUM/LOW

---

### 📥 `ingest.py` — The Ingestion Engine

**Purpose:** Takes your raw documents and builds the searchable knowledge base.

**The pipeline runs in 4 steps:**

```
Step 1: Load Documents     →  Read PDF, TXT, DOCX, MD, CSV files
Step 2: Chunk Documents    →  Split into ~1000-char overlapping pieces
Step 3: Embed & Store      →  Convert to vectors, save in ChromaDB
Step 4: Build BM25 Index   →  Create keyword search index
```

**Key functions explained:**

#### `load_documents()` (line 91)
- Scans the `documents/` directory for all supported file types
- Computes a SHA-256 hash of each file
- Checks against a stored hash registry — **skips unchanged files**
- Loads each file using its appropriate LangChain loader
- Enriches metadata with `file_name`, `file_type`, `file_size_kb`, `ingested_at`
- Returns the loaded documents + a per-file status report

#### `chunk_documents()` (line 176)
- Uses LangChain's `RecursiveCharacterTextSplitter`
- Splits at natural boundaries (paragraphs → lines → sentences → words)
- Adds a `chunk_index` to each chunk's metadata for traceability

#### `get_embeddings()` (line 204)
- Creates a `GoogleGenerativeAIEmbeddings` instance
- Uses either API key or GCP project credentials (based on your `.env`)

#### `create_vectorstore()` (line 219)
- Embeds all chunks using the embedding model
- Stores them in ChromaDB with the collection name `document_qa`
- Persists to disk in the `vectorstore/` directory

#### `build_bm25_index()` (line 256)
- Extracts text from all chunks
- Tokenizes each chunk (lowercased, split by spaces)
- Saves the corpus, tokenized corpus, and metadata as JSON in `bm25_index/`

#### `ingest_pipeline()` (line 359)
- The **main orchestrator** that calls all 4 steps in sequence
- Wraps everything in Rich progress bars and summary tables
- Returns statistics about the ingestion run

---

### 🔍 `rag_chain.py` — The Brain

**Purpose:** When you ask a question, this file finds and generates the answer.

**The `ask()` function (line 249) does 6 things:**

```
1. Hybrid Retrieval   →  Search ChromaDB + BM25, fuse with RRF  →  6 candidates
2. Re-Rank           →  LLM scores each chunk 0–10              →  4 best
3. Format Context    →  Turn chunks into readable text strings
4. Build Prompt      →  Combine context + question + chat history
5. Generate Answer   →  Send to Gemini, get response
6. Parse Confidence  →  Extract HIGH/MEDIUM/LOW from response

Returns: { answer, sources, confidence, confidence_score, latency_seconds, retrieval_method }
```

**Key functions explained:**

#### `_hybrid_retrieve()` (line 94)
- **Semantic search:** Queries ChromaDB for the top-K most similar chunks
- **BM25 search:** Loads the BM25 index, tokenizes the query, gets keyword scores
- **RRF fusion:** Combines both result lists using Reciprocal Rank Fusion
  - Uses content prefix (first 200 chars) as a unique key to deduplicate
  - Weighted by `HYBRID_SEARCH_ALPHA` (0.7 = 70% semantic, 30% keyword)
  - Returns the top-K fused results

#### `_rerank_documents()` (line 158)
- Takes the hybrid results and sends each chunk to Gemini with a simple prompt:
  > "Rate the relevance of this passage to the query, 0–10. Respond with ONLY a number."
- Parses the numeric score from the response
- Sorts by score (highest first) and keeps only the top `RETRIEVAL_RERANK_TOP_K` (4)
- If scoring fails for a chunk, it defaults to 5.0 (mid-score)

#### `_format_docs()` (line 80)
- Takes a list of Document objects and formats them into a readable string:
  ```
  [Document 1 — Source: report.pdf | Chunk 12]
  The quarterly revenue increased by 15%...

  ---

  [Document 2 — Source: notes.txt | Chunk 3]
  Key risk factors include...
  ```

#### `_format_chat_history()` (line 65)
- Retrieves the conversation history for the current session
- Formats it as `User: ... / Assistant: ...` pairs
- Returns "No previous conversation." if empty

#### `_parse_confidence()` (line 205)
- Searches the LLM response for a `CONFIDENCE: HIGH|MEDIUM|LOW` line
- Strips it from the displayed answer
- Maps to numeric scores: HIGH = 0.9, MEDIUM = 0.6, LOW = 0.3

#### Conversation Memory (lines 38–76)
- `_conversation_store` — an in-memory dictionary: `session_id → [(question, answer), ...]`
- `_save_to_memory()` — appends a new turn and trims to `MEMORY_WINDOW_SIZE`
- `clear_memory()` — wipes history for a given session

---

### 💻 `main.py` — The CLI Interface

**Purpose:** Provides terminal commands to run the pipeline.

**Available commands:**

| Command | What it does |
|---------|-------------|
| `python main.py ingest` | Index documents from `./documents` |
| `python main.py ingest --force` | Re-index everything (skip duplicate detection) |
| `python main.py ingest --dir /path` | Index from a custom directory |
| `python main.py query` | Start interactive Q&A mode (with memory) |
| `python main.py query -q "..."` | Ask a single question |
| `python main.py stats` | View vector store statistics |
| `python main.py clear` | Delete all indexes (with confirmation) |
| `python main.py clear -f` | Delete all indexes (skip confirmation) |
| `python main.py -v <command>` | Enable verbose/debug logging |

**Key features:**
- **Interactive mode** (`query` without `-q`):
  - Supports follow-up questions via conversation memory
  - Type `clear` to reset memory, `quit` to exit
- **Rich output** — answers displayed in styled panels with:
  - Confidence badge (🟢 HIGH / 🟡 MEDIUM / 🔴 LOW)
  - Latency timer
  - Source documents table with file type, chunk number, and content preview
- **Credential check** — verifies Google Cloud auth before running

---

### 🌐 `app.py` — The Web Dashboard

**Purpose:** A Streamlit-based web UI for the same pipeline, with a premium dark-mode aesthetic.

**Run with:** `streamlit run app.py`

**Layout:**

```
┌─────────────────┬───────────────────────────────────────────────┐
│   SIDEBAR       │   MAIN CONTENT                                │
│                 │                                               │
│  📁 Upload      │   🧠 Hero Header                               │
│     Documents   │      (badges: Semantic, BM25, Memory, Rerank) │
│                 │                                               │
│  📂 Local Docs  │   📊 Stats Row                                 │
│     list        │      (Documents, Queries, Avg Latency, Status)│
│                 │                                               │
│  🚀 Ingest      │   💬 Chat History                              │
│  🔄 Force       │      (messages with confidence badges)        │
│                 │                                               │
│  📊 Status      │   ❓ Chat Input                                │
│     metrics     │      "Ask a question about your documents..." │
│                 │                                               │
│  🗑️ Clear Chat  │                                               │
│  🧹 Reset All   │                                               │
└─────────────────┴───────────────────────────────────────────────┘
```

**Features:**
- **Drag-and-drop file upload** — uploads are saved to `documents/` and ingested
- **Chat-style Q&A** — messages displayed in chat bubbles with avatars
- **Confidence badges** — color-coded inline badges (green/yellow/red)
- **Source expanders** — click to view source documents with chunk previews
- **Session stats** — tracks total queries and average latency
- **Premium CSS** — custom glassmorphism design with:
  - Inter + JetBrains Mono fonts
  - Animated gradient hero header
  - Card hover effects with glow shadows
  - Fully dark-mode color palette

---

## 6. Technology Stack

| Technology | Role in This Project |
|------------|----------------------|
| **Python 3.10+** | Programming language for the entire project |
| **LangChain** | Framework that connects loaders, splitters, embeddings, and LLMs into a pipeline |
| **Google Gemini 2.0 Flash** | The LLM (Large Language Model) that generates answers and re-ranks documents |
| **Google GenAI Embeddings** | Converts text chunks into 768-dimensional vectors (`text-embedding-004`) |
| **ChromaDB** | Vector database — stores embeddings and performs similarity search locally |
| **rank_bm25** | Implements the BM25 keyword search algorithm |
| **Streamlit** | Builds the web UI dashboard with minimal code |
| **Rich** | Provides beautiful terminal output — tables, spinners, progress bars, colors |
| **python-dotenv** | Loads API keys from `.env` files |
| **PyPDF** | Reads PDF files |
| **docx2txt** | Reads Word (.docx) files |

---

## 7. How to Use

### First-Time Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate              # Windows
# source venv/bin/activate         # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env             # Windows
# cp .env.example .env             # macOS/Linux

# 4. Set your API key in .env
#    Option A: GOOGLE_API_KEY=your-gemini-api-key
#    Option B: GOOGLE_CLOUD_PROJECT=your-project-id (+ gcloud auth)
```

### Daily Workflow

```bash
# Step 1: Add documents to ./documents/ folder
# (PDFs, TXT, DOCX, MD, CSV)

# Step 2: Ingest them
python main.py ingest

# Step 3: Ask questions
python main.py query                        # Interactive mode
python main.py query -q "What is the..."    # Single question

# Or use the web UI
streamlit run app.py
```

### Useful Commands

```bash
# Check what's indexed
python main.py stats

# Re-ingest everything from scratch
python main.py ingest --force

# Reset everything
python main.py clear
```

---

## 8. Configuration Reference

### Environment Variables (`.env` file)

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | One of these | Gemini API key (simplest option) |
| `GOOGLE_CLOUD_PROJECT` | is required | GCP project ID (for Vertex AI) |
| `GOOGLE_CLOUD_LOCATION` | No | GCP region (default: `us-central1`) |
| `HYBRID_SEARCH_ALPHA` | No | Semantic weight: 0.0–1.0 (default: `0.7`) |
| `MEMORY_WINDOW_SIZE` | No | Past turns to remember (default: `5`) |
| `LOG_LEVEL` | No | Logging level (default: `INFO`) |

### Tuning Tips

| Want to... | Adjust... |
|------------|-----------|
| Get more precise keyword matches | Lower `HYBRID_SEARCH_ALPHA` (e.g., `0.5`) |
| Use only semantic search | Set `HYBRID_SEARCH_ALPHA = 1.0` |
| Get more candidate chunks | Increase `RETRIEVAL_TOP_K` |
| Keep more chunks after re-ranking | Increase `RETRIEVAL_RERANK_TOP_K` |
| Handle longer documents per chunk | Increase `CHUNK_SIZE` |
| Remember more conversation turns | Increase `MEMORY_WINDOW_SIZE` |
| Get more creative answers | Increase `LLM_TEMPERATURE` (e.g., `0.7`) |

---

## 9. Glossary

| Term | Definition |
|------|-----------|
| **RAG** | Retrieval-Augmented Generation — find relevant docs first, then let AI answer using them |
| **LLM** | Large Language Model — an AI trained on massive text data to generate human-like responses |
| **Embedding** | A list of numbers (vector) that represents the meaning of a piece of text |
| **Vector** | An ordered list of numbers, like `[0.12, -0.45, 0.78, ...]` |
| **Vector Store** | A database designed to store vectors and find similar ones quickly |
| **ChromaDB** | The specific vector database used in this project |
| **Chunk** | A small piece of a document (~1000 characters) created by splitting |
| **Chunking** | The process of breaking a large document into smaller pieces |
| **BM25** | Best Matching 25 — a keyword frequency algorithm for text search |
| **Semantic Search** | Finding documents by meaning similarity (using embeddings) |
| **Keyword Search** | Finding documents by exact word matches |
| **Hybrid Search** | Combining semantic and keyword search for better results |
| **RRF** | Reciprocal Rank Fusion — merging multiple ranked lists into one |
| **Re-Ranking** | A second pass where the LLM scores retrieved chunks for relevance |
| **Prompt** | The text instruction sent to the LLM (includes context, question, and rules) |
| **Prompt Template** | A reusable template with placeholders that gets filled before sending |
| **Context Window** | The maximum amount of text an LLM can process at once |
| **Token** | A unit of text (roughly 1 word or subword) that the LLM processes |
| **Temperature** | Controls randomness in LLM output (low = factual, high = creative) |
| **Conversation Memory** | Storing past Q&A turns so the AI can handle follow-up questions |
| **Session** | An isolated conversation context — different sessions have separate memories |
| **Ingestion** | The process of loading, chunking, embedding, and storing documents |
| **Similarity Search** | Finding vectors that are mathematically close to a query vector |
| **SHA-256** | A cryptographic hash function used here for detecting unchanged files |
| **Metadata** | Additional information about a document chunk (filename, type, size, etc.) |
| **LangChain** | A Python framework that connects document loaders, embeddings, and LLMs |
| **Streamlit** | A Python library for building web apps with minimal code |

---

> **Built with** LangChain · Google Gemini · ChromaDB · Streamlit · Rich
