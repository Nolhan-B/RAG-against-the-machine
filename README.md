*This project has been created as part of the 42 curriculum by nbilyj*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about the vLLM codebase. The system indexes the vLLM repository, retrieves relevant code snippets and documentation for a given question, and generates an answer using a local LLM (Qwen3-0.6B).

## Instructions

### Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) as package manager

### Installation

```bash
cd student
uv sync
```

### Setup

Place the vLLM repository in `data/raw/`:
```
data/raw/vllm-0.10.1/
```

Place the datasets in `data/datasets/`:
```
data/datasets/AnsweredQuestions/
data/datasets/UnansweredQuestions/
```

### Usage

**Index the repository:**
```bash
uv run python -m src index --max_chunk_size 2000
```

**Search a single query:**
```bash
uv run python -m src search "How to configure OpenAI server?" --k 10
```

**Search a full dataset:**
```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results
```

**Answer a single question:**
```bash
uv run python -m src answer "How to configure OpenAI server?" --k 10
```

**Generate answers for a dataset:**
```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer
```

**Evaluate retrieval performance:**
```bash
uv run python -m src evaluate \
  --student_answer_path data/output/search_results/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10
```

## System Architecture

```
Repository files
      ↓
  Ingester          reads .py and .md files, calls the right chunker
      ↓
  Chunkers          PythonChunker (AST) and MarkdownChunker (headers)
      ↓
  BM25Indexer       builds BM25 index, saves to disk
      ↓
  BM25Retriever     loads index, returns top-k MinimalSource for a query
      ↓
  Generator         builds prompt with retrieved context, calls Qwen3-0.6B
      ↓
  Answer (JSON)
```

The `Evaluator` sits outside this pipeline and compares retriever output against ground truth using recall@k.

## Chunking Strategy

Two strategies are implemented depending on file type:

**Python files** — uses Python's `ast` module to parse the file into a syntax tree. Only top-level functions and classes (`tree.body`) are extracted as chunks. This avoids overlapping chunks from nested definitions. If a node exceeds `max_chunk_size`, it is split by character size.

**Markdown files** — splits on header lines (lines starting with `#`). A new chunk starts at each header or when the current chunk would exceed `max_chunk_size`.

Both strategies track exact character positions (`first_character_index`, `last_character_index`) so the evaluator can compute overlaps correctly.

Maximum chunk size defaults to 2000 characters and is configurable via `--max_chunk_size`.

**Impact of chunk size:**
- Too small → chunks lack context, retrieval finds pieces that don't answer the question
- Too large → fewer chunks, BM25 scores are diluted, recall drops

## Retrieval Method

BM25 (Best Match 25) via the `bm25s` library.

BM25 is an improved version of TF-IDF. For each query term it computes:
- **TF** (term frequency) — how often the word appears in the chunk
- **IDF** (inverse document frequency) — how rare the word is across all chunks
- **Length normalization** — penalizes very long documents that match just by size

Compared to plain TF-IDF, BM25 saturates the term frequency (a word appearing 100 times is not 100x better than one appearing 10 times) and normalizes by document length more aggressively.

The index is built once and saved to `data/processed/`. At retrieval time, the query is tokenized and scored against all chunks. The top-k chunks by BM25 score are returned.

## Performance Analysis

Target thresholds from the subject:
- Recall@5 on docs questions: ≥ 80%
- Recall@5 on code questions: ≥ 50%

A source is counted as found if at least 5% of the ground truth chunk overlaps with any retrieved chunk.

## Design Decisions

**BM25 over TF-IDF** — better length normalization and term saturation, higher recall on long technical documents with repeated terms.

**AST-based Python chunking** — splitting on logical units (functions, classes) rather than fixed character windows preserves semantic meaning and avoids cutting a function in half.

**`tree.body` only** — avoids overlapping chunks from nested definitions. A class is one chunk, not the class + each method separately.

**Context size limit** — the generator caps context at 6000 characters to stay within Qwen3-0.6B token limits without truncating mid-sentence.

**`do_sample=False`** — deterministic generation for reproducible answers in a Q&A context.

## Challenges Faced

**Character index tracking** — chunkers must track exact byte positions in the original file. Using `splitlines(keepends=True)` was necessary to keep `\n` in the line length calculation, otherwise all positions would be off.

**BM25 index size** — the full vLLM repo produces tens of thousands of chunks. Saving the chunks separately as a pickle file and the BM25 index separately avoids memory issues on reload.

**Token limits** — Qwen3-0.6B has a limited context window. Feeding too many chunks causes truncation or errors. The generator cuts context greedily at `MAX_CONTEXT_CHARS`.

## Resources

- [BM25 paper — Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
- [bm25s library](https://github.com/xhluca/bm25s)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Qwen3-0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic v2 docs](https://docs.pydantic.dev/latest/)
- [Python ast module](https://docs.python.org/3/library/ast.html)

**AI usage** — Claude was used to accelerate boilerplate generation (pyproject.toml, Makefile, pydantic models) and to get an initial skeleton for each class. All logic was reviewed, understood, and validated manually. The chunking strategy, BM25 pipeline, and evaluation metric were implemented and debugged by the student.