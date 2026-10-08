# RAG Search Engine

A Python command-line movie search engine that combines keyword, semantic, and
image-based retrieval. It can rerank results and use them to generate answers,
summaries, and citations with an LLM.

![Keyword search returning two astronaut movies from the included sample dataset](examples/keyword-search.png)

*Actual CLI output rendered from the included fictional movie dataset.*

## Motivation

Finding a movie often starts with a remembered scene or a plot idea rather than
an exact title. Keyword search can match specific terms, while semantic search
can connect descriptions that use different words. I built this project to
bring those approaches together and compare how retrieval, ranking, and
generation affect the answers people get. Movie search gives those decisions
a concrete setting: find relevant stories, inspect why they ranked, and use
the retrieved descriptions to answer questions.

## Quick Start

Requires **Python 3.13+** and [uv](https://docs.astral.sh/uv/). The included
sample movies let you run a search without an API key or model download.

```sh
git clone https://github.com/amayakmt/rag-search-engine.git
cd rag-search-engine
uv sync --frozen

export RAG_DATA_DIR="$PWD/examples"
export RAG_CACHE_DIR="$PWD/cache/demo"
uv run rag-keyword build
uv run rag-keyword bm25search "astronaut" --limit 3
```

Expected results:

```text
1. (1) Beyond the Galaxy - Score: 0.48
2. (2) The Last Orbit - Score: 0.46
```

The sample data and demo cache are separate from your own movie collection.
Continue with the same environment variables to try the examples below.

## Usage

### Search and Rank

Compare keyword matching, semantic similarity, and two ways to combine them:

```sh
uv run rag-keyword bm25search "astronaut" --limit 5
uv run rag-semantic search "surviving a disaster in space" --limit 5
uv run rag-semantic search_chunked "discovering a lost civilization" --limit 5
uv run rag-hybrid weighted-search "space adventure" --alpha 0.5 --limit 5
uv run rag-hybrid rrf-search "space adventure" -k 60 --limit 5
```

Semantic and image searches download their embedding models on first use.
Hybrid search automatically builds or refreshes the keyword index; standalone
keyword search requires `rag-keyword build` first.

For query enhancement, reranking, and relevance evaluation:

```sh
uv run rag-hybrid rrf-search "astronuat" --enhance spell
uv run rag-hybrid rrf-search "space adventure" --rerank-method cross_encoder
uv run rag-hybrid rrf-search "space adventure" --rerank-method batch --evaluate --debug
```

| Option | Purpose |
| --- | --- |
| `--limit N` | Maximum number of results; defaults to 5 for these search commands. |
| `--alpha FLOAT` | Keyword weight for weighted search, from 0 to 1; defaults to 0.5. |
| `-k INT` | Reciprocal rank fusion constant; defaults to 60. |
| `--enhance spell\|rewrite\|expand` | LLM-based query enhancement. |
| `--rerank-method individual\|batch\|cross_encoder` | Rerank retrieved candidates with an LLM or a local cross-encoder. |
| `--evaluate` | Ask the LLM to score each result's relevance from 0 to 3. |
| `--debug` | Log the stages of the RRF search pipeline. |

Enhancement, individual/batch reranking, and LLM evaluation require an
`OPENROUTER_API_KEY` in your environment or local `.env`. Model names and the
LLM endpoint are defined in the existing project settings. Cross-encoder
reranking does not require an API key, but downloads its model on first use.
Never commit credentials.

### Generate Answers

Use retrieved movie descriptions as context for answers, summaries, and citations:

```sh
uv run rag-generate rag "Which movies involve astronauts?"
uv run rag-generate summarize "space travel" --limit 5
uv run rag-generate citations "Which movie features a damaged space station?" --limit 5
uv run rag-generate question "What does the astronaut discover in Beyond the Galaxy?" --limit 5
```

These commands require the LLM API key. Generated answers depend on both the
retrieved context and the model's response; citations are not a guarantee of
factual accuracy.

### Search with Images

```sh
uv run rag-multimodal verify_image_embedding path/to/image.jpg
uv run rag-multimodal image_search path/to/image.jpg
uv run rag-describe-image --image path/to/image.jpg --query "Find movies with a similar scene"
```

Image search compares a local image with movie descriptions using CLIP
embeddings. `rag-describe-image` uses the LLM to turn the image and text into
a search query; it requires an API key and an image-capable model.

### Inspect and Evaluate

Inspect keyword scores, split text into chunks, or measure retrieval against
your own relevance labels:

```sh
uv run rag-keyword tf 1 "astronaut"
uv run rag-keyword idf "astronaut"
uv run rag-keyword tfidf 1 "astronaut"
uv run rag-keyword bm25idf "astronaut"
uv run rag-keyword bm25tf 1 "astronaut"
uv run rag-semantic chunk "An astronaut follows a signal into deep space." --chunk-size 5 --overlap 1
uv run rag-semantic semantic_chunk "A signal arrives. An astronaut follows it. A civilization is discovered." --max-chunk-size 2 --overlap 1
uv run rag-evaluate --limit 5
```

`rag-evaluate` reports precision, recall, and F1 using a
`golden_dataset.json` in your selected data directory. The sample collection
does not include relevance labels; add your own before running evaluation.
This evaluation uses labeled results, not an LLM, and needs no API key.

Every command supports `--help`; use subcommand help for its full options:

```sh
uv run rag-hybrid rrf-search --help
```

### Use Your Own Data

Place these files in your selected data directory:

| File | Format |
| --- | --- |
| `movies.json` | `{"movies": [{"id": 1, "title": "Example", "description": "A movie description."}]}` |
| `stopwords.txt` | One stopword per line. |
| `golden_dataset.json` | For evaluation: `{"test_cases": [{"query": "example", "relevant_docs": ["Example"]}]}` |

Movie IDs must be unique integers. Evaluation identifies relevant movies by
title. To switch from the sample collection to the default local data directory:

```sh
unset RAG_DATA_DIR RAG_CACHE_DIR
# Provide data/movies.json and data/stopwords.txt before building.
uv run rag-keyword build
```

Alternatively, set `RAG_DATA_DIR` and `RAG_CACHE_DIR` to your own directories.
Relative paths resolve from the project location, not the launch directory.
Non-editable installations should use external data and writable cache paths.

Local data, generated caches, and `.env` are excluded from Git. Embedding
caches are fingerprinted by model and corpus content; changed datasets rebuild
instead of reusing stale vectors. Older caches without fingerprints rebuild
once. Only use trusted keyword caches: their indexes use Python pickle.

## Contributing

Fork the repository and follow the Quick Start to clone and install it. For
local development, `uv sync` installs the package in editable mode, so source
changes are available without reinstalling.

### Run Tests and Build

```sh
uv run --frozen python -m unittest discover -s tests -v
uv build
```

Tests use temporary caches and mock model inference and LLM calls. They do not
download models or require API credentials. GitHub Actions runs the suite on
Python 3.13. Live retrieval quality and provider responses need separate
integration testing with datasets and credentials.

### Project Structure

```text
rag_search_engine/
    config.py           # Resource paths and public configuration facade
    core/               # Retrieval, ranking, tokenization, and LLM services
    utils/              # Dataset loading and text chunking
cli/                    # Argument parsing and terminal output
    config.py           # Existing project settings
examples/               # Sample movies, stopwords, and CLI demo image
tests/                  # Offline regression tests
data/                   # Local datasets (not committed)
cache/                  # Generated indexes and embeddings (not committed)
pyproject.toml          # Dependencies, packaging, and console commands
uv.lock                 # Locked dependency resolution
```

Keep reusable behavior in `rag_search_engine` and CLI presentation in `cli`.
For example, import `HybridSearch` from
`rag_search_engine.core.hybrid_search`. Module entry points also work:

```sh
uv run python -m cli.hybrid_search_cli rrf-search "space adventure"
```

### Submit a Pull Request

Create a branch in your fork, make a focused change, and add regression tests
for behavior changes. Run the test suite, then open a pull request against
the repository's default branch describing what changed and why. For bugs,
include the command, expected behavior, and actual output without credentials.
