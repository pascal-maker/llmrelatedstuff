# Recap: QMD and llama.cpp retrieval work

This is the public-safe technical summary of the local retrieval work. Private vault contents, note names, databases, embeddings, and model files are not included here.

## What was built and validated

- QMD provides local Markdown indexing and retrieval.
- `node-llama-cpp` runs GGUF models locally.
- `embeddinggemma-300M` provides vector embeddings.
- SQLite with `sqlite-vec` stores and searches embeddings.
- Lexical, vector, and hybrid retrieval paths were exercised.
- The source documents remain canonical; retrieval is a sidecar and does not rewrite them.

Validation results:

- QMD build passed.
- 230 focused tests passed.
- sqlite-vec loaded successfully.
- A full local collection was indexed with 0 pending embeddings.
- A ten-question vector benchmark passed 10/10.
- A ten-question hybrid benchmark passed 10/10.

## Model and runtime findings

Vector retrieval uses the embedding GGUF and works through llama.cpp with Metal acceleration on macOS.

Hybrid retrieval additionally uses:

- a query-expansion model of about 1.28 GB;
- a reranker model of about 639 MB.

The first hybrid run downloads these models and is slow. Later runs use the cache and are much faster. Model downloads should therefore be explicit rather than silently triggered by a benchmark or normal search command.

The sandboxed execution environment could not create a Metal command queue or the normal SQLite WAL sidecar. The same operations succeeded with ordinary macOS GPU and filesystem access, so those failures were environmental rather than model or database failures.

## Companion project

A separate private repository contains the retrieval benchmark and safety wrappers:

- model preflight checks never download files;
- model pulling is an explicit opt-in command;
- benchmark expectations allow valid related or hub documents where hybrid retrieval chooses them over a narrower note;
- no private source documents are committed.

The benchmark is intended to catch retrieval regressions without turning personal source material into a public dataset.
