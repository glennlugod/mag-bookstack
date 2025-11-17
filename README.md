# mag - BookStack (with uv-managed script)

This repository includes a Docker Compose setup for BookStack and a small utility script to populate a BookStack instance with sample content.

Start the application (uses linuxserver/bookstack):

```bash
docker compose up -d
```

Using uv to run the population script (recommended):

```bash
# Run the script using uv (requests will be installed by uv into a managed env)
uv run scripts/populate_bookstack.py -- --token-id <ID> --token-secret <SECRET>
```

If you don't use uv, install requirements and run with Python 3.11:

```bash
python3.11 -m pip install -r scripts/requirements.txt
python3.11 scripts/populate_bookstack.py --token-id <ID> --token-secret <SECRET>
```

The default BookStack instance is exposed at http://localhost:6875

## RAG Indexing & Reindexing (Embeddings & LLM models)

This project includes utilities to create and query a Chroma vector store of BookStack page embeddings for RAG workflows.

### Important points
- Embeddings for a collection must always use the same embedding model at index time and query time.
- If you index with a "dummy" or a small local embedding function (used for testing/sample mode), the stored vectors are shorter (e.g., 16-dim). Querying with a different model (e.g., Google or other provider producing 768/1536-dim vectors) will cause a dimension mismatch error.

### Reindexing (recommended if model changes)

If you've changed the embedding model and need to rebuild your collection (for example, switching from `dummy` to `models/text-embedding-004`), reindex using the `embed_bookstack.py` script and the `--force-reindex` option to delete the old collection first:

```bash
# Example: reindex using the Google embedding model
export GOOGLE_API_KEY=...  # set your Google credentials
export GOOGLE_EMBEDDING_MODEL=models/text-embedding-004

python scripts/embed_bookstack.py \
	--persist-dir ./chroma_db \
	--collection bookstack_pages_sample \
	--force-reindex \
	--google-embedding-model models/text-embedding-004
```

### Querying the collection (RAG)
You can query with the `main.py` CLI task which uses a LangGraph agent internally. Ensure the embedding model and LLM model are set either via CLI args or `.env` values.

```bash
# Query using the same embedding model used to index
python main.py rag \
	--persist-dir ./chroma_db \
	--collection bookstack_pages_sample \
	--query "How do I install X?" \
	--google-embedding-model models/text-embedding-004 \
	--google-llm-model gemini-2.5-flash-lite
```

### Local testing with `dummy` models
For local testing without remote embedding/LLM credentials, the `embed_bookstack.py` script exposes a `--sample` mode (or you can set `GOOGLE_EMBEDDING_MODEL=dummy` and `GOOGLE_LLM_MODEL=dummy`) which uses deterministic local functions for embeddings and a placeholder LLM output string. If you choose this route, make sure to rebuild the collection using `dummy` as the embedding model and query with `dummy` as well.

### Diagnosing dimension mismatch errors
If you see an error like:

```
Collection expecting embedding with dimension of 16, got 768
```

It means your collection was created with 16-dim vectors (dummy) but the embedding model used for query produces 768-dim vectors (Google). Solutions:

- Reindex the collection using the desired model (see the `--force-reindex` command above).
- Or query with the same model used to create the collection (set `--google-embedding-model dummy` if the collection was created with the `--sample` dummy embeddings).

### Helpful diagnostics
The `langgraph_agent` provides a diagnostic helper that can attempt to inspect the collection's stored embedding dimension and count; if you run into errors, use the CLI to inspect the collection and reindex accordingly.

If you want, I can also add a convenience `--reindex` option to the `main.py rag` subcommand which will call `embed_bookstack.py` for you using the provided models — say the word and I'll add it.

