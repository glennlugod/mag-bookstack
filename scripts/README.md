# BookStack population script

This script boots a sample dataset into a BookStack instance running locally (default: http://localhost:6875) using an API token id and secret.

Usage
-----
Run using uv (recommended):

```bash
# Run directly; uv will create a managed environment and install declared dependencies
uv run scripts/populate_bookstack.py -- --token-id <ID> --token-secret <SECRET>

If automatic header discovery fails you can force a header format:

```bash
uv run scripts/populate_bookstack.py -- --header-format token --token-id <ID> --token-secret <SECRET>
```
```

Optional: add and lock the script dependencies using uv before running:

```bash
uv add --script scripts/populate_bookstack.py 'requests>=2.28'
uv lock --script scripts/populate_bookstack.py
uv run --script scripts/populate_bookstack.py -- --token-id <ID> --token-secret <SECRET>
```

Alternative: Install dependencies using pip and run directly (no uv):

```bash
python3 -m pip install -r scripts/requirements.txt
python3 scripts/populate_bookstack.py --token-id <ID> --token-secret <SECRET>
```

Options
-------
- `--url`  — Base URL of BookStack (default: `http://localhost:6875`)
- `--token-id` — **Required** API token ID
- `--token-secret` — **Required** API token secret
- `--template` — JSON file describing structure to create; if omitted a default dataset is used

JSON Template
-------------
Example file structure:

```json
{
  "books": [
    {
      "name": "Getting Started",
      "description": "A sample getting started book",
      "chapters": [
        {
          "name": "Introduction",
          "pages": [
            {"name": "Overview", "content": "<p>Overview page</p>"}
          ]
        }
      ],
      "pages": [
        {"name": "Extra", "content": "<p>Extra book-level page</p>"}
      ]
    }
  ]
}
```

Notes
-----
- The script attempts multiple token header formats to discover the one your BookStack instance uses. If it fails, double-check your token ID/secret and BookStack version.
- If you prefer, you can craft JSON input to create more detailed structure.

Embedding BookStack content (RAG)
--------------------------------
You can generate embeddings for all pages in your BookStack instance and store them in a local Chroma vector DB using the `embed_bookstack.py` script.

Quick start (install dependencies, then run with your BookStack API token):

Using `uv` (recommended):

```bash
# Install dependencies and lock for the script
# uv add --script scripts/embed_bookstack.py 'requests>=2.28' 'beautifulsoup4>=4.12' 'langchain>=0.0.318' 'chromadb>=0.3.38' 'tqdm>=4.65' 'langchain-google-genai>=0.1.0'
uv add --script scripts/embed_bookstack.py 'requests' 'beautifulsoup4' 'langchain' 'chromadb' 'tqdm' 'langchain-google-genai' 'langchain_community'
uv lock --script scripts/embed_bookstack.py

# Run the script in an isolated temporary environment using the locked deps
uv run --script scripts/embed_bookstack.py -- --token-id <ID> --token-secret <SECRET> --url http://localhost:6875 \
  --persist-dir ./chroma_db --collection bookstack_pages
```

Alternatively, without `uv`:

```bash
python3 -m pip install -r scripts/requirements.txt
python3 scripts/embed_bookstack.py --token-id <ID> --token-secret <SECRET> --url http://localhost:6875 \
  --persist-dir ./chroma_db --collection bookstack_pages
```

- `GOOGLE_API_KEY` or application credentials must be available in the environment for Google Generative AI embedding usage
- `--google-embedding-model` can be set to choose a particular embedding model supported by LangChain’s GoogleGenerativeAIEmbeddings wrapper (optional)
  - If not provided, the script will look at the `GOOGLE_EMBEDDING_MODEL` environment variable and fallback to `embed-text-embedding-3` as a default.
Important env vars / configuration:
- `GOOGLE_API_KEY` or application credentials must be available in the environment for Google Generative AI embedding usage
- `--google-embedding-model` can be set to choose a particular embedding model supported by LangChain’s GoogleGenerativeAIEmbeddings wrapper (optional)
  - If not provided, the script will look at the `GOOGLE_EMBEDDING_MODEL` environment variable and fallback to `embed-text-embedding-3` as a default.
- The script can read `.env` from repository root; available env vars:
  - `BOOKSTACK_TOKEN_ID`
  - `BOOKSTACK_TOKEN_SECRET`
  - `BOOKSTACK_URL` (defaults to `http://localhost:6875`)
  - `GOOGLE_API_KEY` (for Google embeddings)
  - `GOOGLE_EMBEDDING_MODEL` (optional; example: `embed-text-embedding-3`)

Example .env file:
```ini
# .env (in repo root)
BOOKSTACK_TOKEN_ID=your_token_id_here
BOOKSTACK_TOKEN_SECRET=your_token_secret_here
BOOKSTACK_URL=http://localhost:6875
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
GOOGLE_EMBEDDING_MODEL=embed-text-embedding-3
```

Compatibility note:
- Some packages (or transitive dependencies like `pypika`) may not be compatible with very new Python versions such as Python 3.14.
  The script is constrained to Python >=3.12 and <3.14 so `uv` will attempt to use a Python 3.12 or 3.13 runtime by default to avoid build issues.
  If `uv lock --script scripts/embed_bookstack.py` still fails during build, try adjusting the pinned package versions or running under a supported Python version.

How to run `uv lock` if you have Python 3.14 by default:

- Option 1: Use `pyenv` or `asdf` to install and select a local Python version (3.12 or 3.13), for example with `pyenv`:

```bash
# Install Python 3.13 (if not installed) and set local project version
pyenv install 3.13.10
pyenv local 3.13.10
uv lock --script scripts/embed_bookstack.py
```

- Option 2: Create a dedicated virtualenv using a compatible interpreter and run `uv` from inside it:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
uv lock --script scripts/embed_bookstack.py
```

If you're unable to change the Python version on your machine or prefer to continue with Python 3.14, consider passing a compatible `chromadb` and `pypika` version in `scripts/requirements.txt`, but note that this can require careful dependency resolution.
- `GOOGLE_API_KEY` or application credentials must be available in the environment for Google Generative AI embedding usage
- `--google-embedding-model` can be set to choose a particular embedding model supported by LangChain’s GoogleGenerativeAIEmbeddings wrapper (optional)

- `--limit` to run only a small subset of pages and validate behavior (useful for testing)
- `--no-persist` to skip persisting the vector DB when running tests

Outputs:
- Embeddings are persisted into `--persist-dir` in a Chroma collection named `--collection`

