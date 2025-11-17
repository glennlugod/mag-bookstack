# mag - AI Agent with BookStack & Asana Integration

A demonstration application showcasing an **AI agent built with LangGraph** that integrates with **BookStack** (documentation platform) and **Asana** (task management) APIs. The agent performs RAG (Retrieval-Augmented Generation) queries against BookStack content and can intelligently create tasks in Asana.

## Features

- **LangGraph Agent**: Multi-step AI workflows with state management and tool integration
- **RAG Workflow**: Vector embeddings of BookStack content for semantic search and generation
- **BookStack Integration**: Fetch, embed, and query documentation via REST API
- **Asana Integration**: Create and manage tasks based on agent decisions
- **Webhook Support**: Real-time BookStack event handling for automatic re-indexing
- **Google Generative AI**: Uses Google's embedding and LLM models for generation
- **Docker Compose**: Pre-configured BookStack environment with MariaDB
- **Chroma Vector Store**: Persistent vector database for embeddings

## Demo

Watch a walkthrough of the AI agent in action: [Demo Video on YouTube](https://youtu.be/RWywpa32NK4)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Query → Retrieval → Generation → Tool Decision       │  │
│  │                                                       │  │
│  │  Tools:                                               │  │
│  │  • Search BookStack (via Chroma Vector Store)         │  │
│  │  • Create Asana Task (via Asana API)                  │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
       ┌───────▼───────┐            ┌────────▼─────────┐
       │   BookStack   │            │     Asana        │
       │ REST API      │            │  Python SDK      │
       │               │            │                  │
       │ • Pages       │            │ • Create Task    │
       │ • Content     │            │ • Update Task    │
       │ • Webhook     │            │ • Manage Projects│
       └───────┬───────┘            └──────────────────┘
               │
       ┌───────▼──────────────┐
       │  Chroma Vector DB    │
       │ (Embeddings Storage) │
       └──────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for BookStack)
- Google API credentials for generative AI
- Asana Personal Access Token (PAT) - *optional for RAG-only mode*

### 1. Clone & Setup

```bash
git clone <repository>
cd mag
```

### 2. Install Dependencies

Using `uv` (recommended):
```bash
uv sync
```

Or with pip:
```bash
pip install -r scripts/requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```bash
# Google Generative AI
export GOOGLE_API_KEY="your_google_api_key_here"
export GOOGLE_LLM_MODEL="gemini-2.0-flash"
export GOOGLE_EMBEDDING_MODEL="models/embedding-001"

# BookStack (for API access)
export BOOKSTACK_TOKEN_ID="your_token_id"
export BOOKSTACK_TOKEN_SECRET="your_token_secret"
export BOOKSTACK_URL="http://localhost:6875"

# Asana (optional, for task creation)
export ASANA_PAT="your_personal_access_token"
export ASANA_WORKSPACE_ID="your_workspace_gid"
export ASANA_PROJECT_ID="your_project_gid"
```

### 4. Start BookStack

```bash
docker compose up -d
```

BookStack will be available at `http://localhost:6875`

### 5. Populate BookStack with Sample Content

```bash
uv run scripts/populate_bookstack.py -- --token-id $BOOKSTACK_TOKEN_ID --token-secret $BOOKSTACK_TOKEN_SECRET
```

### 6. Create Embeddings

Index BookStack content into Chroma:

```bash
uv run scripts/embed_bookstack.py \
  --token-id $BOOKSTACK_TOKEN_ID \
  --token-secret $BOOKSTACK_TOKEN_SECRET \
  --persist-dir ./chroma_db \
  --collection bookstack_pages
```

### 7. Run the Interactive Agent

```bash
python main.py
```

Example interaction:
```
You: What are the installation steps?
Agent: [searches BookStack embeddings] Here are the installation steps...

You: Create a task to document this process
Agent: [uses Asana tool] Task created successfully! Task ID: 1234567890

You: exit
Goodbye!
```

## Project Structure

```
mag/
├── main.py                          # Entry point for interactive agent CLI
├── langgraph_agent.py              # LangGraph agent definition & tools
├── bookstack_webhook.py            # Flask webhook for real-time re-indexing
├── pyproject.toml                  # Project dependencies
├── docker-compose.yml              # BookStack + MariaDB services
├── chroma_db/                      # Vector store (persisted)
├── bookstack_data/                 # BookStack Docker volume
│
├── scripts/
│   ├── populate_bookstack.py       # Create sample content in BookStack
│   ├── embed_bookstack.py          # Index content into Chroma
│   ├── query_bookstack.py          # Direct query interface
│   ├── asana_create_task_test.py   # Test Asana task creation
│   ├── webhook_utils.py            # Webhook event handling
│   └── requirements.txt            # Script-specific dependencies
│
├── docs/
│   ├── ASANA_QUICK_START.md       # Asana integration guide
│   ├── ASANA_TOOL_GUIDE.md        # Asana tool usage patterns
│   └── ASANA_TOOL_IMPLEMENTATION.md # Implementation details
│
└── tests/
    └── test_webhook_payload.py     # Webhook event tests
```

## Usage Patterns

### Pattern 1: RAG Query Only (No Tasks)

```bash
python main.py
# Query BookStack content without creating tasks
```

### Pattern 2: Interactive Agent with Tool Usage

```python
from langgraph_agent import build_langgraph_rag_agent

agent = build_langgraph_rag_agent(
    persist_dir='./chroma_db',
    collection_name='bookstack_pages',
    embedding_model='models/embedding-001',
    llm_model='gemini-2.0-flash'
)

# Agent decides when to use tools
result = agent.invoke("I found a bug in the installation guide - create a task to fix it")
print(result['answer'])        # LLM response
print(result['tool_result'])   # Tool execution result or empty string
```

### Pattern 3: Webhook for Live Updates

```bash
python bookstack_webhook.py \
  --host 0.0.0.0 \
  --port 5000 \
  --token-id $BOOKSTACK_TOKEN_ID \
  --token-secret $BOOKSTACK_TOKEN_SECRET
```

When BookStack pages are created/updated, they're automatically re-indexed into Chroma.

## Key Components

### LangGraph Agent (`langgraph_agent.py`)

Defines a stateful AI workflow:

1. **Retrieval**: Query Chroma for relevant BookStack content
2. **Generation**: Use Google LLM to synthesize an answer
3. **Tool Decision**: LLM decides if an Asana task should be created
4. **Execution**: Create task if needed

**State Variables:**
- `query`: User's question
- `context`: Retrieved BookStack pages
- `answer`: Generated response
- `tool_result`: Task creation result
- `should_create_task`: LLM decision flag

### BookStack Tools (`scripts/embed_bookstack.py`)

- **Fetch Content**: Retrieve all pages from BookStack via REST API
- **Parse HTML**: Extract text using BeautifulSoup
- **Chunk Text**: Split content into manageable pieces
- **Generate Embeddings**: Create vectors using Google Embedding API
- **Store Vectors**: Persist in Chroma with metadata

### Asana Tools (`langgraph_agent.py` - `create_asana_task()`)

- Create tasks with name, notes, workspace, and project assignments
- Called by agent when LLM determines a task is needed
- Integrates with LangGraph agent via tool_use branch

### Webhook Handler (`bookstack_webhook.py`)

Flask server that:
- Receives BookStack webhook events
- Validates webhook signatures
- Extracts affected page IDs
- Re-embeds changed content into Chroma
- Keeps vector store in sync with BookStack

## Model Configuration

### Embedding Models

- `models/embedding-001`: Default, fast, dimension ~768
- `models/text-embedding-004`: Higher quality, dimension ~1536
- `dummy`: Lightweight local embedding (for testing)

**Important:** Use the same embedding model at index-time and query-time to avoid dimension mismatches.

### LLM Models

- `gemini-2.0-flash`: Fastest, good for real-time use
- `gemini-1.5-pro`: Most capable, better reasoning
- `gemini-1.5-flash`: Balanced speed/quality

## Reindexing with Model Changes

If you change embedding models, rebuild the collection:

```bash
uv run scripts/embed_bookstack.py \
  --token-id $BOOKSTACK_TOKEN_ID \
  --token-secret $BOOKSTACK_TOKEN_SECRET \
  --force-reindex \
  --collection bookstack_pages
```

The `--force-reindex` flag deletes and recreates the collection.

## Testing

### Test Asana Task Creation

```bash
uv run scripts/asana_create_task_test.py
```

### Test BookStack Query

```bash
uv run scripts/query_bookstack.py \
  --token-id $BOOKSTACK_TOKEN_ID \
  --token-secret $BOOKSTACK_TOKEN_SECRET \
  --query "installation"
```

### Test Webhook Events

```bash
python -m pytest tests/test_webhook_payload.py
```

## Environment Variables Reference

| Variable | Required | Purpose |
|----------|----------|---------|
| `GOOGLE_API_KEY` | Yes | Google Generative AI authentication |
| `GOOGLE_LLM_MODEL` | Yes | LLM for generation (e.g., gemini-2.0-flash) |
| `GOOGLE_EMBEDDING_MODEL` | Yes | Embedding model (e.g., models/embedding-001) |
| `BOOKSTACK_TOKEN_ID` | Yes | BookStack API authentication |
| `BOOKSTACK_TOKEN_SECRET` | Yes | BookStack API authentication |
| `BOOKSTACK_URL` | No | BookStack URL (default: http://localhost:6875) |
| `ASANA_PAT` | No* | Asana Personal Access Token |
| `ASANA_WORKSPACE_ID` | No* | Asana workspace GID |
| `ASANA_PROJECT_ID` | No* | Asana project GID |
| `BOOKSTACK_WEBHOOK_SECRET` | No | Secret to validate webhook requests |

*Required only if using Asana task creation

## Common Tasks

### Reset Everything

```bash
# Stop containers
docker compose down -v

# Remove vector store
rm -rf chroma_db/

# Restart
docker compose up -d
uv run scripts/populate_bookstack.py -- --token-id $BOOKSTACK_TOKEN_ID --token-secret $BOOKSTACK_TOKEN_SECRET
uv run scripts/embed_bookstack.py --token-id $BOOKSTACK_TOKEN_ID --token-secret $BOOKSTACK_TOKEN_SECRET
```

### Debug Agent Behavior

```python
from langgraph_agent import build_langgraph_rag_agent
import logging

logging.basicConfig(level=logging.DEBUG)

agent = build_langgraph_rag_agent(...)
result = agent.invoke("your query")

print("Full result:", result)
```

### View BookStack Content Directly

```bash
curl -H "Authorization: Token $BOOKSTACK_TOKEN_ID:$BOOKSTACK_TOKEN_SECRET" \
  http://localhost:6875/api/pages
```

## Troubleshooting

**Dimension mismatch error when querying**
- You indexed with one embedding model but querying with another
- Solution: Run `embed_bookstack.py` with the same model, or use `--force-reindex`

**No results from semantic search**
- Content may not be indexed
- Check: `uv run scripts/embed_bookstack.py` was run successfully
- Verify collection exists: Check `chroma_db/` directory

**Webhook not receiving events**
- Verify BookStack URL can reach your webhook endpoint
- Check `BOOKSTACK_WEBHOOK_SECRET` is set on both sides
- Inspect logs: `python bookstack_webhook.py --debug`

**Asana task creation fails**
- Verify credentials: `ASANA_PAT`, `ASANA_WORKSPACE_ID`, `ASANA_PROJECT_ID`
- Check workspace/project IDs with: `uv run scripts/asana_create_task_test.py`

## Advanced Configuration

### Chroma Persistence

By default, vectors are persisted to `./chroma_db/`. To use a different location:

```bash
python main.py --persist-dir /custom/path/to/db --collection my_collection
```

### Custom Text Splitting

Edit `embed_bookstack.py` to adjust chunking:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,        # Increase for longer context
    chunk_overlap=200,      # Increase for more overlap
)
```

### Batch Re-embedding

To update specific pages without re-indexing everything:

```python
from scripts.embed_bookstack import BookStackClient, build_vector_store
from langchain_google_genai import GoogleGenerativeAIEmbeddings

client = BookStackClient(url, token_id, token_secret)
pages = client.fetch_pages(book_id=123)  # Fetch specific book

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)
vector_store = build_vector_store(
    embeddings,
    persist_dir="./chroma_db",
    collection_name="bookstack_pages"
)
# Re-add documents...
```

## Documentation

For detailed information, see:
- [ASANA_QUICK_START.md](docs/ASANA_QUICK_START.md) - Quick guide to Asana integration
- [ASANA_TOOL_GUIDE.md](docs/ASANA_TOOL_GUIDE.md) - Using Asana tools in workflows
- [ASANA_TOOL_IMPLEMENTATION.md](docs/ASANA_TOOL_IMPLEMENTATION.md) - Implementation details

## Technologies Used

- **LangGraph**: Orchestration of AI agent workflows
- **LangChain**: LLM and embedding integrations
- **ChromaDB**: Vector store for semantic search
- **Google Generative AI**: LLM and embedding models
- **BookStack API**: Documentation platform integration
- **Asana SDK**: Task management integration
- **Flask**: Webhook HTTP server
- **BeautifulSoup**: HTML parsing
- **Docker**: Container orchestration

## License

[Specify your license here]

## Contributing

[Specify contribution guidelines]

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review relevant documentation in `docs/`
3. Check test files for usage examples
