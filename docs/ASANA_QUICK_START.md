# Quick Reference: Asana Tool in LangGraph Agent

## Setup

1. Install dependencies (already in pyproject.toml):
   ```bash
   pip install asana langgraph langchain-google-genai
   ```

2. Set environment variables:
   ```bash
   export ASANA_PAT="<your_personal_access_token>"
   export ASANA_WORKSPACE_ID="<workspace_gid>"
   export ASANA_PROJECT_ID="<project_gid>"
   export GOOGLE_LLM_MODEL="gemini-2.0-flash"
   export GOOGLE_EMBEDDING_MODEL="models/embedding-001"
   ```

## 3 Ways to Use the Tool

### Method A: Direct API Call (Immediate execution)
```python
from langgraph_agent import create_asana_task

result = create_asana_task(
    task_name="Fix BookStack installation issue",
    task_notes="User reported database connection timeout"
)
# Returns: "Task created successfully! Task ID: 1234567890"
```

### Method B: Through Agent (LLM decides)
```python
import langgraph_agent

agent = langgraph_agent.build_langgraph_rag_agent(
    persist_dir='./chroma_db',
    collection_name='bookstack_pages',
    embedding_model='models/embedding-001',
    llm_model='gemini-2.0-flash'
)

# Agent's LLM reads your query and decides if a task is needed
result = agent.invoke("I need to document the database backup procedure")

print(result['answer'])       # LLM's response
print(result['tool_result'])  # "Task created successfully!..." or ""
```

### Method C: Agent Wrapper Method (Convenience)
```python
agent = langgraph_agent.build_langgraph_rag_agent(...)

# Same as Method A but through the agent object
result = agent.create_task("Task Title", "Task description")
```

## What the Tool Does

```
User asks agent to create a task
              ↓
LLM retrieves BookStack docs for context
              ↓
LLM generates response
              ↓
If appropriate, LLM includes: CREATE_ASANA_TASK(name="...", notes="...")
              ↓
Tool node parses this directive
              ↓
Tool creates task in Asana via API
              ↓
Result returned: "Task created successfully! Task ID: XYZ"
```

## Common Queries That Trigger Task Creation

```python
# These will likely trigger CREATE_ASANA_TASK from the LLM:

"Create a task to implement database backups"
"I need to document the installation process"
"Set up a reminder to review security settings"
"Create an issue for the nginx configuration"
"Add a task to update the documentation"
```

## Checking if It Works

### Test 1: Verify Credentials
```bash
# Quick test
python -c "
import os
print('PAT set:', bool(os.environ.get('ASANA_PAT')))
print('Workspace ID:', os.environ.get('ASANA_WORKSPACE_ID'))
print('Project ID:', os.environ.get('ASANA_PROJECT_ID'))
"
```

### Test 2: Direct Tool Test
```bash
uv run test_asana_agent.py --direct \
  --task-name "Quick Test" \
  --task-notes "Testing Asana integration"
```

### Test 3: Full Agent Test
```bash
uv run test_asana_agent.py \
  --query "Please create a task for setting up automated backups" \
  --google-embedding-model "models/embedding-001" \
  --google-llm-model "gemini-2.0-flash"
```

## How to Find Workspace/Project IDs

### Get Workspace ID
```python
from asana import ApiClient, Configuration
from asana.api import WorkspacesApi

config = Configuration()
config.access_token = "your_pat"
client = ApiClient(config)
api = WorkspacesApi(client)

workspaces = api.get_workspaces()
for w in workspaces:
    print(f"Workspace: {w['name']} - GID: {w['gid']}")
```

### Get Project ID
```python
from asana.api import ProjectsApi

projects_api = ProjectsApi(client)
projects = projects_api.get_projects(opts={"workspace": workspace_id})
for p in projects:
    print(f"Project: {p['name']} - GID: {p['gid']}")
```

## Understanding the Graph Flow

```
START
  ↓
[retrieve] 
  Parse query
  Search Chroma for relevant docs
  Return matching documents
  ↓
[generate]
  Feed query + docs to LLM
  LLM generates contextual response
  If task creation needed, LLM includes: CREATE_ASANA_TASK(...)
  ↓
[tool]
  Parse response for CREATE_ASANA_TASK directive
  If found: Call create_asana_task() to create in Asana
  If not found: Return empty tool_result
  ↓
END (output includes: answer, docs, tool_result)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "ASANA_PAT not set" | Set `export ASANA_PAT="..."` in your shell or .env |
| "Workspace/Project ID missing" | Set both `ASANA_WORKSPACE_ID` and `ASANA_PROJECT_ID` |
| "Task not creating automatically" | Check that LLM includes CREATE_ASANA_TASK in response |
| "Tool regex not matching" | Ensure exact format: `CREATE_ASANA_TASK(name="...", notes="...")` |
| "API returns 404" | Verify workspace and project IDs are correct |

## Accessing Full Documentation

- **Feature Guide**: See `ASANA_TOOL_GUIDE.md`
- **Implementation Details**: See `ASANA_TOOL_IMPLEMENTATION.md`
- **Code**: Check `langgraph_agent.py` functions:
  - `initialize_asana_client()`
  - `create_asana_task()`
  - `tool_node()` in `build_langgraph_rag_agent()`

## Example Output

```
You: "Create a task to document the BookStack API"

Agent Output:
{
    'answer': 'I found the BookStack API documentation. CREATE_ASANA_TASK(name="Document BookStack API endpoints", notes="Create comprehensive guide for all API endpoints with examples")',
    'tool_result': 'Task created successfully! Task ID: 1205932471234567',
    'docs': [
        {'text': 'BookStack API ...', 'metadata': {'page_name': 'API Overview'}},
        ...
    ]
}
```

## Advanced: Custom Tool Parsing

To add support for more complex tool formats, modify the regex in `tool_node()`:

```python
# Current pattern (in tool_node):
r'CREATE_ASANA_TASK\(name=["\']([^"\']+)["\'](?:,\s*notes=["\']([^"\']*)["\'])?\)'

# Could be extended to support:
# CREATE_ASANA_TASK(name="...", notes="...", due_date="2025-12-31", assignee="user@example.com")
```

## Performance Notes

- Direct tool call: ~500ms (API call to Asana)
- Full agent invoke: ~2-5s (depends on LLM latency)
- Chroma retrieval: ~100-200ms
- Tool parsing: <1ms

## API Rate Limits

Asana's API has rate limits (typically 150 requests/min). The current implementation makes 1 request per task creation, so you can create ~150 tasks per minute.
