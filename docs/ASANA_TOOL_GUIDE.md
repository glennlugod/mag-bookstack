# Asana Tool Support for LangGraph Agent

## Overview

The LangGraph agent now has integrated support for creating Asana tasks. This allows the agent to not only retrieve information from BookStack and generate responses, but also to create action items (tasks) in Asana based on the context and conversation.

## Features

1. **Direct Tool Access**: Call `agent.create_task()` directly to create Asana tasks
2. **Automatic Tool Integration**: The agent's LLM can intelligently decide when to create tasks based on the query
3. **Tool Parsing**: The agent parses special directives in LLM responses to trigger task creation
4. **Error Handling**: Graceful fallbacks when Asana credentials are not available

## Configuration

Set the following environment variables in your `.env` file:

```bash
ASANA_PAT=<your-asana-personal-access-token>
ASANA_WORKSPACE_ID=<your-workspace-id>
ASANA_PROJECT_ID=<your-project-id>
```

## Usage

### Method 1: Direct Tool Usage

```python
import langgraph_agent

# Initialize the agent
agent = langgraph_agent.build_langgraph_rag_agent(
    persist_dir='./chroma_db',
    collection_name='bookstack_pages',
    embedding_model='models/embedding-001',
    llm_model='gemini-2.0-flash'
)

# Create a task directly
result = agent.create_task(
    task_name='Set up BookStack deployment',
    task_notes='Follow the installation guide from BookStack docs'
)
print(result)  # Output: "Task created successfully! Task ID: <gid>"
```

### Method 2: Automatic Tool Invocation via Agent

The agent is configured to process queries through a pipeline: **Retrieve → Generate → Tool**

```python
# When the LLM decides a task needs to be created, it includes this in the response:
# CREATE_ASANA_TASK(name="Task Name", notes="Task description")

result = agent.invoke('Please create a task for backing up the BookStack database daily')
# The agent will:
# 1. Retrieve relevant docs about backups
# 2. Generate a response that includes: CREATE_ASANA_TASK(name="...", notes="...")
# 3. Parse and execute the tool call to create the task in Asana
```

### Method 3: Using the Test Script

```bash
# Test direct task creation
uv run test_asana_agent.py --direct --task-name "My Task" --task-notes "Description"

# Test agent with query that might trigger task creation
uv run test_asana_agent.py --query "Create a task for BookStack setup" \
  --google-embedding-model "models/embedding-001" \
  --google-llm-model "gemini-2.0-flash"
```

## Graph Architecture

The agent now uses a three-stage processing pipeline:

```
┌──────────────┐
│   retrieve   │  ─── Searches Chroma for relevant docs
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  generate    │  ─── LLM generates response (may include tool directives)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    tool      │  ─── Parses CREATE_ASANA_TASK and executes if found
└──────┬───────┘
       │
       ▼
   (output)
```

## State Schema

The agent's state now includes:

```python
class State(TypedDict, total=False):
    query: str           # User's input query
    docs: List[Dict]     # Retrieved documents from vector store
    answer: str          # LLM-generated response
    tool_result: str     # Result of any tool execution
```

## Implementation Details

### Tool Function: `create_asana_task()`

```python
def create_asana_task(
    task_name: str,
    task_notes: str = '',
    project_name: str = None
) -> str:
    """Create a task in Asana.
    
    Args:
        task_name: Name/title of the task
        task_notes: Optional notes/description
        project_name: Optional project name filter
    
    Returns:
        str: Success message with task GID or error message
    """
```

### Tool Node

The `tool_node()` processes the LLM output looking for:

```
CREATE_ASANA_TASK(name="<task_name>", notes="<task_notes>")
```

It uses regex to parse the directive and execute the task creation.

## LLM Prompting

The LLM is instructed in its system prompt:

> "If the user is asking to create a task in Asana, include the following in your response:
> CREATE_ASANA_TASK(name="<task name>", notes="<task description>")"

This guides the LLM to naturally format task creation requests.

## Error Handling

- **Missing Credentials**: If `ASANA_PAT` is not set, the tool logs a warning and returns a user-friendly error message
- **Missing Workspace/Project**: If `ASANA_WORKSPACE_ID` or `ASANA_PROJECT_ID` are not set, the tool returns an error
- **API Errors**: Asana API errors are caught and returned as readable error messages
- **Parsing Errors**: If the tool directive format is incorrect, the tool silently returns without creating a task

## Examples

### Example 1: Query that Creates a Task

```python
query = "I need to document the installation process for BookStack"

result = agent.invoke(query)
# Agent retrieves installation docs
# Agent generates response and includes:
#   "I've found the installation guide. CREATE_ASANA_TASK(name=\"Document BookStack Installation\", 
#    notes=\"Compile installation steps from BookStack documentation\")"
# Tool node parses and creates the task
```

### Example 2: Query without Task Creation

```python
query = "What are the system requirements for BookStack?"

result = agent.invoke(query)
# Agent retrieves system requirements
# Agent generates response without task directive
# Tool node finds no CREATE_ASANA_TASK directive and returns
```

## Dependencies

The agent includes `asana` in its dependencies:

```toml
dependencies = [
    "asana>=5.2.2",
    "langgraph",
    "langchain",
    "langchain-chroma",
    "langchain-google-genai",
    "chromadb",
    "python-dotenv",
    ...
]
```

## Troubleshooting

### Task Not Being Created

1. Check that `ASANA_PAT` is set correctly
2. Verify `ASANA_WORKSPACE_ID` and `ASANA_PROJECT_ID` are correct
3. Check the agent output for the `tool_result` field to see if there were errors
4. Ensure the LLM includes the `CREATE_ASANA_TASK()` directive in its response

### Agent Not Processing Tool

1. Verify the tool node is properly connected in the graph (check graph edges)
2. Check that the regex pattern in `tool_node()` matches the LLM output format
3. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`

## Future Enhancements

Possible improvements:

1. Support for more complex tool calls (e.g., setting due dates, assignees, priority)
2. Conversation memory to track created tasks
3. Multiple tool support (e.g., create GitHub issues, update spreadsheets)
4. Tool confirmation step before execution
5. Structured tool output in state for audit trail
