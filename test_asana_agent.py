#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "langgraph",
#   "langchain",
#   "langchain-chroma",
#   "langchain-google-genai",
#   "chromadb",
#   "python-dotenv",
#   "asana",
# ]
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///
"""
test_asana_agent.py

Test script to demonstrate Asana tool integration with the LangGraph agent.

Usage:
  # Test direct task creation (no LLM needed)
  python test_asana_agent.py --direct --task-name "My Task" --task-notes "Task description"
  
  # Test agent with LLM that may trigger task creation
  python test_asana_agent.py --query "Please create a task for setting up BookStack"
"""
import argparse
import logging
import langgraph_agent
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main():
    parser = argparse.ArgumentParser(description='Test Asana tool integration with agent')
    parser.add_argument('--direct', action='store_true', help='Test direct task creation without agent')
    parser.add_argument('--task-name', default='Test Task', help='Task name for direct creation')
    parser.add_argument('--task-notes', default='Created via test script', help='Task notes for direct creation')
    parser.add_argument('--query', default=None, help='Query to test with agent')
    parser.add_argument('--persist-dir', default='./chroma_db', help='Chroma persist directory')
    parser.add_argument('--collection', default='bookstack_pages', help='Chroma collection name')
    parser.add_argument('--google-llm-model', default=None, help='Google LLM model')
    parser.add_argument('--google-embedding-model', default=None, help='Google embedding model')
    args = parser.parse_args()

    # Test direct tool usage
    if args.direct:
        print('Testing direct Asana task creation...\n')
        result = langgraph_agent.create_asana_task(args.task_name, args.task_notes)
        print(f'Result: {result}\n')
        return

    # Test with agent
    if not args.query:
        print('Error: --query required for agent test (or use --direct for tool-only test)')
        return

    # Build agent
    try:
        agent = langgraph_agent.build_langgraph_rag_agent(
            args.persist_dir,
            args.collection,
            top_k=3,
            embedding_model=args.google_embedding_model,
            llm_model=args.google_llm_model,
        )
    except Exception as e:
        print(f'Failed to build agent: {e}')
        return

    print(f'Testing agent with query: {args.query}\n')

    # Invoke agent - this will run retrieve -> generate -> tool pipeline
    result = agent.invoke(args.query)
    
    print('=== Agent Output ===')
    print(f'Answer: {result.get("answer", "No answer")}\n')
    
    if result.get('tool_result'):
        print(f'Tool Result: {result.get("tool_result")}\n')


if __name__ == '__main__':
    main()
