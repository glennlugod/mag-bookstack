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
# ]
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///
"""
langgraph_agent.py

Builds a simple LangGraph agent that performs a RAG-style retrieval against a
Chroma vector store and optionally generates a response using a Google LLM.

Usage (demo):
  python langgraph_agent.py --query "Installation steps" --persist-dir ./chroma_db --collection bookstack_pages_sample --no-gen

This file exposes a `build_langgraph_rag_agent` factory that returns a compiled
graph object with an `invoke` method. The `main` function provides a simple
CLI for quick testing and integration with `main.py`.

Notes:
 - `embedding_model` and `llm_model` are required; they may be provided via
     arguments or set in the environment as `GOOGLE_EMBEDDING_MODEL` and
     `GOOGLE_LLM_MODEL` respectively.
 - For local testing without external APIs, you can pass the literal value
     `dummy` to either model; `dummy` indicates a lightweight local fallback
     implementation for the Embedding and LLM.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional, TypedDict
from dotenv import load_dotenv

# LangChain & Chroma imports
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# LangGraph imports
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from chromadb.errors import InternalError as ChromaInternalError
from chromadb.errors import InvalidArgumentError as ChromaInvalidArgumentError

load_dotenv()


class State(TypedDict, total=False):
    query: str
    docs: List[Dict]
    answer: str


def build_vector_store(persist_dir: str, collection_name: str, embedding_model: Optional[str] = None):
    """Create a Chroma vector store similar to the existing scripts' helpers.

    This duplicates the same fallback pattern used in other scripts for safety.
    """
    if not embedding_model:
        embedding_model = os.environ.get('GOOGLE_EMBEDDING_MODEL')
    logging.info('Using embedding model: %s', embedding_model)
    if not embedding_model:
        raise ValueError('Embedding model is required: pass `embedding_model` to build_vector_store or set GOOGLE_EMBEDDING_MODEL in environment')

    # Allow a special 'dummy' embedding model for local development / tests
    if embedding_model.lower() == 'dummy':
        class DummyEmbeddingsLocal:
            def __init__(self, dim: int = 16):
                self.dim = dim

            def _vector_from_text(self, text: str):
                s = sum(ord(c) for c in text)
                return [float(((s + i * 31) % 1000)) / 1000.0 for i in range(self.dim)]

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [self._vector_from_text(t) for t in texts]

            def embed_query(self, text: str) -> List[float]:
                return self._vector_from_text(text)

        embeddings = DummyEmbeddingsLocal(dim=16)
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

    vectordb = Chroma(persist_directory=persist_dir, collection_name=collection_name, embedding_function=embeddings)
    return vectordb


def build_langgraph_rag_agent(persist_dir: str, collection_name: str, top_k: int = 3, embedding_model: Optional[str] = None, llm_model: Optional[str] = None):
    """Build a LangGraph StateGraph that does a RAG retrieval and optional generation.

    Returns a compiled graph with an `invoke(state: dict, context: dict)` API.
    """

    # Determine embedding and llm models (allow env var fallback, but require both)
    embedding_model = embedding_model or os.environ.get('GOOGLE_EMBEDDING_MODEL')
    if not embedding_model:
        raise ValueError('`embedding_model` is required. Provide via argument or set GOOGLE_EMBEDDING_MODEL in .env')
    llm_model = llm_model or os.environ.get('GOOGLE_LLM_MODEL')
    if not llm_model:
        raise ValueError('`llm_model` is required. Provide via argument or set GOOGLE_LLM_MODEL in .env')

    # Build vector store and compute expected embedding size
    vectordb = build_vector_store(persist_dir, collection_name, embedding_model=embedding_model)
    # Determine expected embedding dimension by calling embedding function
    try:
        ef = getattr(vectordb, 'embedding_function', None) or getattr(vectordb, '_embedding_function', None)
        if ef and hasattr(ef, 'embed_query'):
            test_vec = ef.embed_query('test')
            expected_embedding_dim = len(test_vec)
        else:
            expected_embedding_dim = None
    except Exception:
        expected_embedding_dim = None

    # llm_model is required now; instantiate or warn if creation fails
    # Provide a dummy LLM for local testing if the special name 'dummy' is provided
    if llm_model.lower() == 'dummy':
        class DummyLLM:
            def __call__(self, prompt: str):
                return f"[DUMMY LLM RESPONSE]\nPrompt: {prompt[:200]}"

        llm = DummyLLM()
    else:
        try:
            llm = GoogleGenerativeAI(model=llm_model)
        except Exception as e:
            logging.warning('Failed to initialize Google LLM: %s (continuing without generation)', e)
            llm = None

    # Define nodes for the graph
    def retrieve_node(state: State, runtime: Runtime) -> dict:
        # The state may include a simple dictionary with a 'query' key.
        q = state.get('query') or ''
        if not q:
            return {'docs': []}

        try:
            docs = []
            # Chroma's similarity_search returns Document objects (Langchain core Document)
            # If using SimpleInMemoryVectorStore fallback, attempt to use .get_all
            if hasattr(vectordb, 'similarity_search'):
                results = vectordb.similarity_search(q, k=top_k)
                for r in results:
                    docs.append({'text': r.page_content, 'metadata': getattr(r, 'metadata', {})})
            else:
                # Fallback: get_in_memory entries
                if hasattr(vectordb, 'get_all'):
                    for d in vectordb.get_all():
                        docs.append({'text': d.get('content'), 'metadata': d.get('metadata', {})})
                else:
                    logging.warning('Vector store does not support similarity_search or get_all; returning empty docs')

            return {'docs': docs}
        except Exception as e:
            logging.error('Error in retrieve_node: %s', e)
            return {'docs': []}

    def generate_node(state: State, runtime: Runtime) -> dict:
        docs = state.get('docs') or []
        q = state.get('query') or ''
        if not q:
            return {'answer': ''}

        if llm is None:
            return {'answer': ''}

        # Construct a simple prompt with docs as context
        context = '\n\n'.join([f"{d.get('metadata', {}).get('page_name', '')}: {d.get('text', '')}" for d in docs])
        prompt = f"Use the following context extracted from BookStack docs:\n\n{context}\n\nQuestion: {q}\nAnswer:"
        try:
            out = llm.invoke(prompt)
            # `GoogleGenerativeAI` wrapper returns a string or Document-like; coerce to str
            return {'answer': str(out)}
        except Exception as e:
            logging.error('LLM generation failed: %s', e)
            return {'answer': ''}

    # Build LangGraph state graph
    graph = StateGraph(state_schema=State)
    graph.add_node('retrieve', retrieve_node)
    graph.add_node('generate', generate_node)
    # Explicitly wire retrieve -> generate so the graph executes generation after retrieval
    graph.add_edge('retrieve', 'generate')
    graph.set_entry_point('retrieve')
    graph.set_finish_point('generate')

    compiled = graph.compile()

    # Attach the vector store and an explicit `run` helper for convenience
    class AgentWrapper:
        def __init__(self, compiled_graph):
            self.compiled = compiled_graph
            self.store = vectordb
            self.expected_embedding_dim = expected_embedding_dim

        def invoke(self, query: str):
            return self.compiled.invoke({'query': query})

        def similarity_search(self, query: str, k: int = 3):
            # For direct access to vectordb retrieval without running the graph
            try:
                if hasattr(self.store, 'similarity_search'):
                    # sanity-check: if we can determine the collection embedding dim, compare
                    diag = diagnose_collection(self.store)
                    stored_dim = diag.get('embedding_dim')
                    if stored_dim and self.expected_embedding_dim and stored_dim != self.expected_embedding_dim:
                        raise RuntimeError(
                            f"Embedding dimension mismatch: collection '{diag.get('collection_name')}' stores vectors of length {stored_dim} but the current embedding model produces length {self.expected_embedding_dim}.\n" +
                            "You should reindex your collection with the matching embedding model or use the same embedding (set GOOGLE_EMBEDDING_MODEL) when querying."
                        )
                    return self.store.similarity_search(query, k=k)
                elif hasattr(self.store, 'get_all'):
                    return self.store.get_all()
                else:
                    return []
            except ChromaInternalError as e:
                # Re-throw a helpful error with diagnostics
                diag = diagnose_collection(self.store)
                raise RuntimeError(
                    'Chroma internal error while running similarity search: {}\nCollection diagnostics: {}\n' .format(e, diag)
                ) from e
            except ChromaInvalidArgumentError as e:
                diag = diagnose_collection(self.store)
                raise RuntimeError(
                    'Chroma invalid argument while running similarity search: {}\nCollection diagnostics: {}\n' .format(e, diag)
                ) from e

    return AgentWrapper(compiled)


def diagnose_collection(vectordb):
    """Return small diagnostic info about a Chroma collection/state for errors.

    This tries multiple known attributes safely to avoid causing further errors.
    """
    info = {}
    try:
        client = getattr(vectordb, 'client', None) or getattr(vectordb, '_client', None)
        collection_name = getattr(vectordb, 'collection_name', None) or getattr(vectordb, '_collection_name', None)
        if client and collection_name:
            try:
                col = client.get_collection(collection_name)
                info['count'] = getattr(col, 'count', lambda: 'unknown')()
                # some versions may expose `metadata` or `properties`
                info['has_metadata'] = hasattr(col, 'metadata')
                info['collection_name'] = collection_name
                # attempt to fetch a sample embedding and report its length
                try:
                    sample_res = col.get(include=['embeddings'], limit=1)
                    if isinstance(sample_res, dict):
                        embeddings = sample_res.get('embeddings') or []
                    else:
                        # older clients may return a more structured object
                        embeddings = getattr(sample_res, 'embeddings', [])
                    if embeddings:
                        emb0 = embeddings[0]
                        info['embedding_dim'] = len(emb0)
                except Exception as ee:
                    info['embedding_fetch_error'] = str(ee)
            except Exception as e:
                info['get_collection_error'] = str(e)
        else:
            info['message'] = 'Client or collection name not available on vectordb'
    except Exception as e:
        info['diagnose_error'] = str(e)
    return info


def main():
    parser = argparse.ArgumentParser(description='Minimal LangGraph RAG agent demo')
    parser.add_argument('--persist-dir', default='./chroma_db', help='Chroma persist directory')
    parser.add_argument('--collection', default='bookstack_pages', help='Chroma collection name')
    parser.add_argument('--query', required=True, help='Query string to search against')
    parser.add_argument('--top-k', type=int, default=3, help='Top K results')
    parser.add_argument('--google-llm-model', default=None, help='Model name for Google Generative AI')
    parser.add_argument('--google-embedding-model', default=None, help='Model name for Google Generative AI embeddings')
    parser.add_argument('--no-gen', action='store_true', help='If set, skip generation step (only show retrieved docs)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # Pass required models from args or allow build to pick from env vars
    try:
        agent = build_langgraph_rag_agent(args.persist_dir, args.collection, top_k=args.top_k, embedding_model=args.google_embedding_model, llm_model=args.google_llm_model)
    except Exception as e:
        logging.error('Failed to build agent: %s', e)
        raise

    # Run a retrieval only
    results = agent.similarity_search(args.query, k=args.top_k)
    if results:
        print('Top results:')
        for i, r in enumerate(results, start=1):
            if isinstance(r, dict):
                text = r.get('content', '')
                meta = r.get('metadata', {})
            else:
                text = getattr(r, 'page_content', '')
                meta = getattr(r, 'metadata', {})
            print('----')
            print(f'Result {i}')
            print('Metadata:', meta)
            print('Text snippet:', text[:300])

    if args.no_gen:
        return

    # Run the full graph to produce a final answer
    out = agent.invoke(args.query)
    if out and out.get('answer'):
        print('\n---\nGenerated answer:')
        print(out['answer'])
    else:
        print('No answer generated; ensure GOOGLE_LLM_MODEL and GOOGLE_API_KEY are correctly configured')


if __name__ == '__main__':
    main()
