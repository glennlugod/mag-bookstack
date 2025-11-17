#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "langchain",
#   "langchain-google-genai",
#   "langchain-chroma",
#   "chromadb",
#   "python-dotenv",
# ]
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///
"""
query_bookstack.py

Demonstration script to query a Chroma DB with stored BookStack embeddings and
optionally perform a simple RAG generation using a Google LLM.

This script reads the `--persist-dir` and `--collection` arguments (and FALLBACKS
to corresponding environment variables) and performs a similarity search for the
provided `--query` string.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()


class DummyEmbeddings:
    def __init__(self, dim: int = 8):
        self.dim = dim

    def _vector_from_text(self, text: str):
        s = sum(ord(c) for c in text)
        return [float(((s + i * 31) % 1000)) / 1000.0 for i in range(self.dim)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector_from_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector_from_text(text)


def build_vector_store(persist_dir: str, collection_name: str, embedding_model: Optional[str] = None):
    if embedding_model:
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    else:
        embeddings = DummyEmbeddings(dim=16)
    vectordb = Chroma(persist_directory=persist_dir, collection_name=collection_name, embedding_function=embeddings)
    return vectordb


def run_query(vectordb: Chroma, query: str, top_k: int = 3):
    docs = vectordb.similarity_search(query, k=top_k)
    return docs


def main():
    parser = argparse.ArgumentParser(description='Query a Chroma DB used for BookStack RAG')
    parser.add_argument('--persist-dir', default='./chroma_db', help='Chroma persist directory')
    parser.add_argument('--collection', default='bookstack_pages', help='Chroma collection name')
    parser.add_argument('--query', required=True, help='Query string to search against')
    parser.add_argument('--top-k', type=int, default=3, help='Top K results to return')
    parser.add_argument('--google-llm-model', default=None, help='Name of Google LLM to use for generation (optional)')
    parser.add_argument('--no-gen', action='store_true', help='Skip generation; only show retrieved docs')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    embedding_model = args.google_llm_model or os.environ.get('GOOGLE_EMBEDDING_MODEL') or 'embed-text-embedding-3'
    # If querying a sample collection, default to DummyEmbeddings to match sample data
    if args.collection and args.collection.endswith('_sample'):
        embedding_model = None

    vectordb = build_vector_store(args.persist_dir, args.collection, embedding_model=embedding_model)

    docs = run_query(vectordb, args.query, top_k=args.top_k)
    if not docs:
        logging.info('No results found for query: %s', args.query)
        return

    print('Top results:')
    for i, d in enumerate(docs, start=1):
        print('----')
        print(f'Result {i}')
        print('Metadata:', d.metadata)
        print('Text snippet:', d.page_content[:300])

    if args.no_gen:
        return

    # Attempt to do a basic RAG generation using Google Generative AI chat model.
    try:
        llm = GoogleGenerativeAI(model=os.environ.get('GOOGLE_LLM_MODEL') or 'chat-bison-001')
    except Exception as e:
        logging.error('Failed to initialize LLM for generation: %s', e)
        print('Skipping generation (LLM not available)')
        return

    context = '\n\n'.join([f"{d.metadata.get('page_name', '')}: {d.page_content}" for d in docs])
    prompt = f"Use the following extracted content from BookStack pages as context:\n\n{context}\n\nQuestion: {args.query}\nAnswer:"
    logging.info('Calling LLM for generation...')
    try:
        answer = llm(prompt)
        print('\n---\nGenerated answer:')
        print(answer)
    except Exception as e:
        logging.error('LLM generation failed: %s', e)
        print('Generation failed; check GOOGLE_API_KEY and valid model settings')


if __name__ == '__main__':
    main()
