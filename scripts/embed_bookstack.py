#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
#   "beautifulsoup4",
#   "langchain",
#   "chromadb",
#   "tqdm",
#   "langchain-google-genai",
#   "python-dotenv",
#   "langchain-chroma>=0.0.6",
# ]
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///
"""
embed_bookstack.py

Fetch BookStack content (books & pages), chunk text, generate embeddings using LangChain
GoogleGenerativeAIEmbeddings and store vectors in a Chroma vector DB.

Usage:
  python scripts/embed_bookstack.py --token-id <id> --token-secret <secret> \
      [--url http://localhost:6875] [--persist-dir ./chroma_db] [--collection bookstack_pages]

Environment:
  - GOOGLE_API_KEY or Google Application credentials as required by Google generative AI SDK
  - Optional: GOOGLE_EMBEDDING_MODEL to choose embedding model used by LangChain wrapper

"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import sys

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain.schema import Document
except Exception:  # pragma: no cover - helpful messaging if packages are missing
    print("Missing required packages. Install requirements from scripts/requirements.txt")
    raise

# Load environment variables from .env at repo root (if present)
load_dotenv()


@dataclass
class PageItem:
    id: int
    name: str
    html: Optional[str]
    book_id: Optional[int]
    book_name: Optional[str]
    url: Optional[str]


class BookStackClient:
    def __init__(self, base_url: str, token_id: str, token_secret: str, timeout: int = 10, header_format: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.token_id = token_id
        self.token_secret = token_secret
        self.timeout = timeout
        self.session = requests.Session()
        if header_format:
            self.auth_header = self._header_by_name(header_format)
        else:
            self.auth_header = self._discover_auth_header()

    def _header_by_name(self, name: str) -> Dict[str, str]:
        name = name.lower()
        if name in ('token', 'auth_token'):
            return {"Authorization": f"Token {self.token_id}:{self.token_secret}"}
        if name in ('token_named', 'token-token'):
            return {"Authorization": f"Token token={self.token_id}:{self.token_secret}"}
        if name in ('x-auth', 'x-auth-token'):
            return {"X-Auth-Token": f"{self.token_id}:{self.token_secret}"}
        if name in ('bearer', 'auth-bearer'):
            return {"Authorization": f"Bearer {self.token_id}:{self.token_secret}"}
        raise RuntimeError(f"Unknown header format name: {name}")

    def _discover_auth_header(self) -> Dict[str, str]:
        trial_formats = [
            lambda t_id, t_secret: {"Authorization": f"Token {t_id}:{t_secret}"},
            lambda t_id, t_secret: {"Authorization": f"Token token={t_id}:{t_secret}"},
            lambda t_id, t_secret: {"X-Auth-Token": f"{t_id}:{t_secret}"},
            lambda t_id, t_secret: {"Authorization": f"Bearer {t_id}:{t_secret}"},
        ]

        probe_url = f"{self.base_url}/api/books"
        for fmt in trial_formats:
            h = fmt(self.token_id, self.token_secret)
            try:
                r = self.session.get(probe_url, headers=h, timeout=self.timeout)
            except Exception:
                continue
            if r.status_code < 400:
                logging.info("Using header format: %s", list(h.keys()))
                return h

        raise RuntimeError("Unable to discover valid token header for BookStack API. Please verify token ID and secret.")

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = kwargs.pop('headers', {})
        headers.update(self.auth_header)
        r = self.session.request(method, url, headers=headers, timeout=self.timeout, **kwargs)
        r.raise_for_status()
        return r

    def list_books(self) -> List[Dict]:
        # Paginate through /api/books
        out = []
        page = 1
        per_page = 100
        while True:
            r = self._request('GET', f'/api/books?page={page}&limit={per_page}')
            try:
                data = r.json().get('data') or r.json()
            except ValueError:
                break
            if not data:
                break
            out.extend(data)
            if len(data) < per_page:
                break
            page += 1
        return out

    def list_pages(self) -> List[Dict]:
        out = []
        page = 1
        per_page = 100
        while True:
            r = self._request('GET', f'/api/pages?page={page}&limit={per_page}')
            try:
                data = r.json().get('data') or r.json()
            except ValueError:
                break
            if not data:
                break
            out.extend(data)
            if len(data) < per_page:
                break
            page += 1
        return out

    def get_page(self, page_id: int) -> Dict:
        r = self._request('GET', f'/api/pages/{page_id}')
        return r.json().get('data') or r.json()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    # Remove script/style
    for el in soup(['script', 'style']):
        el.extract()
    text = soup.get_text(separator='\n')
    # collapse whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return '\n'.join(lines)


def build_vector_store(persist_dir: str, collection_name: str, embedding_model: Optional[str] = None, use_dummy: bool = False):
    # create embeddings - GoogleGenerativeAIEmbeddings likely uses environment credentials
    kwargs = {}
    # If embedding_model not provided, try env var, then default
    if not embedding_model:
        embedding_model = os.environ.get('GOOGLE_EMBEDDING_MODEL') or 'embed-text-embedding-3'
        logging.info('No embedding model provided, defaulting to: %s', embedding_model)
    if use_dummy:
        # Use a deterministic, lightweight embedding provider for tests/CI.
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
        kwargs['model'] = embedding_model
        try:
            embeddings = GoogleGenerativeAIEmbeddings(**kwargs)
        except Exception as e:
            logging.error("Failed to initialize GoogleGenerativeAIEmbeddings: %s", e)
            logging.error("Ensure your model name is valid and GOOGLE_API_KEY credentials are present. Try setting --google-embedding-model or env var GOOGLE_EMBEDDING_MODEL.")
            raise
    vectordb = Chroma(persist_directory=persist_dir, collection_name=collection_name, embedding_function=embeddings)
    return vectordb


def embed_pages(client: BookStackClient, vectordb: Chroma, text_splitter: RecursiveCharacterTextSplitter, page_limit: int = 0):
    pages = client.list_pages()
    if not pages:
        logging.info('No pages found to embed')
        return

    processed = 0
    for pg in tqdm(pages, desc='Pages'):
        page_id = pg.get('id')
        page_name = pg.get('name')
        # fetch page details to get html
        try:
            detail = client.get_page(page_id)
            html = detail.get('html')
        except Exception as e:
            logging.warning('Skipping page %s due to error: %s', page_id, e)
            continue

        book = pg.get('book') or {}
        book_id = book.get('id') if book else pg.get('book_id')
        book_name = book.get('name') if book else None
        page_url = f"{client.base_url}/pages/{pg.get('slug')}" if pg.get('slug') else None

        text = html_to_text(html or '')
        if not text:
            continue
        # Split into chunks
        docs: List[Document] = []
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            meta = {
                'book_id': book_id,
                'book_name': book_name,
                'page_id': page_id,
                'page_name': page_name,
                'page_url': page_url,
                'chunk_index': i,
            }
            # Filter out None values because Chroma expects primitive types only
            filtered_meta = {k: v for k, v in meta.items() if v is not None}
            docs.append(Document(page_content=chunk, metadata=filtered_meta))

        if not docs:
            continue
        # Upsert into Chroma
        vectordb.add_documents(docs)
        logging.info('Added %s chunks for page id %s', len(docs), page_id)
        processed += 1
        if page_limit and processed >= page_limit:
            logging.info('Reached page limit %s; stopping early', page_limit)
            break


def embed_sample_pages(vectordb: Chroma, text_splitter: RecursiveCharacterTextSplitter):
    logging.info('Embedding sample pages (no BookStack connection)')
    sample_pages = [
        {
            'page_id': 1,
            'page_name': 'Overview',
            'book_id': 1,
            'book_name': 'Sample Book',
            'html': '<p>This is an overview page for sample docs in BookStack.</p>'
        },
        {
            'page_id': 2,
            'page_name': 'Installation',
            'book_id': 1,
            'book_name': 'Sample Book',
            'html': '<p>Installation steps: run the installer and configure.</p>'
        },
    ]

    docs: List[Document] = []
    for pg in sample_pages:
        text = html_to_text(pg['html'])
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            meta = {
                'book_id': pg['book_id'],
                'book_name': pg['book_name'],
                'page_id': pg['page_id'],
                'page_name': pg['page_name'],
                'page_url': None,
                'chunk_index': i,
            }
            filtered_meta = {k: v for k, v in meta.items() if v is not None}
            docs.append(Document(page_content=chunk, metadata=filtered_meta))

    if not docs:
        logging.info('No sample docs to embed')
        return

    vectordb.add_documents(docs)
    logging.info('Added %s sample chunks', len(docs))


def main():
    parser = argparse.ArgumentParser(description='Embed BookStack pages with LangChain (GoogleGenerativeAIEmbeddings) and Chroma')
    parser.add_argument('--url', default='http://localhost:6875', help='Base URL of BookStack instance')
    parser.add_argument('--token-id', required=False, help='API token ID (falls back to BOOKSTACK_TOKEN_ID env var)')
    parser.add_argument('--token-secret', required=False, help='API token secret (falls back to BOOKSTACK_TOKEN_SECRET env var)')
    parser.add_argument('--header-format', help='Optional: force token header format (token, token_named, x-auth, bearer)')
    parser.add_argument('--persist-dir', default='./chroma_db', help='Chroma persist directory')
    parser.add_argument('--collection', default='bookstack_pages', help='Chroma collection name')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size for text splitting')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Chunk overlap for text splitting')
    parser.add_argument('--google-embedding-model', default=None, help='Optional embedding model name (provider-specific)')
    parser.add_argument('--limit', type=int, default=0, help='Optional: limit number of pages processed (0 = no limit)')
    parser.add_argument('--sample', action='store_true', help='Create and embed sample pages (no BookStack connection)')
    parser.add_argument('--force-reindex', action='store_true', help='Clear any existing vectors in the collection before embedding')
    parser.add_argument('--no-persist', action='store_true', help='If set, do not persist Chroma DB (useful for testing)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # Allow tokens/url to be supplied via environment variables (.env loaded above)
    token_id = args.token_id or os.environ.get('BOOKSTACK_TOKEN_ID')
    token_secret = args.token_secret or os.environ.get('BOOKSTACK_TOKEN_SECRET')
    url = args.url or os.environ.get('BOOKSTACK_URL') or 'http://localhost:6875'
    client = None
    if not args.sample:
        if not token_id or not token_secret:
            logging.error('Missing BookStack API credentials: pass --token-id and --token-secret or set BOOKSTACK_TOKEN_ID and BOOKSTACK_TOKEN_SECRET in env')
            sys.exit(1)
        client = BookStackClient(url, token_id, token_secret, header_format=args.header_format)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    embedding_model = args.google_embedding_model or os.environ.get('GOOGLE_EMBEDDING_MODEL')
    # If running sample mode without persisting, use in-memory Chroma (persist_directory=None)
    persist_dir_final = None if (args.sample and args.no_persist) else args.persist_dir
    collection_final = args.collection + '_sample' if args.sample and not args.no_persist else args.collection
    try:
        vectordb = build_vector_store(persist_dir_final, collection_final, embedding_model=embedding_model, use_dummy=args.sample)
    except Exception as e:
        logging.error('Failed to create vector store: %s', e)
        raise

    # If the user requested a full reindex, try to clear the collection safely
    if args.force_reindex:
        logging.info('Force reindex requested; attempting to clear existing collection: %s', collection_final)
        try:
            # Prefer client.delete_collection if available (Chroma client)
            client = getattr(vectordb, 'client', getattr(vectordb, '_client', None))
            if client and hasattr(client, 'delete_collection'):
                client.delete_collection(collection_final)
                logging.info('Deleted collection via client.delete_collection: %s', collection_final)
            else:
                # try the collection object delete method
                col = getattr(vectordb, '_collection', None)
                if col and hasattr(col, 'delete'):
                    col.delete()
                    logging.info('Cleared entries in collection via collection.delete')
                else:
                    logging.warning('Could not find client.delete_collection or collection.delete to clear data; attempting to re-create an empty collection by reinitializing vectordb and removing persistence directory')
                    # fallback to deleting disk persist dir if present
                    if persist_dir_final:
                        import shutil
                        try:
                            shutil.rmtree(persist_dir_final)
                            logging.info('Removed persist directory: %s', persist_dir_final)
                        except Exception as e:
                            logging.warning('Failed to remove persist directory: %s', e)
            # Recreate vectordb to ensure it points to fresh collection
            vectordb = build_vector_store(persist_dir_final, collection_final, embedding_model=embedding_model, use_dummy=args.sample)
        except Exception as e:
            logging.warning('Force reindex attempted but failed: %s', e)

    logging.info('Embedding pages now...')
    if args.sample:
        embed_sample_pages(vectordb, text_splitter)
    else:
        if client is None:
            logging.error('No BookStack client available; cannot embed pages. Use --sample to run sample mode without BookStack.')
            sys.exit(1)
        embed_pages(client, vectordb, text_splitter, page_limit=args.limit)
    # Persist to disk (Chroma will persist directory automatically)
    if not args.no_persist:
        try:
            vectordb.persist()
        except Exception:
            # some versions of LangChain/Chroma call this automatically or don't expose
            pass
    logging.info('Complete. Vectors stored in %s (collection: %s)', persist_dir_final or 'in-memory', collection_final)


if __name__ == '__main__':
    main()
