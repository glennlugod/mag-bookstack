#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "Flask",
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
bookstack_webhook.py

Simple Flask-based webhook to receive BookStack events for page create / update
and re-index the affected page into the Chroma vector DB using the same logic
as `scripts/embed_bookstack.py`.

Usage (local dev):
  python scripts/bookstack_webhook.py --host 0.0.0.0 --port 5000 --token-id <id> --token-secret <secret>

Environment:
  - BOOKSTACK_TOKEN_ID / BOOKSTACK_TOKEN_SECRET - fallback credentials for BookStack API
  - BOOKSTACK_WEBHOOK_SECRET - *optional* secret to validate incoming webhook requests (header: X-BOOKSTACK-WEBHOOK-SECRET)
  - GOOGLE_EMBEDDING_MODEL - optional model for embeddings

This script reuses BookStackClient, build_vector_store and html_to_text from embed_bookstack.py
"""
from __future__ import annotations

import argparse
import logging
import os
# typing.Optional removed (not used in this module)

from flask import Flask, request, jsonify

from scripts.webhook_utils import find_page_id_from_payload, normalize_detail_from_payload
try:
    # import functions from embed_bookstack where possible
    from embed_bookstack import (
        BookStackClient,
        build_vector_store,
        html_to_text,
        RecursiveCharacterTextSplitter,
        Document,
    )
except Exception:
    # fallback to direct imports if script is executed from repo root
    from scripts.embed_bookstack import (
        BookStackClient,
        build_vector_store,
        html_to_text,
        RecursiveCharacterTextSplitter,
        Document,
    )


app = Flask(__name__)


# find_page_id_from_payload is implemented in scripts/webhook_utils.py


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/webhook/bookstack", methods=["POST"])
def bookstack_webhook():
    # Optional webhook secret validation
    secret = os.environ.get('BOOKSTACK_WEBHOOK_SECRET')
    if secret:
        header_secret = request.headers.get('X-BOOKSTACK-WEBHOOK-SECRET')
        if not header_secret or header_secret != secret:
            logging.warning('Rejected webhook due to invalid secret')
            return jsonify({"error": "unauthorized"}), 401

    try:
        payload = request.get_json(force=True)
    except Exception as e:
        logging.error('Invalid JSON in webhook: %s', e)
        return jsonify({"error": "invalid_json"}), 400

    # Optionally ignore events we don't care about
    event = payload.get('event') or payload.get('action')
    if event:
        event_lower = str(event).lower()
        if not (('page' in event_lower) and any(k in event_lower for k in ('create', 'update', 'revision', 'restore', 'edit'))):
            logging.info('Ignoring event: %s', event)
            return jsonify({"status": "ignored", "event": event}), 200

    page_id = find_page_id_from_payload(payload)
    if not page_id:
        logging.warning('No page id found in payload')
        return jsonify({"error": "no_page_id"}), 400

    # Re-index the single page
    detail = None

    try:
        # Use environment-provided BookStack credentials if available, else try to obtain html from payload
        token_id = app.config.get('bookstack_token_id')
        token_secret = app.config.get('bookstack_token_secret')
        if token_id and token_secret:
            client = BookStackClient(app.config.get('bookstack_url'), token_id, token_secret, header_format=app.config.get('bookstack_header_format'))
            detail = client.get_page(page_id)
        else:
            # If request contains page data (like html), use it rather than calling the API
            # If payload contains page info, normalize via helper
            # Use normalize_detail_from_payload helper to build a detail-like dict
            detail = normalize_detail_from_payload(payload)
            if not detail or not detail.get('id'):
                logging.error('No API credentials and no page html in payload; cannot fetch page %s', page_id)
                return jsonify({"error": "missing_credentials"}), 400
    except Exception as e:
        logging.error('Failed to fetch page %s: %s', page_id, e)
        return jsonify({"error": "fetch_failed", "message": str(e)}), 500

    html = detail.get('html') if isinstance(detail, dict) else None
    if not html:
        logging.warning('Page %s had no html content to index', page_id)
        return jsonify({"status": "no_content"}), 200

    text = html_to_text(html) if html else None
    if not text:
        logging.info('Page %s contained no text after conversion', page_id)
        return jsonify({"status": "no_text"}), 200

    # Create text splitter using settings configured in app.config
    chunk_size = app.config.get('chunk_size', 1000)
    chunk_overlap = app.config.get('chunk_overlap', 200)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Create vector store if not already created
    vectordb = app.config.get('vectordb')
    if not vectordb:
        vectordb = build_vector_store(app.config.get('persist_dir'), app.config.get('collection'), embedding_model=app.config.get('embedding_model'))
        app.config['vectordb'] = vectordb

    # Attempt to delete existing chunks for the page
    deleted = False
    try:
        col = getattr(vectordb, '_collection', None) or getattr(vectordb, 'collection', None)
        if col and hasattr(col, 'delete'):
            # Chromadb collection delete supports `where` filter
            try:
                col.delete(where={"page_id": page_id})
                deleted = True
            except Exception as e:
                logging.info('Unable to delete by metadata via col.delete: %s', e)
        # Some wrappers expose delete_documents on vectordb
        if not deleted and hasattr(vectordb, 'delete'):
            try:
                vectordb.delete(where={"page_id": page_id})
                deleted = True
            except Exception as e:
                logging.info('vectordb.delete(where={}) not supported: %s', e)
    except Exception as e:
        logging.info('Failed to clear existing data for page %s: %s', page_id, e)

    # Build docs and add to the collection
    chunks = text_splitter.split_text(text) if text else []
    docs = []
    for i, chunk in enumerate(chunks):
        meta = {
            'book_id': detail.get('book_id'),
            'book_name': detail.get('book', {}).get('name') if detail.get('book') else None,
            'page_id': page_id,
            'page_name': detail.get('name'),
            'page_url': detail.get('url') or (f"{app.config.get('bookstack_url')}/pages/{detail.get('slug')}" if detail.get('slug') else None),
            'chunk_index': i,
        }
        filtered_meta = {k: v for k, v in meta.items() if v is not None}
        docs.append(Document(page_content=chunk, metadata=filtered_meta))

    if not docs:
        logging.warning('No docs to add for page %s', page_id)
        # If no docs because there was no text (but we had a fallback detail metadata), record as no_content
        if not html:
            logging.info('No html/content provided for page %s; skipping index', page_id)
            return jsonify({"status": "no_content", "page_id": page_id}), 200
        return jsonify({"status": "no_docs"}), 200

    try:
        vectordb.add_documents(docs)
        # Persist if backing directory is set
        if app.config.get('persist_dir'):
            try:
                vectordb.persist()
            except Exception:
                pass
    except Exception as e:
        logging.error('Failed to add documents for page %s: %s', page_id, e)
        return jsonify({"error": "index_failed", "message": str(e)}), 500

    logging.info('Indexed %s chunks for page %s (deleted prior: %s)', len(docs), page_id, deleted)
    return jsonify({"status": "ok", "page_id": page_id, "chunks": len(docs), "deleted_prior": deleted}), 200


def cli_run():
    parser = argparse.ArgumentParser(description='Run BookStack webhook server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--url', default='http://localhost:6875', help='BookStack base URL')
    parser.add_argument('--token-id', required=False, help='API token ID (falls back to BOOKSTACK_TOKEN_ID env var)')
    parser.add_argument('--token-secret', required=False, help='API token secret (falls back to BOOKSTACK_TOKEN_SECRET env var)')
    parser.add_argument('--header-format', help='Optional: force token header format (token, token_named, x-auth, bearer)')
    parser.add_argument('--persist-dir', default='./chroma_db', help='Chroma persist directory')
    parser.add_argument('--collection', default='bookstack_pages', help='Chroma collection name')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size for text splitting')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Chunk overlap for text splitting')
    parser.add_argument('--google-embedding-model', default=None, help='Optional embedding model name (provider-specific)')
    parser.add_argument('--no-persist', action='store_true', help='If set, do not persist Chroma DB (useful for testing)')
    parser.add_argument('--use-dummy-embeddings', action='store_true', help='Use DummyEmbeddingsLocal instead of external provider (useful for dev/tests)')
    parser.add_argument('--force-reindex', action='store_true', help='Clear entire collection before processing events')
    args = parser.parse_args()

    # Setup app config and vectordb
    token_id = args.token_id or os.environ.get('BOOKSTACK_TOKEN_ID')
    token_secret = args.token_secret or os.environ.get('BOOKSTACK_TOKEN_SECRET')
    url = args.url or os.environ.get('BOOKSTACK_URL') or 'http://localhost:6875'
    embedding_model = args.google_embedding_model or os.environ.get('GOOGLE_EMBEDDING_MODEL')
    persist_dir_final = None if args.no_persist else args.persist_dir
    collection_final = args.collection
    # Create vectordb
    vectordb = build_vector_store(persist_dir_final, collection_final, embedding_model=embedding_model, use_dummy=args.use_dummy_embeddings)
    if args.force_reindex:
        # Attempt to wipe collection similar to embed_bookstack
        try:
            client = getattr(vectordb, 'client', getattr(vectordb, '_client', None))
            if client and hasattr(client, 'delete_collection'):
                client.delete_collection(collection_final)
            else:
                col = getattr(vectordb, '_collection', None)
                if col and hasattr(col, 'delete'):
                    col.delete()
                elif persist_dir_final:
                    import shutil
                    shutil.rmtree(persist_dir_final, ignore_errors=True)
            # Recreate vectordb
            vectordb = build_vector_store(persist_dir_final, collection_final, embedding_model=embedding_model, use_dummy=False)
        except Exception as e:
            logging.warning('Force reindex requested but failed: %s', e)

    # store configuration in the Flask app context
    app.config.update({
        'bookstack_token_id': token_id,
        'bookstack_token_secret': token_secret,
        'bookstack_header_format': args.header_format,
        'bookstack_url': url,
        'persist_dir': persist_dir_final,
        'collection': collection_final,
        'embedding_model': embedding_model,
        'vectordb': vectordb,
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
    })

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Starting BookStack webhook server on %s:%s', args.host, args.port)
    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    cli_run()
