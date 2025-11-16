#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "requests>=2.28",
# ]
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///
"""
populate_bookstack.py

Simple script to populate a local BookStack instance using the API and a token id/secret.

Usage:
  python scripts/populate_bookstack.py --token-id <id> --token-secret <secret> [--url http://localhost:6875]

The script will try a few common token header formats and fall back to helpful diagnostics
if authentication fails.

It creates a sample dataset: 2 books (each with a chapter and pages). You can provide a
JSON template file to create a custom structure.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests


@dataclass
class BookItem:
    id: int
    name: str
    slug: Optional[str] = None


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
        """Try a few token header formats and return the one that works.

        We attempt the most likely formats and accept the first that yields 2xx on GET /api/books
        to verify authentication.
        """
        trial_formats = [
            # Common BookStack API token header format (id:secret)
            lambda t_id, t_secret: {"Authorization": f"Token {t_id}:{t_secret}"},
            # Another reasonable variant
            lambda t_id, t_secret: {"Authorization": f"Token token={t_id}:{t_secret}"},
            # Sometimes an "X-Auth-Token" header is used or simple bearer-like header
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
                print(f"Using header format: {list(h.keys())} -> {list(h.values())}")
                return h

        # None of the headers worked — return a default and caller can override
        raise RuntimeError(
            "Unable to discover valid token header for BookStack API. Please verify token ID and secret or consult the docs."
        )

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = kwargs.pop('headers', {})
        headers.update(self.auth_header)
        r = self.session.request(method, url, headers=headers, timeout=self.timeout, **kwargs)
        r.raise_for_status()
        return r

    # Resource creation helpers
    def create_book(self, name: str, description: Optional[str] = None) -> BookItem:
        body = {'name': name}
        if description:
            body['description'] = description
        r = self._request('POST', '/api/books', json=body)
        data = r.json().get('data') or r.json()
        return BookItem(id=data['id'], name=data['name'], slug=data.get('slug'))

    def create_chapter(self, name: str, book_id: int, description: Optional[str] = None) -> BookItem:
        body = {'name': name, 'book_id': book_id}
        if description:
            body['description'] = description
        r = self._request('POST', '/api/chapters', json=body)
        data = r.json().get('data') or r.json()
        return BookItem(id=data['id'], name=data['name'], slug=data.get('slug'))

    def create_page(self, name: str, book_id: Optional[int] = None, chapter_id: Optional[int] = None, html: Optional[str] = None) -> BookItem:
        body: Dict[str, object] = {'name': name}
        if chapter_id:
            body['chapter_id'] = chapter_id
        elif book_id:
            body['book_id'] = book_id
        if html:
            body['html'] = html
        r = self._request('POST', '/api/pages', json=body)
        data = r.json().get('data') or r.json()
        return BookItem(id=data['id'], name=data['name'], slug=data.get('slug'))


def load_template(template_path: Optional[str]) -> dict:
    if not template_path:
        return {
            "books": [
                {
                    "name": "Getting Started",
                    "description": "A sample getting started book",
                    "chapters": [
                        {
                            "name": "Introduction",
                            "pages": [
                                {"name": "Overview", "content": "<p>Overview page</p>"},
                                {"name": "Quick Start", "content": "<p>Quick start content</p>"},
                            ],
                        }
                    ],
                },
                {
                    "name": "HowTos",
                    "description": "Sample how-to book",
                    "pages": [
                        {"name": "Install", "content": "<p>Install steps</p>"},
                        {"name": "Configure", "content": "<p>Configuration steps</p>"},
                    ],
                },
            ]
        }

    with open(template_path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def populate(client: BookStackClient, template: dict) -> dict:
    created: Dict[str, List[Dict]] = {'books': [], 'chapters': [], 'pages': []}
    for book_def in template.get('books', []):
        print(f"Creating book: {book_def.get('name')}")
        book = client.create_book(book_def.get('name'), book_def.get('description'))
        created['books'].append({'id': book.id, 'name': book.name})

        # Create chapters
        for ch_def in book_def.get('chapters', []):
            print(f"  Creating chapter: {ch_def.get('name')}")
            chap = client.create_chapter(ch_def.get('name'), book.id, ch_def.get('description'))
            created['chapters'].append({'id': chap.id, 'name': chap.name, 'book_id': book.id})

            for p_def in ch_def.get('pages', []):
                print(f"    Creating page: {p_def.get('name')}")
                page = client.create_page(p_def.get('name'), chapter_id=chap.id, html=p_def.get('content'))
                created['pages'].append({'id': page.id, 'name': page.name, 'chapter_id': chap.id})

        # Create pages at book root
        for p_def in book_def.get('pages', []):
            print(f"  Creating book-level page: {p_def.get('name')}")
            page = client.create_page(p_def.get('name'), book_id=book.id, html=p_def.get('content'))
            created['pages'].append({'id': page.id, 'name': page.name, 'book_id': book.id})

    return created


def main():
    parser = argparse.ArgumentParser(description='Populate a local BookStack instance using the API token.')
    parser.add_argument('--url', default='http://localhost:6875', help='Base URL of BookStack instance')
    parser.add_argument('--token-id', required=True, help='API token ID')
    parser.add_argument('--token-secret', required=True, help='API token secret')
    parser.add_argument('--header-format', help='Optional: force token header format (token, token_named, x-auth, bearer)')
    parser.add_argument('--template', help='JSON template file for content structure')
    args = parser.parse_args()

    template = load_template(args.template)
    try:
        client = BookStackClient(args.url, args.token_id, args.token_secret, header_format=args.header_format)
    except RuntimeError as e:
        print(f"Authentication error: {e}")
        sys.exit(1)

    created = populate(client, template)
    print('\nCreated resources:')
    print(json.dumps(created, indent=2))


if __name__ == '__main__':
    main()
