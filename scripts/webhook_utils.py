"""
Helpers for parsing webhook payloads for BookStack
"""
from typing import Optional, Dict


def find_page_id_from_payload(payload: Dict) -> Optional[int]:
    """Return the page id found in many BookStack webhook payload structures.

    Recognizes 'page_id', 'pageId', 'id', and 'related_item': { 'id': ... }.
    """
    if not isinstance(payload, dict):
        return None
    # top-level keys
    for k in ('page_id', 'pageId', 'id'):
        if k in payload and payload[k] is not None:
            try:
                return int(payload[k])
            except Exception:
                continue

    # top-level nested fields
    if 'page' in payload and isinstance(payload['page'], dict):
        pid = payload['page'].get('id') or payload['page'].get('page_id')
        if pid:
            try:
                return int(pid)
            except Exception:
                pass

    nested_candidates = ("data", "entity", "related_item", "related")
    for nc in nested_candidates:
        if nc in payload and isinstance(payload[nc], dict):
            for k in ("page_id", "pageId", "id"):
                if k in payload[nc] and payload[nc][k] is not None:
                    try:
                        return int(payload[nc][k])
                    except Exception:
                        continue
            # Special-case BookStack webhook format: { related_item: { id: ... } }
            if nc in ("related_item", "related"):
                ri = payload[nc]
                if isinstance(ri, dict) and ri.get('id'):
                    try:
                        return int(ri.get('id'))
                    except Exception:
                        pass
    return None


def normalize_detail_from_payload(payload: Dict) -> Dict:
    """Normalize a payload's 'related_item' or nested page data into a detail-like dict.

    If the payload contains 'related_item', map the expected keys to a normalized dictionary
    that approximate what BookStackClient.get_page() returns (id/name/slug/book_id/html).
    """
    if not isinstance(payload, dict):
        return {}
    page_payload = payload.get('page') or payload.get('data') or payload.get('related_item') or payload.get('related') or {}
    if not isinstance(page_payload, dict):
        return {}
    detail = {
        'id': page_payload.get('id'),
        'name': page_payload.get('name'),
        'slug': page_payload.get('slug') or page_payload.get('book_slug') or None,
        'book_id': page_payload.get('book_id') or page_payload.get('book', {}).get('id') if isinstance(page_payload.get('book'), dict) else page_payload.get('book_id'),
        'book': {'id': page_payload.get('book_id'), 'name': None},
        'html': page_payload.get('html') or (page_payload.get('current_revision', {}) or {}).get('html'),
    }
    return detail
