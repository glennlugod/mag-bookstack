"""
Simple tests for webhook payload parsing utilities
"""
from scripts.webhook_utils import find_page_id_from_payload, normalize_detail_from_payload


def test_find_page_id_basic():
    payloads = [
        ({'page_id': 10}, 10),
        ({'pageId': '20'}, 20),
        ({'id': 30}, 30),
        ({'page': {'id': 40, 'html': '<p>hi</p>'}}, 40),
        ({'data': {'id': 50}}, 50),
        ({'related_item': {'id': 2432, 'name': 'My wonderful updated page'}}, 2432),
        ({'no_page': 3}, None),
    ]
    for payload, expected in payloads:
        res = find_page_id_from_payload(payload)
        assert res == expected, f"payload {payload} gave {res} expected {expected}"


def test_normalize_detail_from_related_item():
    sample = {
        "event": "page_update",
        "url": "https://bookstack.local/books/my-awesome-book/page/my-wonderful-updated-page",
        "related_item": {
            "id": 2432,
            "book_id": 13,
            "name": "My wonderful updated page",
            "slug": "my-wonderful-updated-page",
            "current_revision": {"id": 597, "summary": "Updated the title"}
        }
    }
    detail = normalize_detail_from_payload(sample)
    assert detail['id'] == 2432
    assert detail['slug'] == 'my-wonderful-updated-page'
    assert detail['name'] == 'My wonderful updated page'
    assert detail['book_id'] == 13
    assert detail['html'] is None


if __name__ == '__main__':
    test_find_page_id_basic()
    print('All tests OK')
