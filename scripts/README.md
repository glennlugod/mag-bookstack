# BookStack population script

This script boots a sample dataset into a BookStack instance running locally (default: http://localhost:6875) using an API token id and secret.

Usage
-----
Run using uv (recommended):

```bash
# Run directly; uv will create a managed environment and install declared dependencies
uv run scripts/populate_bookstack.py -- --token-id <ID> --token-secret <SECRET>

If automatic header discovery fails you can force a header format:

```bash
uv run scripts/populate_bookstack.py -- --header-format token --token-id <ID> --token-secret <SECRET>
```
```

Optional: add and lock the script dependencies using uv before running:

```bash
uv add --script scripts/populate_bookstack.py 'requests>=2.28'
uv lock --script scripts/populate_bookstack.py
uv run --script scripts/populate_bookstack.py -- --token-id <ID> --token-secret <SECRET>
```

Alternative: Install dependencies using pip and run directly (no uv):

```bash
python3 -m pip install -r scripts/requirements.txt
python3 scripts/populate_bookstack.py --token-id <ID> --token-secret <SECRET>
```

Options
-------
- `--url`  — Base URL of BookStack (default: `http://localhost:6875`)
- `--token-id` — **Required** API token ID
- `--token-secret` — **Required** API token secret
- `--template` — JSON file describing structure to create; if omitted a default dataset is used

JSON Template
-------------
Example file structure:

```json
{
  "books": [
    {
      "name": "Getting Started",
      "description": "A sample getting started book",
      "chapters": [
        {
          "name": "Introduction",
          "pages": [
            {"name": "Overview", "content": "<p>Overview page</p>"}
          ]
        }
      ],
      "pages": [
        {"name": "Extra", "content": "<p>Extra book-level page</p>"}
      ]
    }
  ]
}
```

Notes
-----
- The script attempts multiple token header formats to discover the one your BookStack instance uses. If it fails, double-check your token ID/secret and BookStack version.
- If you prefer, you can craft JSON input to create more detailed structure.
