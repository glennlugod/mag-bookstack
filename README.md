# mag - BookStack (with uv-managed script)

This repository includes a Docker Compose setup for BookStack and a small utility script to populate a BookStack instance with sample content.

Start the application (uses linuxserver/bookstack):

```bash
docker compose up -d
```

Using uv to run the population script (recommended):

```bash
# Run the script using uv (requests will be installed by uv into a managed env)
uv run scripts/populate_bookstack.py -- --token-id <ID> --token-secret <SECRET>
```

If you don't use uv, install requirements and run with Python 3.11:

```bash
python3.11 -m pip install -r scripts/requirements.txt
python3.11 scripts/populate_bookstack.py --token-id <ID> --token-secret <SECRET>
```

The default BookStack instance is exposed at http://localhost:6875

