# Journal Mood Tracker

Local-first AI-powered journaling app with:
- FastAPI backend for journal storage and analysis
- Streamlit dashboard for writing and visualization
- HuggingFace models for sentiment and emotion extraction
- Ollama for local weekly/monthly reflective insights
- Sentence-transformers + FAISS for semantic search

## privacy_notice
All data stays on your machine.
- Journal entries are stored in a local SQLite file (`journal.db`).
- Sentiment, emotion, embeddings, and insight generation run locally.
- No cloud APIs or external LLM services are required.
- Model downloads happen once from provider registries and are cached locally for future runs.

## Project Structure
```text
journal-mood-tracker/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── database.py
│   ├── sentiment.py
│   ├── embeddings.py
│   └── insights.py
├── tests/
│   └── test_sentiment_and_insights.py
├── dashboard.py
├── requirements.txt
└── README.md
```

## Requirements
- Python 3.9+
- Ollama installed and running locally
- macOS/Linux/Windows (Apple Silicon supported)

## Setup
1. Create and activate a virtual environment.
2. Install dependencies.
3. Pull a local Ollama model.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3
```

Optional faster model on low-memory machines:
```bash
ollama pull qwen2.5:3b
```

If using a different model, set:
```bash
export OLLAMA_MODEL=qwen2.5:3b
```

## Run The API
```bash
python -m app.main
```

API will be available at `http://127.0.0.1:8000`.

### Safe Mode Startup Checks
You can run startup safety checks to verify local prerequisites before serving requests:

```bash
python -m app.main --safe-mode
```

Safe mode verifies:
- Ollama service is reachable
- The configured model (`OLLAMA_MODEL`, default `llama3`) is pulled locally
- The SQLite DB path is writable

You can also enable it with environment variable:

```bash
export SAFE_MODE=1
python -m app.main
```

### Optional Local Auth For Remote Access
By default, local requests are allowed without auth. For non-localhost access, set a token:

```bash
export LOCAL_API_TOKEN="your-local-token"
python -m app.main
```

Remote clients can then pass either:
- `Authorization: Bearer your-local-token`
- `X-API-Token: your-local-token`

## Seed Demo Data
Use demo mode to create 30 synthetic days of entries:

```bash
python -m app.main --demo
```

## Run Dashboard
In another terminal (with API running):

```bash
streamlit run dashboard.py
```

Dashboard will open in your browser.

## API Endpoints
- `POST /entries` create a dated entry and run analysis
- `GET /entries` list all entries
- `GET /entries/{date}` fetch one entry by `YYYY-MM-DD`
- `GET /insights/weekly` generate local 7-day insight
- `GET /insights/monthly` generate local 30-day insight
- `GET /search?q=...` semantic search across entries

## Tests
Run tests:

```bash
pytest -q
```

Current test coverage includes:
- Sentiment and emotion analysis flow (mocked pipelines)
- Insight generation with mocked Ollama client
- API integration tests for `/entries`, `/search`, `/insights/weekly`, `/insights/monthly`

## CI
GitHub Actions workflow is included at `.github/workflows/ci.yml` with:
- lint (`ruff`)
- tests (`pytest`)
- dependency vulnerability scan (`pip-audit`)

## Contributing
Contributions are welcome, including first-time contributions.

If you want to contribute:
- read `CONTRIBUTING.md`
- check open issues or propose a new one
- open a focused PR with test evidence

Project contribution and community files:
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `.github/pull_request_template.md`
- `.github/ISSUE_TEMPLATE/`

## Pull Requests
PRs are reviewed for:
- correctness and regressions
- privacy-first impact
- test coverage and maintainability

Please keep PRs scoped and explain the motivation clearly.

## Roadmap Ideas
Areas where contributions are especially useful:
- performance improvements for semantic search
- stronger API integration test coverage
- better UX for insight and timeline views
- packaging and local installer improvements


