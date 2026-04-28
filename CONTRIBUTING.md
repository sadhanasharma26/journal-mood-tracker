# Contributing to Journal Mood Tracker

Thanks for your interest in contributing.

This project is local-first and privacy-focused. Please keep that principle in mind for all proposed changes.

## Ways to Contribute
- Report bugs with clear reproduction steps.
- Propose UX, performance, and reliability improvements.
- Improve tests and docs.
- Submit pull requests for open issues or your own ideas.

## Ground Rules
- Be respectful and constructive.
- Prefer small, focused pull requests.
- Discuss larger changes in an issue before implementation.
- Do not add cloud dependencies or telemetry without explicit discussion.

## Development Setup
1. Fork and clone your fork.
2. Create and activate a virtual environment.
3. Install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

4. Run tests.

```bash
pytest -q
```

5. Start backend and dashboard locally.

```bash
python -m app.main --demo
streamlit run dashboard.py
```

## Coding Expectations
- Keep changes privacy-first and local-first.
- Add or update tests for behavior changes.
- Keep commits readable and scoped.
- Use clear names and concise comments.
- Maintain compatibility with the existing project structure.

## Pull Request Process
1. Create a branch from `main`.
2. Make your changes and add tests.
3. Run checks locally:

```bash
ruff check . --select=E9,F63,F7,F82
pytest -q
```

4. Open a pull request with:
- clear summary
- motivation/problem statement
- test evidence
- screenshots for UI changes (if relevant)

## PR Review Criteria
- Correctness and no regressions
- Readability and maintainability
- Adequate tests
- Privacy and security impact considered

## Security Notes
If you find a security issue, please avoid public disclosure details in an issue. Open a minimal issue asking maintainers for a secure reporting channel.
