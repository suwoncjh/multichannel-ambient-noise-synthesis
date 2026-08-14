# Contributing

Install the development dependencies and run the full test suite before submitting changes:

```bash
pip install -e ".[dev]"
pytest -q
```

For synthesis changes, also run the static and advanced examples and compare SCM/coherence metrics.
