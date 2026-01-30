# Contributing

Thank you for your interest in contributing to Portfolio Analysis!

## Ways to Contribute

- **Report bugs**: Open an issue with a minimal reproducible example
- **Suggest features**: Open an issue describing your use case
- **Submit PRs**: Fix bugs or implement new features
- **Improve docs**: Fix typos, add examples, clarify explanations
- **Share**: Star the repo, tweet about it, write a blog post

## Development Setup

```bash
# Clone the repo
git clone https://github.com/engineerinvestor/Portfolio-Analysis.git
cd Portfolio-Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with dev dependencies
pip install -e ".[all]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=portfolio_analysis --cov-report=html

# Run specific test file
pytest tests/test_factors.py -v
```

## Code Style

We use:

- **Black** for code formatting
- **Ruff** for linting
- **NumPy-style** docstrings

```bash
# Format code
black portfolio_analysis/

# Check linting
ruff check portfolio_analysis/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Add tests for new functionality
5. Run tests and linting
6. Commit with a clear message
7. Push and open a PR

## Commit Messages

Use clear, descriptive commit messages:

```
Add rolling factor regression support

- Implement run_rolling_regression() method
- Add window parameter for configurable lookback
- Include tests for edge cases
```

## Adding New Features

When adding new features:

1. **Discuss first**: Open an issue to discuss the design
2. **Write tests**: Aim for >80% coverage of new code
3. **Add docstrings**: NumPy-style with Parameters/Returns sections
4. **Update docs**: Add to user guide and API reference
5. **Add example**: Show how to use the feature

## Questions?

- Open a [GitHub Discussion](https://github.com/engineerinvestor/Portfolio-Analysis/discussions)
- Tweet [@egr_investor](https://twitter.com/egr_investor)
