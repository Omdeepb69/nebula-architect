# Contributing to NEBULA Architect

Thank you for your interest in contributing to NEBULA Architect! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read it before contributing.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in the Issues section
2. If not, create a new issue with a clear and descriptive title
3. Include as much relevant information as possible:
   - Steps to reproduce the bug
   - Expected behavior
   - Actual behavior
   - Screenshots if applicable
   - System information (OS, Python version, etc.)

### Suggesting Features

1. Check if the feature has already been suggested in the Issues section
2. If not, create a new issue with a clear and descriptive title
3. Provide a detailed description of the proposed feature
4. Explain why this feature would be useful
5. Include any relevant examples or mockups

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Write or update tests as needed
5. Ensure all tests pass
6. Update documentation if necessary
7. Submit a pull request

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nebula-architect.git
cd nebula-architect
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

### Code Style

- Follow PEP 8 style guide
- Use type hints
- Write docstrings for all functions and classes
- Keep lines under 100 characters
- Use meaningful variable and function names

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting a pull request
- Run tests with:
```bash
pytest
```

### Documentation

- Update README.md if necessary
- Add docstrings to new functions and classes
- Update API documentation if needed
- Keep examples up to date

## Project Structure

```
nebula-architect/
├── src/
│   ├── core/                 # Core system components
│   ├── models/              # AI model implementations
│   ├── rendering/           # 3D rendering and visualization
│   ├── simulation/          # Physics and agent simulation
│   ├── audio/              # Speech processing
│   └── utils/              # Utility functions
├── tests/                  # Test suite
├── examples/               # Example usage and demos
├── configs/               # Configuration files
└── docs/                  # Documentation
```

## Commit Messages

- Use clear and descriptive commit messages
- Reference issues and pull requests when applicable
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")

## Review Process

1. All pull requests will be reviewed by maintainers
2. Reviews may request changes
3. Once approved, a maintainer will merge the pull request

## License

By contributing to NEBULA Architect, you agree that your contributions will be licensed under the project's MIT License. 