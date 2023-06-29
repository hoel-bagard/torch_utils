# torch_utils
PyTorch utils library

## Installation

```console
pip install hbtorch-utils
```

### Local Installation
### Requirements
- Python>=3.11
- Poetry

### Pip Dependencies

If simply using the package (for torch, you can install the version you want with pip instead of using `--with torch`):
```console
poetry install --with torch & poetry shell
```

If developing:
```console
poetry install --with dev,test,torch & poetry shell
```

### Tests
A few tests have been written using Pytest. Once the test dependencies are installed, those tests can be run with:
```
poetry run pytest tests
```
