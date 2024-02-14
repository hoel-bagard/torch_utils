# torch_utils
PyTorch utils library



## Misc
### Installation
```
pip install -r requirements-dev.txt
pre-commit install
```

### Tests
A few tests have been written using Pytest. Once Pytest is installed, those tests can be run with:
```
python -m pytest -v
```

### Formatting
The code is trying to follow diverse PEPs conventions (notably PEP8). To have a similar dev environment you can install the following packages (pacman is for arch-based linux distros):
```
sudo pacman -S flake8 python-flake8-docstrings
pip install pep8-naming flake8-import-order flake8-bugbear flake8-quotes flake8-comprehensions
```

### Run pre-commit on all the files.
It's a good idea to run the hooks against all of the files when adding new hooks (usually pre-commit will only run on the changed files during git hooks).
```
pre-commit run --all-files
```
