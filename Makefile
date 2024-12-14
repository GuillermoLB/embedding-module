#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = embedding-module
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 embedding_module
	isort --check --diff --profile black embedding_module
	black --check --config pyproject.toml embedding_module

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml embedding_module




## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	python3 -m venv venv
    @echo ">>> New virtualenv created. Activate with:\nsource venv/bin/activate"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) embedding_module/data_processing/dataset.py

## Generate Embeddings
.PHONY: embed
embed:
	$(PYTHON_INTERPRETER) embedding_module/embedding/embedder.py

## Build Index
.PHONY: index
index:
	$(PYTHON_INTERPRETER) embedding_module/indexing/indexer.py build-index

## Query Index
.PHONY: query
query:
	$(PYTHON_INTERPRETER) embedding_module/indexing/indexer.py query-index --query-embedding-file path/to/query_embedding.json --top-k 5


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
