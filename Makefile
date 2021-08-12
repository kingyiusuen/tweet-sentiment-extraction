.PHONY: install
install:
	pip install -e . --no-cache-dir

.PHONY: install-dev
install-dev:
	pip install -e ".[dev]" --no-cache-dir

.PHONY: venv
venv:
	python3 -m venv venv
	source venv/bin/activate
	python -m pip install --upgrade pip setuptools wheel

.PHONY: lint
lint:
	isort .
	black .
	flake8 .

.PHONY: streamlit
streamlit:
	streamlit run streamlit/app.py

.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -rf .mypy_cache