# Convenience targets. These assume an activated environment (pixi or venv).
.PHONY: help install test lint format clean demo produce consume dashboard

help:
	@echo "Targets:"
	@echo "  install    Install the package + dev/stream extras (pip)"
	@echo "  test       Run pytest (pure-Python known-answer tests always run)"
	@echo "  demo       Run the seeded synthetic demo end-to-end (real numbers)"
	@echo "  lint       ruff check + mypy"
	@echo "  format     ruff format"
	@echo "  produce    Publish events onto the Kafka topic (needs the stack)"
	@echo "  consume    Read back events from the Kafka topic (needs the stack)"
	@echo "  dashboard  Launch the Streamlit dashboard"
	@echo "  clean      Remove caches and build artefacts"

install:
	pip install -e ".[dev,stream]"

test:
	pytest

demo:
	clickstream demo

lint:
	ruff check .
	mypy src

format:
	ruff format .

produce:
	python -m clickstream.pipeline

consume:
	python -c "from clickstream.pipeline import consume_events; print(len(consume_events(topic='clickstream.events')))"

dashboard:
	streamlit run app/dashboard.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
