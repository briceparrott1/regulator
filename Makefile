# Regulator — DOC-flag-driven parse helpers.
#
# Usage:
#   make parse DOC=<flag>   parse one regulatory PDF into output/<flag>/
#   make clear DOC=<flag>   remove output/<flag>/
#
# <flag> is a case-insensitive substring matched against the filenames in
# data/regulations/*.pdf (the matching itself lives in regulator/parsing.py).

PYTHON := .venv/bin/python

.DEFAULT_GOAL := help

.PHONY: help parse parse-sop clear

help:
	@echo "Regulator make targets:"
	@echo "  make parse DOC=<flag>   Parse one regulatory PDF into output/<flag>/"
	@echo "  make parse-sop          Parse data/sop/original.docx into output/sop/"
	@echo "  make clear DOC=<flag>   Delete output/<flag>/ (e.g. DOC=sop)"
	@echo "  make help               Show this message"
	@echo ""
	@echo "<flag> is a case-insensitive substring of a data/regulations/*.pdf name."

parse-sop:
	$(PYTHON) -c "from pathlib import Path; from dotenv import load_dotenv; \
from regulator.parsing import parse_operating_procedure; load_dotenv(); \
d = parse_operating_procedure(Path('data/sop/original.docx'), Path('output/sop')); \
print(f'Parsed SOP {d.doc_id!r}: {len(d.nodes)} nodes -> output/sop/{d.doc_id}.jsonl')"

parse:
ifndef DOC
	$(error DOC is not set. Usage: make parse DOC=<flag>)
endif
	$(PYTHON) -m regulator.parse_cli "$(DOC)" --out "output/$(DOC)"

clear:
ifndef DOC
	$(error DOC is not set. Usage: make clear DOC=<flag>)
endif
	@if [ -d "output/$(DOC)" ]; then \
		rm -rf "output/$(DOC)"; \
		echo "Removed output/$(DOC)"; \
	else \
		echo "Nothing to clear: output/$(DOC) does not exist."; \
	fi
