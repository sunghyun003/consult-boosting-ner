.PHONY: install install-dev update start
.DEFAULT_GOAL := help

APP_ENV ?= dev
HOST ?= 0.0.0.0
PORT ?= 8000

help:
	@echo "make install                 - install (without dev)"
	@echo "make install-dev             - install (with dev)"
	@echo "make update                  - update all dependencies"
	@echo "make start                   - run uvicorn with --reload"

install:
	poetry install --without dev

install-dev:
	poetry install --with dev

update:
	poetry update

start:
	APP_ENV=$(APP_ENV) poetry run uvicorn module_ner_infer.main:app \
		--host "$(HOST)" --port "$(PORT)" --reload
