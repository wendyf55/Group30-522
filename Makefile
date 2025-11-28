.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: all
all: ## runs the targets: cl, env, build
	$(MAKE) cl
	$(MAKE) env
	$(MAKE) build

.PHONY: cl
cl: ## create conda lock for multiple platforms
	# the linux-aarch64 is used for ARM Macs using linux docker container
	conda-lock lock --file environment.yaml -p linux-64 -p osx-64 -p osx-arm64 -p win-64 -p linux-aarch64

.PHONY: env
env: ## remove previous and create environment from lock file
	# remove the existing env, and ignore if missing
	conda env remove -n group30-522 || true
	conda-lock install -n group30-522 conda-lock.yml

.PHONY: build
build: ## build the docker image from the Dockerfile
	docker build -t group30-522 --file Dockerfile .

.PHONY: run
run: ## alias for the up target
	$(MAKE) up

.PHONY: up
up: ## stop and start docker-compose services
	# by default stop everything before re-creating
	$(MAKE) stop
	docker-compose up -d

.PHONY: stop
stop: ## stop docker-compose services
	docker-compose stop