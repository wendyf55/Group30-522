
.DEFAULT_GOAL := help

#make all target
.PHONY: all
all: reports/abalone_rings.html

.PHONY: help
help: # Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: cl
cl: # create conda lock for multiple platforms
	# the linux-aarch64 is used for ARM Macs using linux docker container
	conda-lock lock --file environment.yaml -p linux-64 -p osx-64 -p osx-arm64 -p win-64 -p linux-aarch64

.PHONY: env
env: # remove previous and create environment from lock file
	# remove the existing env, and ignore if missing
	conda env remove -n group30-522 || true
	conda-lock install -n group30-522 conda-lock.yml

.PHONY: build
build: # build the docker image from the Dockerfile
	docker build -t group30-522 --file Dockerfile .

.PHONY: run
run: # alias for the up target
	$(MAKE) up

.PHONY: up
up: # stop and start docker-compose services
	# by default stop everything before re-creating
	$(MAKE) stop
	docker-compose up -d

.PHONY: stop
stop: # stop docker-compose services
	docker-compose stop


.PHONY: test
test: # Run unit tests with pytest
	python -m pytest tests/ -v


# Download  data
data/raw/abalone.data: scripts/download_data.py
	python scripts/download_data.py

# Clean + split data
data/processed/abalone_train.csv data/processed/abalone_test.csv: \
scripts/data_cleaning.py data/raw/abalone.data
	python scripts/data_cleaning.py \
		--origin_path data/raw/abalone.data \
		--output_dir data/processed

# Data validation 
results/data_validation/target_boxplot.png \
results/data_validation/target_histogram.png \
results/data_validation/correlation_plot.png: \
scripts/data_validation.py data/processed/abalone_train.csv
	python scripts/data_validation.py

# EDA
results/eda/descriptive_stats.csv \
results/eda/pandas_profiling.html \
results/eda/interaction_plot.png: \
scripts/eda.py data/processed/abalone_train.csv
	python scripts/eda.py

# Model 
results/model/model_results_actual_vs_predicted.png \
results/model/model_results_metrics.csv \
results/model/model_results_model_comparison.png \
results/model/model_results_residuals.png \
results/model/model_results_lr_scatter.png \
results/model/model_results_lr_coefficients.csv \
results/model/model_results_predictions.csv \
results/model/model_results_models.pkl: \
scripts/train_model.py \
data/processed/abalone_train.csv \
data/processed/abalone_test.csv
	python scripts/train_model.py

# make report
reports/abalone_rings.html: \
reports/abalone_rings.qmd \
results/data_validation/target_boxplot.png \
results/data_validation/target_histogram.png \
results/data_validation/correlation_plot.png \
results/eda/pandas_profiling.html \
results/eda/interaction_plot.png \
results/model/model_results_metrics.csv
	quarto render reports/abalone_rings.qmd

# Clean 
.PHONY: clean
clean: # Remove all generated data and results
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf results/data_validation/*
	rm -rf results/eda/*
	rm -rf results/model/*
	rm -f reports/abalone_rings.html


