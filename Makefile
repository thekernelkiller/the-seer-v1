# Makefile for The Seer

# --- Variables ---
VENV_DIR = .venv
VENV_ACTIVATE = . $(VENV_DIR)/bin/activate
PYTHON = $(VENV_DIR)/bin/python

# --- Phony targets to avoid conflicts with filenames ---
.PHONY: all setup services-up services-down run-backend run-frontend run quality style isort

# --- Default target ---
all: run

# --- Setup ---
setup:
	@echo "--- Setting up Python virtual environment ---"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		python3 -m venv $(VENV_DIR); \
		echo "Virtual environment created at $(VENV_DIR)"; \
	fi
	@echo "--- Activating virtual environment and installing dependencies ---"
	@$(VENV_ACTIVATE); \
	$(PYTHON) -m pip install --upgrade pip; \
	$(PYTHON) -m pip install pip-tools; \
	$(PYTHON) -m piptools sync requirements.txt; \
	echo "Dependencies installed successfully."
	@echo "\nSetup complete. You can now run the application."

# --- Docker Services ---
services-up:
	@echo "--- Starting Docker services (Redis, Postgres, Qdrant) ---"
	@docker compose up -d
	@echo "Services started."

services-down:
	@echo "--- Stopping Docker services ---"
	@docker compose down
	@echo "Services stopped."

# --- Application ---
run-backend:
	@echo "--- Starting FastAPI backend ---"
	@$(VENV_ACTIVATE); uvicorn main:app --reload

run-frontend:
	@echo "--- Starting Streamlit frontend ---"
	@$(VENV_ACTIVATE); streamlit run frontend/app.py

run:
	@echo "--- Welcome to The Seer ---"
	@echo "This command will guide you through setting up and running the application."
	@echo "\nStep 1: Setting up the environment..."
	@make setup
	@echo "\nStep 2: Starting background services..."
	@make services-up
	@echo "\n--- Setup Complete! ---"
	@echo "\nNow, you need to run the backend and frontend in separate terminal windows."
	@echo "\nIn your first terminal, run:"
	@echo "make run-backend"
	@echo "\nIn your second terminal, run:"
	@echo "make run-frontend"

# --- Code Quality ---
check_dirs := .
quality:
	@echo "--- Checking code quality ---"
	@$(VENV_ACTIVATE); black --check $(check_dirs)
	@$(VENV_ACTIVATE); ruff check $(check_dirs)

style:
	@echo "--- Formatting code ---"
	@$(VENV_ACTIVATE); black $(check_dirs)
	@$(VENV_ACTIVATE); ruff check $(check_dirs) --fix

isort:
	@echo "--- Sorting imports ---"
	@$(VENV_ACTIVATE); black $(check_dirs)
	@$(VENV_ACTIVATE); ruff check $(check_dirs) --select I --fix
