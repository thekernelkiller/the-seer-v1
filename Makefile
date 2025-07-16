venv:
	@python3.10 -m venv .venv
	@echo "Created Venv"

check_dirs := .

quality:
	black --check $(check_dirs)
	ruff check $(check_dirs)

style:
	black $(check_dirs)
	ruff check $(check_dirs) --fix

isort:
	black $(check_dirs)
	ruff check $(check_dirs) --select I --fix
