# Set PYTHON variable according to OS
ifeq ($(OS),Windows_NT)
	PYTHON=python
else
	PYTHON=python3
endif

pytest:
	$(PYTHON) -m pytest tests

pytest-slow:
	$(PYTHON) -m pytest tests -vv -m slow --runslow

coverage:
	$(PYTHON) -m pytest --cov=rocketpy tests

coverage-report:
	$(PYTHON) -m pytest --cov=rocketpy tests --cov-report html

install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-optional.txt
	pip install -e .

format:
	ruff check --select I --fix rocketpy/ tests/ docs/
	ruff format rocketpy/ tests/ docs/

lint: ruff-lint pylint

ruff-lint:
	ruff check rocketpy/ tests/ docs/ --output-file=.ruff-report.txt

pylint:
	-pylint rocketpy/ tests/ docs/ --output=.pylint-report.txt

build-docs:
	cd docs && $(PYTHON) -m pip install -r requirements.txt && make html
	cd ..