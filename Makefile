pytest:
	python3 -m pytest tests

pytest-slow:
	python3 -m pytest tests -vv -m slow --runslow

coverage:
	python3 -m pytest --cov=rocketpy tests

coverage-report:
	python3 -m pytest --cov=rocketpy tests --cov-report html

install:
	python3 -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-optional.txt
	pip install -e .

isort:
	isort --profile black rocketpy/ tests/ docs/

black:
	black rocketpy/ tests/ docs/
	
pylint:
	-pylint rocketpy tests --output=.pylint-report.txt

build-docs:
	cd docs && python3 -m pip install -r requirements.txt && make html
	cd ..