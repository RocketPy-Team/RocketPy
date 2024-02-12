test:
	python -m pytest tests -vv

testfile:
	python -m pytest tests/$(file) -vv

tests:
	test

coverage: 
	python -m pytest --cov=rocketpy tests -vv

coverage-report:
	python -m pytest --cov=rocketpy tests -vv --cov-report html

install: 
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	python setup.py install

verify-lint:
	flake8 --select BLK rocketpy
	flake8 --select BLK test

lint:
	black rocketpy
	black tests
