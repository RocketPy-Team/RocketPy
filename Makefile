test:
	pytest tests -vv

tests: 
	test

coverage: 
	pytest --cov=rocketpy tests -vv

install: 
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	python setup.py install

verify-lint:
	black --check rocketpy
	black --check tests

lint:
	black rocketpy
	black tests
