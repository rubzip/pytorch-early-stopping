.PHONY: example test clean

# runs the example
example:
	python examples/MNIST_example.py

# runs the tests
test:
	pytest tests/test_early_stopping.py

# cleans temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +
