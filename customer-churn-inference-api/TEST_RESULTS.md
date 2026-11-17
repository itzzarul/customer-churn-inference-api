# Test Results

Local verification completed in the build environment:

```text
python -m compileall -q app tests train_model.py
# Python syntax compilation passed

python -m pytest -q
# 6 passed
```

The test suite validates the root endpoint, health endpoint, prediction response, probability range, experimental-model warning, invalid plan category, invalid numeric input, and rejection of extra fields.

Docker image build/run was not executed in this environment; follow the Docker commands in README.md before production deployment.
