python -m flake8 .
python -m isort . --skip ./outputs
python -m black . --exclude ./outputs/**
