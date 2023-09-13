.PHONY: test
test:
	@pytest

.PHONY: fmt
fmt:
	@pre-commit run --all-files

.PHONY: build
build:
	@docker build -t fvt-ml .
