.PHONY: dev-backend dev-frontend dev test-backend build-frontend

dev-backend:
	cd backend && uv run langgraph dev

dev-frontend:
	cd frontend && npm.cmd run dev

dev:
	$(MAKE) dev-backend & $(MAKE) dev-frontend

test-backend:
	cd backend && uv run pytest

build-frontend:
	cd frontend && npm.cmd run build
