# TCE Development Makefile
.PHONY: install dev build test clean backend frontend all

# Installation
install: install-backend install-frontend

install-backend:
	cd backend && pip install -e ".[dev]" --break-system-packages

install-frontend:
	cd frontend && npm install

# Development
dev: dev-backend dev-frontend

dev-backend:
	cd backend && uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

# Build
build: build-frontend
	@echo "Build complete"

build-frontend:
	cd frontend && npm run build

# Testing
test: test-backend test-frontend

test-backend:
	cd backend && pytest -v

test-frontend:
	cd frontend && npm run test

# Linting
lint: lint-backend lint-frontend

lint-backend:
	cd backend && ruff check . && mypy app

lint-frontend:
	cd frontend && npm run lint

# Database
db-init:
	cd backend && python -c "from app.db.database import init_db; init_db()"

db-reset:
	rm -f backend/tce.db
	$(MAKE) db-init

# Clean
clean:
	rm -rf frontend/dist
	rm -rf frontend/node_modules
	rm -rf backend/__pycache__
	rm -rf backend/**/__pycache__
	rm -rf backend/*.egg-info
	rm -f backend/tce.db

# All-in-one development (requires tmux or separate terminals)
all:
	@echo "Run 'make dev-backend' and 'make dev-frontend' in separate terminals"
	@echo "Or use: make dev-backend & make dev-frontend"
