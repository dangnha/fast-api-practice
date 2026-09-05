# AI Inference Platform

A production-shaped FastAPI service that classifies handwritten digits with a
small convolutional neural network, while exercising the full range of FastAPI
engineering patterns: dependency injection, SQLAlchemy persistence, JWT
authentication with role-based access control, background processing, Redis
Pub/Sub events, and WebSocket real-time progress.

> **Purpose.** This project is an implementation reference. Each FastAPI feature
> maps to a concrete, idiomatic pattern rather than a toy abstraction. The model
> is deliberately small (two convolutional layers, trained on MNIST) so the
> entire stack runs on a laptop CPU — yet the HTTP and service boundaries are
> designed to survive a swap to a larger or GPU-hosted model without touching the
> API layer.

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Getting started](#getting-started)
  - [Local development](#local-development)
  - [Docker](#docker)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [End-to-end usage](#end-to-end-usage)
- [Testing](#testing)
- [Production hardening](#production-hardening)
- [License](#license)

---

## Features

| Area | Implementation |
| --- | --- |
| Routing & validation | FastAPI `APIRouter`, Pydantic v2 models, typed response contracts |
| Dependency injection | `Depends` + `Annotated` aliases for database sessions and auth |
| Persistence | SQLAlchemy 2.0 ORM with session-per-request |
| Authentication | JWT (HS256) access tokens, Argon2 password hashing |
| Authorization | Role-based access control (`user` / `admin`) via dependency guards |
| File upload | Streaming, size-capped uploads with content-type allow-listing |
| Inference | Tiny CNN (`MnistCNN`) exposed behind a stable `MnistClassifier` interface |
| Background work | `BackgroundTasks` with `asyncio.to_thread` to keep the event loop free |
| Real-time events | Redis Pub/Sub fan-out to per-task WebSocket channels |
| Observability | Request ID propagation and process-time headers via HTTP middleware |
| Operations | Health/readiness endpoints, Dockerfile, Docker Compose, pytest suite |

---

## Architecture

```text
 Client ──REST + JWT──▶ FastAPI ────▶ PostgreSQL (task state)
                         │
                         ▼
                  Background job
                  MnistClassifier
                         │
                         ▼
                 Redis Pub/Sub ──▶ WebSocket ──▶ Client
```

1. The client registers or signs in and receives a JWT.
2. `POST /api/v1/tasks` validates and stores the upload, creates a `pending`
   task record, and returns the task ID.
3. A background job marks the task `running`, loads the image, runs the model,
   and writes the outcome as `succeeded` or `failed`.
4. Each state change is published to Redis; a per-task WebSocket subscribes to
   the channel and streams progress back to the client.

The database is the source of truth. If Redis is unavailable, inference still
completes and the result is persisted — the event layer degrades gracefully.

---

## Project structure

```text
fastapi-ai-platform/
├── app/
│   ├── __init__.py
│   ├── main.py            # application factory, lifespan, middleware, routers
│   ├── config.py          # pydantic-settings configuration
│   ├── database.py        # engine, session factory, declarative base
│   ├── dependencies.py    # DI aliases and auth guards
│   ├── models.py          # SQLAlchemy ORM models
│   ├── schemas.py         # Pydantic request/response models
│   ├── security.py        # password hashing and JWT helpers
│   ├── ml.py              # CNN definition and inference interface
│   ├── services.py        # background inference pipeline + Redis publishing
│   ├── websocket.py       # WebSocket task-progress endpoint
│   └── routers/
│       ├── __init__.py
│       ├── auth.py        # registration and token issuance
│       └── tasks.py       # task creation, listing, retrieval
├── scripts/
│   └── train.py           # trains and persists the CNN artifact
├── tests/
│   ├── test_health.py
│   └── test_model.py
├── models/                # generated model artifacts (git-ignored)
├── uploads/               # temporary upload staging (git-ignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Prerequisites

- Python 3.12+
- Redis 7 (for WebSocket events; optional for core inference)
- Docker and Docker Compose (for the containerized workflow)

---

## Getting started

### Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the CNN and produce the model artifact (downloads MNIST on first run):

```bash
python -m scripts.train
```

Start Redis:

```bash
docker run --rm -d -p 6379:6379 redis:7-alpine
```

Configure and run:

```bash
cp .env.example .env
uvicorn app.main:app --reload
```

The interactive API documentation is available at
<http://localhost:8000/docs>.

### Docker

```bash
python -m scripts.train    # generate the model artifact first
docker compose up --build
```

Compose provisions the API, PostgreSQL, and Redis together.

---

## Configuration

Configuration is managed by `pydantic-settings` and read from environment
variables or a `.env` file.

| Variable | Default | Description |
| --- | --- | --- |
| `APP_NAME` | `AI Inference Platform` | Application title |
| `ENVIRONMENT` | `development` | Runtime environment (`development` / `production`) |
| `DATABASE_URL` | `sqlite:///./fastapi_ai.db` | SQLAlchemy connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `JWT_SECRET` | `change-me` | Secret used to sign JWTs |
| `ACCESS_TOKEN_MINUTES` | `30` | Access token lifetime in minutes |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins (JSON array) |
| `MODEL_PATH` | `models/mnist_cnn.pt` | Path to the trained model weights |

> `CORS_ORIGINS` is typed as `list[str]`, so its environment value must be a
> JSON array. In production, `JWT_SECRET` must be replaced; startup refuses to
> run otherwise.

---

## API reference

All endpoints are documented interactively at `/docs`.

### System

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (DB connectivity + model loaded) |

### Authentication

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/v1/auth/register` | Create a user account |
| `POST` | `/api/v1/auth/token` | Obtain a JWT (OAuth2 password flow) |

### Tasks

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `POST` | `/api/v1/tasks` | Bearer | Upload an image and enqueue inference |
| `GET` | `/api/v1/tasks` | Bearer | List the caller's tasks (paginated) |
| `GET` | `/api/v1/tasks/{task_id}` | Bearer | Retrieve a task (owner or admin) |

### WebSocket

| Path | Description |
| --- | --- |
| `/ws/tasks/{task_id}?token={token}` | Stream task status/progress events |

---

## End-to-end usage

```bash
# 1. Register a user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"researcher","password":"strong-pass-123"}'

# 2. Obtain an access token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=researcher&password=strong-pass-123" | jq -r .access_token)

# 3. Submit an image for inference
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@digit.png"

# 4. Poll for the result
curl http://localhost:8000/api/v1/tasks \
  -H "Authorization: Bearer $TOKEN"
```

To observe progress in real time, open a WebSocket connection to
`ws://localhost:8000/ws/tasks/{task_id}?token={token}` immediately after task
creation. A production frontend should connect as soon as the task ID is
returned, render the initial state, and reconnect with backoff on network loss.

---

## Testing

```bash
pytest
```

The suite covers the health endpoint and the model interface. A full
production test matrix would additionally cover database fixtures, dependency
overrides, ownership and RBAC enforcement, upload size limits, task state
transitions, Redis failure modes, and WebSocket behavior.

---

## Production hardening

The current implementation uses `BackgroundTasks`, local disk storage, and
`Base.metadata.create_all` as deliberate learning simplifications. To move to
production, replace them behind the same interfaces:

| Component | Learning choice | Production direction |
| --- | --- | --- |
| Background job | `BackgroundTasks` | Durable task queue with dedicated workers |
| File storage | Local disk | Object storage with lifecycle policies and malware scanning |
| Schema | `create_all` | Alembic migrations in CI/CD |
| Model | In-process CNN | Separate model server/GPU worker, batching, versioning |
| Events | Redis Pub/Sub | Pub/Sub or durable streams per requirements |
| Auth | Access JWT | Rotation, refresh/revocation, audit, external IdP |
| Observability | Timing headers | Structured logs, metrics, tracing, alerts, SLOs |
| Deployment | Docker Compose | Reverse proxy, load balancer, orchestrator |

The HTTP contract and the `MnistClassifier.predict` boundary are designed so
that these upgrades are replacements, not rewrites.

---

## License

This project is provided as an educational reference implementation.
