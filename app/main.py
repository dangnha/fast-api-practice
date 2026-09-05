import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app import websocket
from app.config import settings
from app.database import Base, SessionLocal, engine
from app.ml import MnistClassifier
from app.routers import auth, tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.jwt_secret == "change-me" and settings.environment == "production":
        raise RuntimeError("JWT_SECRET must be configured in production")

    Base.metadata.create_all(bind=engine)  # Replace with migrations in production
    app.state.model = MnistClassifier(settings.model_path)
    yield
    app.state.model = None
    engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    request.state.request_id = request_id
    started_at = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{time.perf_counter() - started_at:.6f}"
    return response


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["system"])
def ready() -> dict[str, object]:
    with SessionLocal() as db:
        db.execute(text("SELECT 1"))
    return {"status": "ready", "model_loaded": app.state.model is not None}


app.include_router(auth.router, prefix="/api/v1")
app.include_router(tasks.router, prefix="/api/v1")
app.include_router(websocket.router)
