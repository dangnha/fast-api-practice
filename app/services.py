import asyncio
import json
from pathlib import Path

from redis.asyncio import Redis

from app.database import SessionLocal
from app.ml import MnistClassifier
from app.models import InferenceTask


async def publish_event(redis: Redis, task: InferenceTask) -> None:
    event = {
        "task_id": task.id,
        "status": task.status,
        "progress": task.progress,
        "label": task.label,
        "confidence": task.confidence,
        "error": task.error,
    }
    try:
        await redis.publish(f"task:{task.id}:events", json.dumps(event))
    except Exception:
        # The database remains the source of truth if Redis is unavailable.
        pass


async def run_inference(
    task_id: str,
    file_path: str,
    model: MnistClassifier,
    redis_url: str,
) -> None:
    db = SessionLocal()
    redis = Redis.from_url(redis_url, decode_responses=True)
    path = Path(file_path)

    try:
        task = db.get(InferenceTask, task_id)
        if task is None:
            return

        task.status = "running"
        for progress in (10, 35, 70):
            task.progress = progress
            db.commit()
            await publish_event(redis, task)
            await asyncio.sleep(0.25)

        content = await asyncio.to_thread(path.read_bytes)
        prediction = await asyncio.to_thread(model.predict, content)

        task.status = "succeeded"
        task.progress = 100
        task.label = prediction.label
        task.confidence = prediction.confidence
        db.commit()
        await publish_event(redis, task)
    except Exception as exc:
        db.rollback()
        task = db.get(InferenceTask, task_id)
        if task is not None:
            task.status = "failed"
            task.error = str(exc)[:500]
            db.commit()
            await publish_event(redis, task)
    finally:
        path.unlink(missing_ok=True)
        await redis.aclose()
        db.close()
