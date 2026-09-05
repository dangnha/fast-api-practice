import json

import jwt
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from redis.asyncio import Redis
from sqlalchemy import select

from app.config import settings
from app.database import SessionLocal
from app.models import InferenceTask, User
from app.security import decode_access_token

router = APIRouter()


@router.websocket("/ws/tasks/{task_id}")
async def task_events(websocket: WebSocket, task_id: str, token: str):
    try:
        username = decode_access_token(token)
        with SessionLocal() as db:
            user = db.scalar(select(User).where(User.username == username))
            task = db.get(InferenceTask, task_id)
            if user is None or task is None:
                raise ValueError("Unknown user or task")
            if task.owner_id != user.id and user.role != "admin":
                raise ValueError("Access denied")
            initial_event = {
                "task_id": task.id,
                "status": task.status,
                "progress": task.progress,
                "label": task.label,
                "confidence": task.confidence,
                "error": task.error,
            }
    except (jwt.InvalidTokenError, ValueError):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    pubsub = redis.pubsub()
    channel = f"task:{task_id}:events"
    await pubsub.subscribe(channel)

    try:
        await websocket.send_json(initial_event)
        if initial_event["status"] in {"succeeded", "failed"}:
            return

        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            event = json.loads(message["data"])
            await websocket.send_json(event)
            if event.get("status") in {"succeeded", "failed"}:
                break
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
        await redis.aclose()
