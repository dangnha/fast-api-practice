from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from sqlalchemy import select

from app.config import settings
from app.dependencies import CurrentUser, DbSession
from app.models import InferenceTask
from app.schemas import TaskRead
from app.services import run_inference

router = APIRouter(prefix="/tasks", tags=["tasks"])

ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
MAX_BYTES = 5 * 1024 * 1024


async def save_upload(file: UploadFile, task_id: str) -> Path:
    suffix = ALLOWED_TYPES.get(file.content_type or "")
    if suffix is None:
        raise HTTPException(status_code=415, detail="Unsupported media type")

    path = Path("uploads") / f"{task_id}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with path.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_BYTES:
                path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File is too large")
            output.write(chunk)
    return path


@router.post("", response_model=TaskRead, status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    user: CurrentUser,
    db: DbSession,
) -> InferenceTask:
    task_id = str(uuid4())
    path = await save_upload(file, task_id)

    task = InferenceTask(
        id=task_id,
        owner_id=user.id,
        filename=file.filename or path.name,
        status="pending",
        progress=0,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    background_tasks.add_task(
        run_inference,
        task.id,
        str(path),
        request.app.state.model,
        settings.redis_url,
    )
    return task


@router.get("", response_model=list[TaskRead])
def list_tasks(
    user: CurrentUser,
    db: DbSession,
    offset: int = 0,
    limit: int = 20,
) -> list[InferenceTask]:
    limit = min(max(limit, 1), 100)
    statement = (
        select(InferenceTask)
        .where(InferenceTask.owner_id == user.id)
        .order_by(InferenceTask.created_at.desc())
        .offset(max(offset, 0))
        .limit(limit)
    )
    return list(db.scalars(statement))


@router.get("/{task_id}", response_model=TaskRead)
def get_task(task_id: str, user: CurrentUser, db: DbSession) -> InferenceTask:
    task = db.get(InferenceTask, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.owner_id != user.id and user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    return task
