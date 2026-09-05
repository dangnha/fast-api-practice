from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    USER = "user"
    ADMIN = "admin"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    password: str = Field(min_length=8, max_length=128)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    role: Role


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TaskRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    owner_id: int
    filename: str
    status: TaskStatus
    progress: int = Field(ge=0, le=100)
    label: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    error: str | None = None
    created_at: datetime
