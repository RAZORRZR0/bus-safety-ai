from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Bus(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    bus_id: str = Field(index=True, unique=True)
    route_id: str = Field(index=True)
    capacity: int
    model: str
    year: int
    last_maintenance: datetime = Field(default_factory=utc_now)


class Route(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    route_id: str = Field(index=True, unique=True)
    name: str
    start_location: str
    end_location: str
    distance_km: float


class BusCreate(BaseModel):
    bus_id: str
    route_id: str
    capacity: int
    model: str
    year: int


class RouteCreate(BaseModel):
    route_id: str
    name: str
    start_location: str
    end_location: str
    distance_km: float


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    role: str = Field(default="operator")
    password_hash: str
    twofa_secret: str
    twofa_verified: bool = False


class ViolationEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    bus_id: str = Field(index=True)
    route_id: str = Field(index=True)
    timeslot: str = Field(index=True)
    seat_misuse: bool = False
    violence: bool = False
    people_count: int = 0
    aggression_score: float = 0.0
    severity: str
    created_at: datetime = Field(default_factory=utc_now, index=True)
    evidence_hash: Optional[str] = None
    summary: str


class Alert(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    violation_event_id: int = Field(index=True)
    bus_id: str = Field(index=True)
    route_id: str = Field(index=True)
    severity: str = Field(index=True)
    status: str = Field(default="open", index=True)
    created_at: datetime = Field(default_factory=utc_now, index=True)
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


class TrustScore(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    scope_type: str = Field(index=True)  # bus | route | timeslot
    scope_id: str = Field(index=True)
    score: float = 100.0
    updated_at: datetime = Field(default_factory=utc_now, index=True)


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "operator"


class LoginRequest(BaseModel):
    username: str
    password: str


class Verify2FARequest(BaseModel):
    token: str
    code: str


class EventIngestRequest(BaseModel):
    bus_id: str
    route_id: str
    timeslot: str
    reserved_seats_total: int
    reserved_seats_occupied: int
    special_category_present: bool
    aggression_score: float
    model_confidence: float = 0.9


class Feedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    event_reference_id: str = Field(index=True)
    feedback_type: str = Field(index=True)  # e.g., "violence_false_positive", "seat_false_positive"
    is_correct: bool = Field(index=True)
    comments: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now, index=True)
    created_by: str = Field(index=True)


class FeedbackSubmitRequest(BaseModel):
    event_reference_id: str
    feedback_type: str
    is_correct: bool
    comments: Optional[str] = None

