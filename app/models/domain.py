from datetime import datetime, timezone
from typing import Annotated, Optional

from pydantic import BaseModel, StringConstraints
from pydantic import Field as PField
from sqlmodel import Field, SQLModel

# IDs end up in file names, HTML and onclick handlers, so keep them to a safe alphabet.
ID_PATTERN = r"^[A-Za-z0-9_.:\-]{1,64}$"
Ident = Annotated[str, StringConstraints(pattern=ID_PATTERN)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


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
    bus_id: Ident
    route_id: Ident
    capacity: int = PField(gt=0, le=500)
    model: ShortText
    year: int = PField(ge=1950, le=2100)


class RouteCreate(BaseModel):
    route_id: Ident
    name: ShortText
    start_location: ShortText
    end_location: ShortText
    distance_km: float = PField(ge=0, le=10000)


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
    username: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.\-]{3,32}$")]
    password: Annotated[str, StringConstraints(min_length=8, max_length=128)]
    role: str = "operator"


class LoginRequest(BaseModel):
    username: Annotated[str, StringConstraints(max_length=64)]
    password: Annotated[str, StringConstraints(max_length=128)]


class Verify2FARequest(BaseModel):
    token: Annotated[str, StringConstraints(max_length=2048)]
    code: Annotated[str, StringConstraints(pattern=r"^\d{6}$")]


class EventIngestRequest(BaseModel):
    bus_id: Ident
    route_id: Ident
    timeslot: Ident
    reserved_seats_total: int = PField(ge=0, le=100)
    reserved_seats_occupied: int = PField(ge=0, le=100)
    special_category_present: bool
    aggression_score: float = PField(ge=0.0, le=1.0)
    model_confidence: float = PField(default=0.9, ge=0.0, le=1.0)


class Feedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    event_reference_id: str = Field(index=True)
    feedback_type: str = Field(index=True)  # e.g., "violence_false_positive", "seat_false_positive"
    is_correct: bool = Field(index=True)
    comments: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now, index=True)
    created_by: str = Field(index=True)


class FeedbackSubmitRequest(BaseModel):
    event_reference_id: Annotated[str, StringConstraints(pattern=r"^(\d{1,12}|manual_[0-9a-f]{12}\.[A-Za-z0-9]{1,5})$")]
    feedback_type: Annotated[str, StringConstraints(pattern=r"^[a-z_]{1,64}$")]
    is_correct: bool
    comments: Optional[Annotated[str, StringConstraints(max_length=1000)]] = None

