from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from app.core.config import settings

engine = create_engine(settings.db_url, echo=False)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    _run_migrations()


def _has_column(table: str, column: str) -> bool:
    with engine.connect() as conn:
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return any(row[1] == column for row in rows)


def _run_migrations() -> None:
    # Lightweight migration for older DB files created before people_count was added.
    if _has_column("violationevent", "people_count"):
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE violationevent ADD COLUMN people_count INTEGER DEFAULT 0"))


def get_session():
    with Session(engine) as session:
        yield session
