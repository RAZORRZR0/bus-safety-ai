from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.storage.db import engine, init_db
from app.storage.demo import seed_demo


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if settings.demo_mode:
        seed_demo(engine)
    yield


app = FastAPI(title="Transport Policing System", version="0.2.0", lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory="dashboard"), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response
