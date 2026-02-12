from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.storage.db import init_db


app = FastAPI(title="Transport Policing System", version="0.1.0")
app.include_router(router)
app.mount("/static", StaticFiles(directory="dashboard"), name="static")


@app.on_event("startup")
def startup() -> None:
    init_db()

