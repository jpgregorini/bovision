"""BoviSight API — Main entry point."""
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.database import Base, engine
from api.routers import weighings, animals, reports

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="BoviSight API",
    description="API para pesagem automatizada de bovinos por visao computacional.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(weighings.router, prefix="/api")
app.include_router(animals.router, prefix="/api")
app.include_router(reports.router, prefix="/api")


@app.get("/")
def root():
    return {"name": "BoviSight API", "version": "0.1.0", "status": "running"}
