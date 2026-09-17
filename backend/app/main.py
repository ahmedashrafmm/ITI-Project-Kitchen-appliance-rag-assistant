from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes.query import router as query_router
from app.services.retrieval import load_vector_store
from app.services import vision
from app.utils.logging_config import setup_logging

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading vector store and embedding model...")
    load_vector_store()
    logger.info("Loading YOLO model...")
    vision.load_yolo_model()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Kitchen Appliance RAG Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
