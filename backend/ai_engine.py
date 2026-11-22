
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# Load environment variables from .env when available.
load_dotenv()

# Application settings
APP_ENV = os.getenv("CHATBOT_ENV", "development").strip().lower()
IS_DEVELOPMENT = APP_ENV in {"development", "dev", "local"}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("chatbot_ai")


# Only allow explicitly configured frontend addresses.
allowed_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CHATBOT_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

# Local development addresses.
if not allowed_origins and IS_DEVELOPMENT:
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]


app = FastAPI(
    title="ChatBot AI",
    description=(
        "The backend API for ChatBot AI, created by Neliseka Mdunge."
    ),
    version="0.1.0",
    docs_url="/docs" if IS_DEVELOPMENT else None,
    redoc_url=None,
    openapi_url="/openapi.json" if IS_DEVELOPMENT else None,
)


# Configure browser access for the frontend.
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=600,
    )


@app.get("/", tags=["General"])
async def home():
    return {
        "name": "ChatBot AI",
        "creator": "Neliseka Mdunge",
        "status": "online",
        "message": "Welcome to the ChatBot AI backend.",
        "documentation": "/docs" if IS_DEVELOPMENT else None,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "ChatBot AI API",
        "version": "0.1.0",
    }


@app.get("/api/v1/status", tags=["General"])
async def api_status():
    return {
        "api": "operational",
        "ai_engine": "not_configured",
        "message": (
            "The backend is running. "
            "The AI engine will be connected in a later development step."
        ),
    }


@app.exception_handler(Exception)
async def unexpected_error_handler(request, exc):
    logger.exception("Unexpected server error")

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_server_error",
                "message": (
                    "Something went wrong. "
                    "Please try again later."
                ),
            }
        },
    )


@app.on_event("startup")
async def startup_event():
    logger.info("ChatBot AI API is starting.")
    logger.info("Environment: %s", APP_ENV)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("ChatBot AI API is shutting down.")
