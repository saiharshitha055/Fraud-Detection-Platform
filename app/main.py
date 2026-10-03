from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import engine, Base
from app.models import database_models
from app.api import predictions

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Real-Time Fraud Detection & Risk Analytics API",
    description="Production-grade fintech risk scoring and analyst investigation backend.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router)

@app.get("/health", tags=["System Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "fraud-detection-backend",
        "database": "connected"
    }