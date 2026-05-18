from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.v1.router import api_router

app = FastAPI(title="UpDown Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Health-Check
@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}

# All V1 routes 
app.include_router(api_router)