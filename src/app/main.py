from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services import market_data
from app.api.v1.router import api_router
from app.config import get_settings
from app.core.influx import get_client

app = FastAPI(title="UpDown Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}


app.include_router(api_router, prefix="/v1")

if __name__ == "__main__":
    client = get_client()
    result = market_data.fetch_ticker_data(
        ticker = "DBK",
        start_date = datetime(2000, 1, 1),
        end_date =datetime(2020, 6, 30)
    )
    print(result.to_string(index=False))