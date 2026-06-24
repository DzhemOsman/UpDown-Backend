from datetime import datetime

from app.services import market_data

if __name__ == "__main__":
    result = market_data.fetch_ticker_data(
        ticker="DBK",
        start_date=datetime(2000, 1, 1),
        end_date=datetime(2020, 6, 30),
    )
    print(result.to_string(index=False))
