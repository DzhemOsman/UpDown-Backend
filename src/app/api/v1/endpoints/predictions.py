from fastapi import APIRouter, HTTPException, status

from app.schemas.api.predict_request import PredictRequest
from app.services.prediction import predict_ticker_trend

router = APIRouter()


@router.post(
    "/predict",
    summary="Trifft eine Trend-Vorhersage für eine Aktie an einem bestimmten Datum",
)
def get_prediction(request: PredictRequest):
    """
    Nimmt einen Ticker und ein historisches Datum entgegen, berechnet die
    technischen Indikatoren sowie den globalen Marktkontext zur Laufzeit
    und liefert die LightGBM-Klassifikation zurück.
    """
    try:
        result = predict_ticker_trend(ticker=request.ticker, date_str=request.date)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unerwarteter Fehler bei der Modell-Inferenz: {str(exc)}",
        )
