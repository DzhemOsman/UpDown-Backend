import logging
import pandas as pd
from app.services.ingestion import ingest_all
import urllib.request
import io

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_diversified_200_tickers() -> list[str]:
    """
    Lädt die aktuellen Tickerlisten aus Wikipedia (S&P 500 und DAX) mit einem
    User-Agent und bereitet die Daten als StringIO für Pandas auf.
    """
    logger.info("Hole aktuelle Ticker-Listen aus dem Web (mit Browser-User-Agent)...")

    # Browser-Identität simulieren
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

    # 1. S&P 500 laden
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req_sp500 = urllib.request.Request(sp500_url, headers=headers)
    with urllib.request.urlopen(req_sp500) as response:
        html_text = response.read().decode('utf-8')
        sp500_table = pd.read_html(io.StringIO(html_text), flavor="html5lib")[0]
    us_tickers = sp500_table["Symbol"].tolist()

    # 2. DAX laden
    dax_url = "https://en.wikipedia.org/wiki/DAX"
    req_dax = urllib.request.Request(dax_url, headers=headers)
    with urllib.request.urlopen(req_dax) as response:
        html_text = response.read().decode('utf-8')
        dax_table = pd.read_html(io.StringIO(html_text), flavor="html5lib")[4]
    de_tickers = dax_table["Ticker"].tolist()

    # Bereinigung (yfinance Suffix für deutsche Aktien)
    de_tickers = [t.replace("GY", "DE") if "GY" in t else f"{t}.DE" for t in de_tickers]

    # 3. Kombinieren auf 200 Assets
    final_tickers = us_tickers[:160] + de_tickers[:40]
    final_tickers = [t.replace(".", "-") for t in final_tickers]

    return final_tickers


if __name__ == "__main__":
    try:
        tickers_200 = get_diversified_200_tickers()
        logger.info(f"🚀 Starte Massen-Download für {len(tickers_200)} diversifizierte Aktien...")

        # Deine bestehende Ingestion-Routine starten
        ingest_all(tickers_200)

        logger.info("🎉 Alle 200 Aktien erfolgreich in der InfluxDB gespeichert!")
    except Exception as e:
        logger.error(f"Fehler beim Bulk-Ingest: {e}")