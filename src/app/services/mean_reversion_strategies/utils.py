import logging
import time
from datetime import datetime

from app.services.mean_reversion_strategies.mean_reversion_defaults import DEFAULT_INITIAL_CAPITAL
from app.services.mean_reversion_strategies.mean_reversion_strategy import optimize_grid_search
from app.services.mean_reversion_strategies.money_management_reversion import (
    optimize_money_management_with_grid_search,
    optimize_bayesian
)

logger = logging.getLogger(__name__)


def compare_grid_search_and_bayesian(tickers: list[str]):
    # Gleicher Zeitraum für fairen Vergleich
    start_date = datetime(2014, 1, 1)
    end_date = datetime(2024, 12, 31)

    # Parameter-Raum für die Grid-Search definieren
    # (3 * 3 * 2 * 2 * 2 * 2 = 144 Kombinationen)
    drop_opts = [3, 5, 7]
    hold_opts = [2, 4, 6]
    tp_opts = [1.5, 2.5]
    sl_opts = [2.0, 5.0]
    max_pos_opts = [1, 3]
    alloc_opts = [10.0, 20.0]

    logger.info("=== STARTE LAUFZEIT-VERGLEICH ===")

    # ---------------------------------------------------------
    # 1. GRID SEARCH MESSUNG
    # ---------------------------------------------------------
    logger.info("\n1. Starte Grid Search (144 Kombinationen)...")
    start_time_grid = time.time()

    result_grid = optimize_money_management_with_grid_search(
        tickers=tickers,
        drop_options=drop_opts,
        hold_options=hold_opts,
        take_profit_options=tp_opts,
        stop_loss_options=sl_opts,
        max_positions_options=max_pos_opts,
        allocation_options=alloc_opts,
        initial_capital=10000,
        start=start_date,
        end=end_date
    )

    end_time_grid = time.time()
    duration_grid = end_time_grid - start_time_grid

    # ---------------------------------------------------------
    # 2. BAYESIAN OPTIMIZATION MESSUNG
    # ---------------------------------------------------------
    trials = 50
    logger.info(f"\n2. Starte Bayesian Optimization ({trials} Trials)...")
    start_time_bayes = time.time()

    result_bayes = optimize_bayesian(
        tickers=tickers,
        n_trials=trials,
        initial_capital=10000,
        start=start_date,
        end=end_date
    )

    end_time_bayes = time.time()
    duration_bayes = end_time_bayes - start_time_bayes

    # ---------------------------------------------------------
    # 3. AUSWERTUNG & VERGLEICH
    # ---------------------------------------------------------
    logger.info("\n========================================")
    logger.info("           VERGLEICHS-ERGEBNIS          ")
    logger.info("========================================")

    # Grid Search Ergebnisse
    logger.info("\n--- GRID SEARCH ---")
    logger.info(f"Dauer:         {duration_grid:.2f} Sekunden")
    if result_grid:
        logger.info(f"Bester ROI:    {result_grid['roi_pct']}%")
        logger.info(
            f"Beste Params:  Drop: {result_grid['best_drop_threshold']}%, Hold: {result_grid['best_hold_days']}T, TP: {result_grid['best_take_profit_pct']}%, SL: {result_grid['best_stop_loss_pct']}%, MaxPos: {result_grid['best_max_positions']}, Alloc: {result_grid['best_allocation_pct']}%")
    else:
        logger.info("Kein Ergebnis gefunden.")

    # Bayesian Ergebnisse
    logger.info("\n--- BAYESIAN OPTIMIZATION ---")
    logger.info(f"Dauer:         {duration_bayes:.2f} Sekunden")
    if result_bayes:
        logger.info(f"Bester ROI:    {result_bayes['roi_pct']}%")
        logger.info(
            f"Beste Params:  Drop: {result_bayes['best_drop_threshold']}%, Hold: {result_bayes['best_hold_days']}T, TP: {result_bayes['best_take_profit_pct']}%, SL: {result_bayes['best_stop_loss_pct']}%, MaxPos: {result_bayes['best_max_positions']}, Alloc: {result_bayes['best_allocation_pct']}%")
    else:
        logger.info("Kein Ergebnis gefunden.")

    # Fazit
    logger.info("\n--- FAZIT ---")
    if duration_bayes > 0 and duration_grid > 0:
        if duration_grid > duration_bayes:
            speedup = duration_grid / duration_bayes
            logger.info(f"=> Bayesian Optimization war {speedup:.1f}x SCHNELLER!")
        else:
            speedup = duration_bayes / duration_grid
            logger.info(
                f"=> Grid Search war {speedup:.1f}x SCHNELLER! (Wahrscheinlich war der Parameter-Raum für Grid zu klein gewählt)")


def test_money_management_with_bayesian(tickers: list[str], is_kadane: bool = False, is_trend: bool = False):
    result = optimize_bayesian(
        tickers=tickers,
        n_trials=100,  # 100 Versuche sind oft schon extrem gut und schnell
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        start=datetime(2014, 1, 1),
        end=datetime(2024, 12, 31),
        is_kadane=is_kadane,
        is_trend=is_trend
    )

    if result is None:
        logger.warning("Keine gültige Konfiguration gefunden.")
        return

    logger.info("\n=== BESTE KONFIGURATION ===")
    logger.info(f"Drop Threshold:   {result['best_drop_threshold']}%")
    logger.info(f"Hold Days:        {result['best_hold_days']} Tage")
    logger.info(f"Take Profit:      {result['best_take_profit_pct']}%")
    logger.info(f"Stop Loss:        {result['best_stop_loss_pct']}%")
    logger.info(f"Max Positions:    {result['best_max_positions']}")
    logger.info(f"Capital Alloc:    {result['best_allocation_pct']}% pro Trade")
    logger.info(f"Kadane:           {is_kadane}")
    logger.info(f"SMA:              {is_trend}")

    logger.info("\n=== PERFORMANCE ===")
    logger.info(f"ROI:              {result['roi_pct']}%")
    logger.info(f"Total Profit:     {result['total_profit']}")
    logger.info(f"Win Rate:         {result['win_rate']}%")
    logger.info(f"Total Trades:     {result['total_number_of_trades']}")
    logger.info(f"Search Type:      grid search")

    logger.info("\n=== ERSTE 10 TRADES ===")
    for trade in result["trades"][:10]:
        logger.info(trade)

    logger.info("\n=== ERSTE 10 EQUITY-PUNKTE ===")
    for point in result["equity_curve_data"][:10]:
        logger.info(point)


def test_money_management_with_grid_search(tickers: list[str], is_kadane: bool = False, is_trend: bool = False):
    result = optimize_money_management_with_grid_search(
        tickers=tickers,
        drop_options=[3, 4, 5],
        hold_options=[2, 3, 5],
        take_profit_options=[2.0, 3.0],
        stop_loss_options=[2.0, 5.0, 10.0],
        max_positions_options=[1, 2, ],
        allocation_options=[10.0, 20.0, ],
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        start=datetime(2014, 1, 1),
        end=datetime(2024, 12, 31),
        is_kadane=is_kadane,
        is_trend=is_trend
    )

    if result is None:
        logger.warning("Keine gültige Konfiguration gefunden.")
        return

    logger.info("\n=== BESTE KONFIGURATION ===")
    logger.info(f"Drop Threshold:   {result['best_drop_threshold']}%")
    logger.info(f"Hold Days:        {result['best_hold_days']} Tage")
    logger.info(f"Take Profit:      {result['best_take_profit_pct']}%")
    logger.info(f"Stop Loss:        {result['best_stop_loss_pct']}%")
    logger.info(f"Max Positions:    {result['best_max_positions']}")
    logger.info(f"Capital Alloc:    {result['best_allocation_pct']}% pro Trade")
    logger.info(f"Kadane:           {is_kadane}")
    logger.info(f"SMA:              {is_trend}")

    logger.info("\n=== PERFORMANCE ===")
    logger.info(f"ROI:              {result['roi_pct']}%")
    logger.info(f"Total Profit:     {result['total_profit']}")
    logger.info(f"Win Rate:         {result['win_rate']}%")
    logger.info(f"Total Trades:     {result['total_number_of_trades']}")
    logger.info(f"Search Type:      grid search")

    logger.info("\n=== ERSTE 5 TRADES ===")
    for trade in result["trades"][:5]:
        logger.info(trade)

    logger.info("\n=== ERSTE 5 EQUITY-PUNKTE ===")
    for point in result["equity_curve_data"][:5]:
        logger.info(point)


def test_old_mean_reversion():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # tickers = ["AAPL", "MSFT", "DBK", "TSLA", "NVDA", "CRM"]
    tickers = ["MSFT", "AAPL"]

    result = optimize_grid_search(
        tickers=tickers,
        drop_options=[3, 4, 5, 6, 7],
        hold_options=[2, 3, 4, 5, 6],
        take_profit_options=[1.5, 2.0, 2.5, 3.0],
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        start=datetime(2014, 1, 1),
        end=datetime(2024, 12, 31)
    )

    if result is None:
        logger.warning("Keine gültige Konfiguration gefunden.")
        return

    logger.info("\n=== BESTE KONFIGURATION ===")
    logger.info(f"Drop Threshold:   {result['best_drop_threshold']}%")
    logger.info(f"Hold Days:        {result['best_hold_days']}")
    logger.info(f"Take Profit:      {result['best_take_profit_pct']}%")
    logger.info(f"Stop Loss:        keins")

    logger.info("\n=== PERFORMANCE ===")
    logger.info(f"ROI:              {result['roi_pct']}%")
    logger.info(f"Total Profit:     {result['total_profit']}")
    logger.info(f"Win Rate:         {result['win_rate']}%")
    logger.info(f"Total Trades:     {result['total_number_of_trades']}")
    logger.info(f"Search Type:      grid search")

    logger.info("\n=== ERSTE 5 TRADES ===")
    for trade in result["trades"][:5]:
        logger.info(trade)

    logger.info("\n=== ERSTE 10 EQUITY-PUNKTE ===")
    for point in result["equity_curve_data"][:5]:
        logger.info(point)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # tickers = ["AAPL", "MSFT", "DBK", "TSLA", "NVDA", "CRM"]
    tickers = ["MSFT", "AAPL"]

    #test_money_management_with_bayesian(tickers)
    #test_money_management_with_bayesian(tickers, is_kadane=True)

    test_money_management_with_grid_search(tickers)
    test_money_management_with_grid_search(tickers, is_kadane=True)
    test_money_management_with_grid_search(tickers, is_trend=True)
    test_money_management_with_grid_search(tickers, is_kadane=True, is_trend=True)
    # compare_grid_search_and_bayesian(tickers)
