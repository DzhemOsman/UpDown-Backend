import logging
from datetime import datetime

from app.services.mean_reversion_strategies.mean_reversion_defaults import (
    DEFAULT_INITIAL_CAPITAL,
)
from app.services.mean_reversion_strategies.money_management_optimizer import (
    optimize_money_management_with_grid_search,
)
from app.services.mean_reversion_strategies.optimizer import optimize_grid_search

logger = logging.getLogger(__name__)


def test_money_management_with_grid_search(
    tickers: list[str], is_kadane: bool = False, is_trend: bool = False
):
    result = optimize_money_management_with_grid_search(
        tickers=tickers,
        drop_options=[3, 4, 5],
        hold_options=[2, 3, 5],
        take_profit_options=[2.0, 3.0],
        stop_loss_options=[2.0, 5.0, 10.0],
        max_positions_options=[
            1,
            2,
        ],
        allocation_options=[
            10.0,
            20.0,
        ],
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        start=datetime(2014, 1, 1),
        end=datetime(2024, 12, 31),
        is_kadane=is_kadane,
        is_trend=is_trend,
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
    logger.info("Search Type:      grid search")

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
        end=datetime(2024, 12, 31),
    )

    if result is None:
        logger.warning("Keine gültige Konfiguration gefunden.")
        return

    logger.info("\n=== BESTE KONFIGURATION ===")
    logger.info(f"Drop Threshold:   {result['best_drop_threshold']}%")
    logger.info(f"Hold Days:        {result['best_hold_days']}")
    logger.info(f"Take Profit:      {result['best_take_profit_pct']}%")
    logger.info("Stop Loss:        keins")

    logger.info("\n=== PERFORMANCE ===")
    logger.info(f"ROI:              {result['roi_pct']}%")
    logger.info(f"Total Profit:     {result['total_profit']}")
    logger.info(f"Win Rate:         {result['win_rate']}%")
    logger.info(f"Total Trades:     {result['total_number_of_trades']}")
    logger.info("Search Type:      grid search")

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

    # test_money_management_with_grid_search(tickers)
    # test_money_management_with_grid_search(tickers, is_kadane=True)
    # test_money_management_with_grid_search(tickers, is_trend=True)
    # test_money_management_with_grid_search(tickers, is_kadane=True, is_trend=True)
