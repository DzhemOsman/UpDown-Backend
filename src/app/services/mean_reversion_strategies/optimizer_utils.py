import pandas as pd

from app.schemas.internal.best_parameter_combination_dict import (
    Combo,
    ParameterCombinationDict,
)
from app.schemas.internal.trade_result_dict import TradeResultDict
from app.services.mean_reversion_strategies.mean_reversion_defaults import (
    DEFAULT_FEE_RATE,
    DEFAULT_LOOKBACK_DAYS,
)


def trades_to_metrics(
    trades: list[TradeResultDict], initial_capital: int
) -> tuple[float, float, float, int]:
    if not trades:
        return 0.0, 0.0, 0.0, 0
    df = pd.DataFrame(trades)
    profit = float(df["profit_abs"].sum())
    roi = (profit / initial_capital) * 100
    win_rate = float((df["profit_abs"] > 0).mean() * 100)
    return roi, profit, win_rate, len(trades)


def combo_to_params(combo: Combo) -> ParameterCombinationDict:
    return ParameterCombinationDict(
        drop_threshold=combo.drop_threshold,
        lookback_days=DEFAULT_LOOKBACK_DAYS,
        hold_days=combo.hold_days,
        take_profit_pct=combo.take_profit_pct,
        stop_loss_pct=combo.stop_loss_pct,
        max_positions=combo.max_positions,
        allocation_pct=combo.allocation_pct,
        fee_pct=DEFAULT_FEE_RATE,
    )
