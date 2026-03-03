"""
Transaction Cost Model — V3 Layer 13: Realistic spread+slippage deduction.

Applies per-symbol transaction costs to backtest results for honest P&L:
  - Spread cost: bid-ask spread per trade
  - Slippage: market impact from execution delay
  - Commission: broker fee (if applicable)
"""

import logging

log = logging.getLogger("transaction_cost")

# Default costs per symbol (in price units, e.g., pips)
SYMBOL_COSTS = {
    "EURUSD": {"spread_pips": 1.5, "slippage_pips": 0.5, "commission_pips": 0.0},
    "GBPUSD": {"spread_pips": 2.0, "slippage_pips": 0.5, "commission_pips": 0.0},
    "USDJPY": {"spread_pips": 1.5, "slippage_pips": 0.5, "commission_pips": 0.0},
    "AUDUSD": {"spread_pips": 2.0, "slippage_pips": 0.5, "commission_pips": 0.0},
    "USDCAD": {"spread_pips": 2.5, "slippage_pips": 0.5, "commission_pips": 0.0},
    "NZDUSD": {"spread_pips": 3.0, "slippage_pips": 0.5, "commission_pips": 0.0},
}

PIP_VALUES = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01,
    "AUDUSD": 0.0001, "USDCAD": 0.0001, "NZDUSD": 0.0001,
}


def get_cost_per_trade(symbol, lot_size=1.0):
    """
    Get total cost per trade in price units.
    
    Returns cost as fraction of price movement needed to break even.
    """
    costs = SYMBOL_COSTS.get(symbol, SYMBOL_COSTS["EURUSD"])
    pip_value = PIP_VALUES.get(symbol, 0.0001)
    
    total_pips = costs["spread_pips"] + costs["slippage_pips"] + costs["commission_pips"]
    return total_pips * pip_value


def apply_costs_to_pnl(results, symbol):
    """
    Apply transaction costs to backtest results.
    
    Modifies results in-place, adding 'pnl_after_costs' field.
    Returns adjusted P&L summary.
    """
    cost = get_cost_per_trade(symbol)
    
    total_pnl = 0
    adjusted_pnl = 0
    n_trades = len(results)
    
    for r in results:
        raw = 1 if r.get("correct", 0) else -1
        adjusted = raw - cost
        r["pnl_raw"] = raw
        r["pnl_adjusted"] = adjusted
        total_pnl += raw
        adjusted_pnl += adjusted
    
    cost_impact = total_pnl - adjusted_pnl
    
    log.info("Transaction costs: %d trades, cost=%.5f/trade, "
             "raw_pnl=%.1f, adjusted=%.1f, cost_impact=%.1f",
             n_trades, cost, total_pnl, adjusted_pnl, cost_impact)
    
    return {
        "n_trades": n_trades,
        "raw_pnl": total_pnl,
        "adjusted_pnl": adjusted_pnl,
        "total_costs": cost_impact,
        "cost_per_trade": cost,
    }
