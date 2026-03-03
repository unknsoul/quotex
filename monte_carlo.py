"""
Monte Carlo Simulation — V3 Layer 13: Trade sequence stress testing.

Runs 1000 random shuffles of trade outcomes to estimate:
  - Probability of ruin (>20% drawdown)
  - Expected max drawdown
  - Sharpe ratio distribution
  - Confidence interval for equity curve

Pass criterion: P(ruin > 20%) < 5%
"""

import numpy as np
import logging

log = logging.getLogger("monte_carlo")

N_SIMULATIONS = 1000
RUIN_THRESHOLD = 0.20  # 20% drawdown = ruin
MAX_RUIN_PROBABILITY = 0.05  # Must be < 5%


def run_monte_carlo(trade_results, n_sims=N_SIMULATIONS, initial_capital=10000):
    """
    Run Monte Carlo simulation on trade results.
    
    Args:
        trade_results: list of floats (P&L per trade, positive = win)
        n_sims: number of simulations
        initial_capital: starting capital
    
    Returns:
        dict with simulation results
    """
    n_trades = len(trade_results)
    if n_trades < 10:
        log.warning("Too few trades (%d) for Monte Carlo", n_trades)
        return {"error": "insufficient trades"}
    
    results = np.array(trade_results, dtype=np.float64)
    
    max_drawdowns = []
    final_equities = []
    sharpe_ratios = []
    ruin_count = 0
    
    for sim in range(n_sims):
        # Random shuffle of trade sequence
        shuffled = np.random.permutation(results)
        
        # Build equity curve
        equity = np.cumsum(shuffled) + initial_capital
        
        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = float(drawdown.max())
        max_drawdowns.append(max_dd)
        
        # Check ruin
        if max_dd >= RUIN_THRESHOLD:
            ruin_count += 1
        
        # Final equity
        final_equities.append(float(equity[-1]))
        
        # Sharpe ratio (annualized, assuming 288 trades/day for M5)
        daily_pnl = shuffled.reshape(-1, min(288, n_trades // 2 + 1)).sum(axis=1) \
            if n_trades > 576 else shuffled
        if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
            sharpe = float(np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252))
        else:
            sharpe = 0.0
        sharpe_ratios.append(sharpe)
    
    p_ruin = ruin_count / n_sims
    
    result = {
        "n_simulations": n_sims,
        "n_trades": n_trades,
        "p_ruin": p_ruin,
        "pass": p_ruin < MAX_RUIN_PROBABILITY,
        "max_drawdown_mean": float(np.mean(max_drawdowns)),
        "max_drawdown_95pct": float(np.percentile(max_drawdowns, 95)),
        "max_drawdown_99pct": float(np.percentile(max_drawdowns, 99)),
        "final_equity_mean": float(np.mean(final_equities)),
        "final_equity_5pct": float(np.percentile(final_equities, 5)),
        "final_equity_95pct": float(np.percentile(final_equities, 95)),
        "sharpe_mean": float(np.mean(sharpe_ratios)),
        "sharpe_5pct": float(np.percentile(sharpe_ratios, 5)),
        "win_rate": float(np.mean(results > 0)),
        "avg_win": float(np.mean(results[results > 0])) if np.any(results > 0) else 0,
        "avg_loss": float(np.mean(results[results <= 0])) if np.any(results <= 0) else 0,
    }
    
    status = "PASS" if result["pass"] else "FAIL"
    log.info("Monte Carlo [%s]: P(ruin)=%.1f%%, MaxDD_95=%.1f%%, Sharpe=%.2f",
             status, p_ruin * 100, result["max_drawdown_95pct"] * 100, result["sharpe_mean"])
    
    return result


def print_monte_carlo_report(result):
    """Print formatted Monte Carlo report."""
    if "error" in result:
        print(f"  Monte Carlo: {result['error']}")
        return
    
    status = "PASS" if result["pass"] else "FAIL"
    print(f"\n  MONTE CARLO STRESS TEST ({result['n_simulations']} simulations)")
    print(f"  {'='*50}")
    print(f"  Trades: {result['n_trades']}")
    print(f"  Win rate: {result['win_rate']:.1%}")
    print(f"  Avg win: {result['avg_win']:.2f}, Avg loss: {result['avg_loss']:.2f}")
    print(f"  P(ruin > 20%): {result['p_ruin']:.1%} [{status}]")
    print(f"  Max drawdown: mean={result['max_drawdown_mean']:.1%}, "
          f"95th={result['max_drawdown_95pct']:.1%}")
    print(f"  Final equity: {result['final_equity_5pct']:.0f} — "
          f"{result['final_equity_95pct']:.0f} (mean={result['final_equity_mean']:.0f})")
    print(f"  Sharpe: {result['sharpe_mean']:.2f} (5th pct: {result['sharpe_5pct']:.2f})")
