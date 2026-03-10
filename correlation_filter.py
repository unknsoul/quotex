"""
Cross-Pair Correlation Filter — Upgrade 1.

Checks if correlated currency pairs agree on direction.
If EURUSD predicts UP but GBPUSD (highly correlated) predicts DOWN,
the signal is unreliable and should be skipped.

Correlation groups (based on fundamental relationships):
  - USD-weak group: EURUSD, GBPUSD, AUDUSD (all move UP when USD weakens)
  - JPY-cross group: USDJPY, GBPJPY (both move with JPY flows)
  - Inverse: EURUSD vs USDJPY tend to move opposite
"""

import logging

log = logging.getLogger("correlation_filter")

# v10 Correlation clusters for EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF
CORRELATION_CLUSTERS = {
    "USD_weak": {
        "pairs": {"EURUSD", "GBPUSD", "AUDUSD"},
        "description": "All move UP when USD weakens",
    },
    "USD_strong": {
        "pairs": {"USDJPY", "USDCAD", "USDCHF"},
        "description": "All move UP when USD strengthens",
    },
}

# Inverse correlations: these pairs should move OPPOSITE
INVERSE_PAIRS = [
    ("EURUSD", "USDCHF"),  # EUR/CHF nearly 1:1 inverse via USD
    ("EURUSD", "USDJPY"),  # EUR up usually means USD down = JPY pairs down
    ("GBPUSD", "USDCAD"),  # GBP up / CAD down via USD
    ("AUDUSD", "USDCHF"),  # commodity vs safe haven
]


def check_correlation(symbol: str, direction: str,
                      all_predictions: dict) -> dict:
    """
    Check if the predicted direction for `symbol` is consistent
    with correlated pairs' predictions.

    Args:
        symbol: the pair being evaluated (e.g. 'EURUSD')
        direction: predicted direction ('UP' or 'DOWN')
        all_predictions: {sym: pred_dict} for all symbols predicted this cycle

    Returns:
        dict with:
            corr_pass: bool — True if correlation is consistent
            corr_score: 0-1 — fraction of correlated pairs that agree
            conflicts: list of conflicting pairs
            reason: human-readable explanation
    """
    conflicts = []
    agreements = 0
    total_checks = 0

    # Check cluster agreement
    for cluster_name, cluster in CORRELATION_CLUSTERS.items():
        if symbol not in cluster["pairs"]:
            continue

        # Find other predicted pairs in this cluster
        for other_sym in cluster["pairs"]:
            if other_sym == symbol or other_sym not in all_predictions:
                continue

            other_dir = all_predictions[other_sym].get("suggested_direction", "")
            total_checks += 1

            if other_dir == direction:
                agreements += 1
            else:
                conflicts.append(f"{other_sym}={other_dir}")

    # Check inverse correlations
    for pair_a, pair_b in INVERSE_PAIRS:
        if symbol == pair_a and pair_b in all_predictions:
            other_dir = all_predictions[pair_b].get("suggested_direction", "")
            total_checks += 1
            # Inverse: they should move OPPOSITE
            if other_dir != direction:
                agreements += 1  # opposite is correct for inverse pairs
            else:
                conflicts.append(f"{pair_b}={other_dir}(should be opposite)")

        elif symbol == pair_b and pair_a in all_predictions:
            other_dir = all_predictions[pair_a].get("suggested_direction", "")
            total_checks += 1
            if other_dir != direction:
                agreements += 1
            else:
                conflicts.append(f"{pair_a}={other_dir}(should be opposite)")

    # Score
    if total_checks == 0:
        # No correlated pairs available — pass by default
        return {
            "corr_pass": True,
            "corr_score": 1.0,
            "conflicts": [],
            "reason": "No correlated pairs to check",
        }

    corr_score = agreements / total_checks
    passed = corr_score >= 0.5  # v2: allow partial conflicts (was: ANY conflict = skip)

    if not passed:
        reason = f"Correlation conflict: {', '.join(conflicts)}"
    else:
        reason = f"Correlation OK ({agreements}/{total_checks} agree)"

    return {
        "corr_pass": passed,
        "corr_score": round(corr_score, 2),
        "conflicts": conflicts,
        "reason": reason,
    }
