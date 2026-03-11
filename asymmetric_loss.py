"""
Asymmetric Loss — V3 Layer 6: Custom XGBoost objective with FP penalty.

In trading, a false positive (predicting UP but price goes DOWN) costs money.
A false negative (missing a trade) only costs opportunity.
Therefore FP should be penalized more than FN.

This module provides:
  1. Custom XGBoost objective function with asymmetric FP penalty
  2. Custom evaluation metric matching the asymmetric loss
"""

import numpy as np
import logging

log = logging.getLogger("asymmetric_loss")

FP_PENALTY = 3.0  # False positive penalty multiplier


def asymmetric_logloss_objective(y_true, y_pred):
    """
    Custom XGBoost objective: asymmetric log loss.
    
    FP penalty = 3× (predicting UP when price goes DOWN is expensive).
    FN penalty = 1× (missing a trade is cheaper).
    
    Sklearn API: receives (y_true, y_pred) as numpy arrays.
    Returns (gradient, hessian) for XGBoost.
    """
    # Sigmoid transform
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    
    # Asymmetric weights: FP gets higher penalty
    # When y_true=0 and p is high → FP → higher weight
    # When y_true=1 and p is low → FN → normal weight
    weights = np.where(y_true == 0, FP_PENALTY, 1.0)
    
    # Gradient: weighted cross-entropy gradient
    grad = weights * (p - y_true)
    
    # Hessian: weighted cross-entropy hessian
    hess = weights * p * (1 - p)
    
    return grad, hess


def asymmetric_logloss_eval(y_true, y_pred):
    """
    Custom XGBoost evaluation metric matching the asymmetric objective.
    
    Sklearn API: receives (y_true, y_pred) as numpy arrays.
    Returns (name, value).
    """
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    
    weights = np.where(y_true == 0, FP_PENALTY, 1.0)
    
    loss = -weights * (y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    
    return "asymm_logloss", float(np.mean(loss))


def get_asymmetric_xgb_params(base_params=None):
    """
    Get XGBoost params configured for asymmetric loss.
    
    Replaces default objective with custom asymmetric objective.
    Returns params dict + objective function + eval function.
    """
    params = base_params.copy() if base_params else {}
    
    # Remove objective since we use custom
    params.pop("objective", None)
    params.pop("eval_metric", None)
    
    # Disable default objective
    params["disable_default_eval_metric"] = True
    
    return params, asymmetric_logloss_objective, asymmetric_logloss_eval
