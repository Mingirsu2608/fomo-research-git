"""Create the 32 rolling-window features used by the baseline notebook.

Six features retain the executable Week 1 formulas. The other 26 reconstruct
the definitions in Clustering_Presentation.md because their Python source is
not present. Every market signal uses information available by t-1.
"""

from pathlib import Path

# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd


CANDIDATE_32 = [
    "rally_buy_lift", "near_high_buy_lift", "ath_proximity_buy_ratio",
    "green_streak_buy_bias", "volume_surge_buy_lift", "high_volatility_buy_lift",
    "rally_size_excess", "near_high_size_lift", "rally_cash_excess",
    "max_cash_deployment_ratio", "all_in_frequency", "net_buy_cash_ratio",
    "rally_burst_excess", "consecutive_buy_run_ratio", "buy_frequency_acceleration",
    "inter_buy_gap_cv", "weekend_buy_share", "order_intensity_per_active_week",
    "rally_reactivation_excess", "dormant_breakout_buy_rate", "fast_reentry_rate",
    "inactivity_to_rally_ratio", "first_buy_in_rally_bias", "dip_buy_lift",
    "fomo_vs_dip_ratio", "pullback_entry_bias", "buy_sell_momentum_imbalance",
    "rally_sell_suppression", "sell_to_buy_day_ratio",
    "portfolio_turnover_velocity", "panic_sell_post_top_rate",
    "loss_cut_vs_profit_exit_ratio",
]

REQUIRED_MARKET_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
REQUIRED_TRADE_COLUMNS = {
    "order_id", "timestamp", "investor_id", "side", "price", "quantity",
    "amount", "fee", "cash_after",
}

RETURN_DAYS, RALLY_RETURN = 7, 0.10
HIGH_LOOKBACK, NEAR_HIGH_RATIO = 90, 0.95
WINDOW_DAYS, STEP_DAYS = 180, 30
BASELINE_DAYS, MIN_HISTORY_DAYS, MIN_BASELINE_BUYS = 90, 30, 3
MIN_CONTEXT_DAYS, MIN_CONDITION_BUY_DAYS = 5, 2
BURST_DAYS, CASH_DAYS = 7, 3
GREEN_STREAK_DAYS, VOLUME_LOOKBACK, VOLATILITY_LOOKBACK = 3, 30, 30
DIP_RETURN, FAST_REENTRY_DAYS, PANIC_SELL_DAYS, ALL_IN_RATIO = -0.05, 7, 7, 0.80


def _check_columns(frame, required, label):
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _ratio(numerator, denominator):
    if denominator is None or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _bounded_change(current, baseline):
    denominator = current + baseline
    return (current - baseline) / denominator.where(denominator > 0)


def _contrast(frame, value, state, known, minimum):
    good = known.fillna(False) & frame[value].notna()
    yes, no = good & state.fillna(False), good & ~state.fillna(False)
    if int(yes.sum()) < minimum or int(no.sum()) < minimum:
        return np.nan
    return float(frame.loc[yes, value].mean() - frame.loc[no, value].mean())


def _market_context(market):
    market = market.copy()
    market["date"] = pd.to_datetime(market["timestamp"]).dt.normalize()
    market = market.set_index("date").sort_index()
    if market.index.duplicated().any():
        raise ValueError("Market data contains duplicate dates")

    context = pd.DataFrame(index=market.index)
    daily_return = market["close"].pct_change(fill_method=None)
    context["past_return"] = market["close"].pct_change(
        RETURN_DAYS, fill_method=None
    ).shift(1)
    context["prior_high"] = market["high"].rolling(
        HIGH_LOOKBACK, min_periods=HIGH_LOOKBACK
    ).max().shift(1)
    context["rally_known"] = context["past_return"].notna()
    context["near_known"] = context["prior_high"].notna()
    context["rally"] = context["past_return"].ge(RALLY_RETURN)
    context["near_high"] = market["close"].shift(1).div(context["prior_high"]).ge(
        NEAR_HIGH_RATIO
    )

    positive = daily_return.gt(0).astype(float)
    context["green_streak"] = positive.shift(1).rolling(
        GREEN_STREAK_DAYS, min_periods=GREEN_STREAK_DAYS
    ).sum().eq(GREEN_STREAK_DAYS)
    volume_median = market["volume"].shift(1).rolling(
        VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK
    ).median()
    context["volume_surge"] = market["volume"].shift(1).gt(1.5 * volume_median)
    true_range = (market["high"] - market["low"]).div(market["open"])
    volatility_limit = true_range.shift(1).rolling(
        VOLATILITY_LOOKBACK, min_periods=VOLATILITY_LOOKBACK
    ).quantile(0.75)
    context["high_volatility"] = true_range.shift(1).gt(volatility_limit)
    context["dip"] = context["past_return"].le(DIP_RETURN)
    context["pullback"] = context["past_return"].gt(0) & daily_return.shift(1).lt(0)
    context["momentum"] = context["past_return"].abs().ge(0.05)
    context["left_top"] = context["near_high"].shift(1).eq(True) & ~context["near_high"]
    context["post_top"] = context["left_top"].shift(1).rolling(
        PANIC_SELL_DAYS, min_periods=1
    ).max().fillna(0).astype(bool)
    return market, context


def _daily_trades(trades):
    trades = trades.copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    trades["date"] = trades["timestamp"].dt.normalize()
    trades["side"] = trades["side"].str.strip().str.upper()
    if not trades["side"].isin(["BUY", "SELL"]).all():
        raise ValueError("Trade side must be BUY or SELL")
    trades = trades.sort_values(["investor_id", "timestamp", "order_id"])
    trades["is_buy"] = trades["side"].eq("BUY")
    trades["is_sell"] = trades["side"].eq("SELL")
    trades["buy_amount"] = trades["amount"].where(trades["is_buy"], 0.0)
    trades["sell_amount"] = trades["amount"].where(trades["is_sell"], 0.0)
    trades["buy_cost"] = (trades["amount"] + trades["fee"]).where(trades["is_buy"], 0.0)
    trades["sell_proceeds"] = (trades["amount"] - trades["fee"]).where(trades["is_sell"], 0.0)
    trades["cash_before"] = np.where(
        trades["is_buy"],
        trades["cash_after"] + trades["amount"] + trades["fee"],
        trades["cash_after"] - trades["amount"] + trades["fee"],
    )

    prior_value = trades["buy_amount"].groupby(trades["investor_id"]).cumsum() - trades["buy_amount"]
    buy_quantity = trades["quantity"].where(trades["is_buy"], 0.0)
    prior_quantity = buy_quantity.groupby(trades["investor_id"]).cumsum() - buy_quantity
    trades["reference_buy_price"] = prior_value / prior_quantity.replace(0, np.nan)
    trades["loss_exit"] = trades["is_sell"] & trades["price"].lt(trades["reference_buy_price"])
    trades["profit_exit"] = trades["is_sell"] & trades["price"].ge(trades["reference_buy_price"])

    return trades.groupby(["investor_id", "date"], sort=False).agg(
        n_trades=("order_id", "size"), n_buys=("is_buy", "sum"),
        n_sells=("is_sell", "sum"), buy_amount=("buy_amount", "sum"),
        sell_amount=("sell_amount", "sum"), buy_cost=("buy_cost", "sum"),
        sell_proceeds=("sell_proceeds", "sum"), first_cash=("cash_before", "first"),
        cash_after=("cash_after", "last"), loss_exits=("loss_exit", "sum"),
        profit_exits=("profit_exit", "sum"),
    )


def _prepare_daily(raw, calendar, context):
    first_date = raw.index.min()
    daily = raw.reindex(calendar[calendar >= first_date]).copy().join(context)
    zero_columns = [
        "n_trades", "n_buys", "n_sells", "buy_amount", "sell_amount",
        "buy_cost", "sell_proceeds", "loss_exits", "profit_exits",
    ]
    daily[zero_columns] = daily[zero_columns].fillna(0)
    daily["buy_day"] = daily["n_buys"].gt(0)
    daily["sell_day"] = daily["n_sells"].gt(0)
    daily["active_day"] = daily["n_trades"].gt(0)
    daily["cash_after"] = daily["cash_after"].ffill()
    daily["cash_start"] = daily["cash_after"].shift(1)
    if pd.notna(daily.loc[first_date, "first_cash"]):
        daily.loc[first_date, "cash_start"] = daily.loc[first_date, "first_cash"]

    prior_n = daily["n_buys"].shift(1).rolling(
        BASELINE_DAYS, min_periods=MIN_HISTORY_DAYS
    ).sum()
    prior_amount = daily["buy_amount"].shift(1).rolling(
        BASELINE_DAYS, min_periods=MIN_HISTORY_DAYS
    ).sum()
    prior_mean = prior_amount / prior_n.where(prior_n >= MIN_BASELINE_BUYS)
    order_mean = daily["buy_amount"] / daily["n_buys"].where(daily["n_buys"] > 0)
    daily["size_change"] = _bounded_change(order_mean, prior_mean)

    past_buys = daily["n_buys"].shift(BURST_DAYS).rolling(
        BASELINE_DAYS, min_periods=MIN_HISTORY_DAYS
    )
    baseline_rate = past_buys.mean().where(past_buys.sum() >= MIN_BASELINE_BUYS)
    recent_count = daily["n_buys"].rolling(BURST_DAYS, min_periods=BURST_DAYS).sum()
    daily["burst_change"] = _bounded_change(recent_count, BURST_DAYS * baseline_rate)

    net_spend = (daily["buy_cost"] - daily["sell_proceeds"]).rolling(
        CASH_DAYS, min_periods=CASH_DAYS
    ).sum()
    cash_at_start = daily["cash_start"].shift(CASH_DAYS - 1)
    daily["cash_deployment"] = (
        net_spend / cash_at_start.where(cash_at_start > 0)
    ).clip(0, 1)

    dates = pd.Series(daily.index, index=daily.index)
    last_buy = dates.where(daily["buy_day"]).ffill().shift(1)
    daily["buy_gap"] = (dates - last_buy).dt.days.clip(upper=BASELINE_DAYS)
    history = daily["n_buys"].shift(1).rolling(
        BASELINE_DAYS, min_periods=MIN_HISTORY_DAYS
    )
    daily["expected_gap"] = (
        1 / history.mean().where(history.sum() >= MIN_BASELINE_BUYS)
    ).clip(upper=BASELINE_DAYS)
    daily["reactivation_change"] = _bounded_change(
        daily["buy_gap"], daily["expected_gap"]
    ).where(daily["buy_day"])
    daily["dormant_buy"] = daily["buy_day"] & daily["buy_gap"].gt(daily["expected_gap"])
    last_sell = dates.where(daily["sell_day"]).ffill().shift(1)
    daily["days_since_sell"] = (dates - last_sell).dt.days
    return daily


def _run_ratio(values):
    values = np.asarray(values, dtype=bool)
    if values.sum() == 0:
        return np.nan
    previous, following = np.r_[False, values[:-1]], np.r_[values[1:], False]
    return float((values & (previous | following)).sum() / values.sum())


def _window_features(window):
    buy, sell = window["buy_day"], window["sell_day"]
    n_buy, n_sell = int(buy.sum()), int(sell.sum())
    features = {
        "rally_buy_lift": _contrast(window, "buy_day", window["rally"], window["rally_known"], MIN_CONTEXT_DAYS),
        "near_high_buy_lift": _contrast(window, "buy_day", window["near_high"], window["near_known"], MIN_CONTEXT_DAYS),
        "rally_size_excess": _contrast(window, "size_change", window["rally"], window["rally_known"] & buy, MIN_CONDITION_BUY_DAYS),
        "rally_cash_excess": _contrast(window, "cash_deployment", window["rally"], window["rally_known"] & buy, MIN_CONDITION_BUY_DAYS),
        "rally_burst_excess": 0.5 * _contrast(window, "burst_change", window["rally"], window["rally_known"], MIN_CONTEXT_DAYS),
        "rally_reactivation_excess": _contrast(window, "reactivation_change", window["rally"], window["rally_known"] & buy, MIN_CONDITION_BUY_DAYS),
        "ath_proximity_buy_ratio": _ratio((buy & window["near_high"]).sum(), n_buy),
        "green_streak_buy_bias": _contrast(window, "buy_day", window["green_streak"], window["rally_known"], MIN_CONTEXT_DAYS),
        "volume_surge_buy_lift": _contrast(window, "buy_day", window["volume_surge"], window["rally_known"], MIN_CONTEXT_DAYS),
        "high_volatility_buy_lift": _contrast(window, "buy_day", window["high_volatility"], window["rally_known"], MIN_CONTEXT_DAYS),
        "near_high_size_lift": _contrast(window, "size_change", window["near_high"], window["near_known"] & buy, MIN_CONDITION_BUY_DAYS),
        "max_cash_deployment_ratio": float(window.loc[buy, "cash_deployment"].max()) if buy.any() else np.nan,
        "all_in_frequency": _ratio((buy & window["cash_deployment"].ge(ALL_IN_RATIO)).sum(), n_buy),
        "consecutive_buy_run_ratio": _run_ratio(buy),
        "inter_buy_gap_cv": np.nan,
        "weekend_buy_share": _ratio((buy & (window.index.dayofweek >= 5)).sum(), n_buy),
        "sell_to_buy_day_ratio": _ratio(n_sell, n_buy),
        "loss_cut_vs_profit_exit_ratio": _ratio(window["loss_exits"].sum(), window["loss_exits"].sum() + window["profit_exits"].sum()),
    }

    buy_dates = window.index[buy]
    if len(buy_dates) >= 3:
        gaps = pd.Series(buy_dates).diff().dt.days.dropna()
        features["inter_buy_gap_cv"] = _ratio(gaps.std(ddof=0), gaps.mean())
    half = len(window) // 2
    early, late = window["buy_day"].iloc[:half].mean(), window["buy_day"].iloc[half:].mean()
    features["buy_frequency_acceleration"] = (late - early) / (late + early) if late + early > 0 else np.nan
    active_weeks = window.loc[window["active_day"]].index.to_period("W").nunique()
    features["order_intensity_per_active_week"] = _ratio(window["n_trades"].sum(), active_weeks)
    features["fast_reentry_rate"] = _ratio((buy & window["days_since_sell"].between(1, FAST_REENTRY_DAYS)).sum(), n_sell)

    dormant = window["dormant_buy"]
    dormant_rally = _ratio((dormant & window["rally"]).sum(), dormant.sum())
    features["dormant_breakout_buy_rate"] = dormant_rally
    features["inactivity_to_rally_ratio"] = dormant_rally
    rally_opportunity = window.loc[window["rally_known"], "rally"].mean()
    features["first_buy_in_rally_bias"] = dormant_rally - rally_opportunity if np.isfinite(dormant_rally) and pd.notna(rally_opportunity) else np.nan
    features["dip_buy_lift"] = _contrast(window, "buy_day", window["dip"], window["rally_known"], MIN_CONTEXT_DAYS)
    fomo, dip = features["rally_buy_lift"], features["dip_buy_lift"]
    features["fomo_vs_dip_ratio"] = (fomo - dip) / (abs(fomo) + abs(dip)) if np.isfinite(fomo) and np.isfinite(dip) and abs(fomo) + abs(dip) > 0 else np.nan
    features["pullback_entry_bias"] = _contrast(window, "buy_day", window["pullback"], window["rally_known"], MIN_CONTEXT_DAYS)

    momentum = window["momentum"] & window["rally_known"]
    activity = window.loc[momentum, "n_buys"] + window.loc[momentum, "n_sells"]
    features["buy_sell_momentum_imbalance"] = _ratio((window.loc[momentum, "n_buys"] - window.loc[momentum, "n_sells"]).sum(), activity.sum())
    rally_sell_lift = _contrast(window, "sell_day", window["rally"], window["rally_known"], MIN_CONTEXT_DAYS)
    features["rally_sell_suppression"] = -rally_sell_lift if np.isfinite(rally_sell_lift) else np.nan

    cash = window["cash_start"].dropna()
    cash_reference = float(cash.iloc[0]) if len(cash) else np.nan
    net_buy = window["buy_cost"].sum() - window["sell_proceeds"].sum()
    features["net_buy_cash_ratio"] = _ratio(net_buy, cash_reference)
    features["portfolio_turnover_velocity"] = _ratio(window["buy_amount"].sum() + window["sell_amount"].sum(), cash_reference * max(len(window) / 7, 1))
    features["panic_sell_post_top_rate"] = _ratio((sell & window["post_top"]).sum(), window["post_top"].sum())
    return features


def _investor_windows(investor_id, raw, calendar, context, window_ends):
    daily, rows = _prepare_daily(raw, calendar, context), []
    for end in window_ends:
        start = end - pd.Timedelta(days=WINDOW_DAYS - 1)
        if end < daily.index.min():
            continue
        window = daily.loc[start:end]
        row = {
            "investor_id": investor_id, "window_start": start, "window_end": end,
            "observed_days": len(window), "n_trades": int(window["n_trades"].sum()),
            "n_buys": int(window["n_buys"].sum()), "buy_days": int(window["buy_day"].sum()),
            "sell_days": int(window["sell_day"].sum()),
            "rally_days": int((window["rally_known"] & window["rally"]).sum()),
            "calm_days": int((window["rally_known"] & ~window["rally"]).sum()),
            "buy_amount": float(window["buy_amount"].sum()),
            "sell_amount": float(window["sell_amount"].sum()),
        }
        row.update(_window_features(window))
        rows.append(row)
    return rows


def build_features(config):
    """Read both raw files and write one row per investor/180-day window."""
    market = pd.read_csv(config["data"]["market_path"])
    trades = pd.read_csv(config["data"]["trades_path"], dtype={"order_id": str, "investor_id": str})
    _check_columns(market, REQUIRED_MARKET_COLUMNS, "Market data")
    _check_columns(trades, REQUIRED_TRADE_COLUMNS, "Trade log")
    market, context = _market_context(market)
    daily_trades = _daily_trades(trades)
    calendar = pd.date_range(market.index.min(), market.index.max(), freq="D")
    if not market.index.equals(calendar):
        raise ValueError("OHLCV calendar has missing dates")

    window_days = int(config["features"].get("window_days", WINDOW_DAYS))
    step_days = int(config["features"].get("step_days", STEP_DAYS))
    if window_days != WINDOW_DAYS:
        raise ValueError("This baseline reproduces the original 180-day window")
    window_ends = pd.date_range(calendar.min() + pd.Timedelta(days=window_days - 1), calendar.max(), freq=f"{step_days}D")
    if len(window_ends) and window_ends[-1] != calendar.max():
        window_ends = window_ends.append(pd.DatetimeIndex([calendar.max()]))

    rows = []
    for investor_id, raw in daily_trades.groupby(level="investor_id", sort=True):
        rows.extend(_investor_windows(investor_id, raw.droplevel("investor_id"), calendar, context, window_ends))
    features = pd.DataFrame(rows)
    features[CANDIDATE_32] = features[CANDIDATE_32].replace([np.inf, -np.inf], np.nan)
    min_buys = int(config["features"].get("min_buys", config["features"].get("min_trades", 5)))
    min_buy_days = int(config["features"].get("min_buy_days", 3))
    features = features.loc[
        features["observed_days"].eq(window_days)
        & features["n_buys"].ge(min_buys)
        & features["buy_days"].ge(min_buy_days)
    ].sort_values(["window_end", "investor_id"]).reset_index(drop=True)
    output = Path(config["features"]["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    return features
