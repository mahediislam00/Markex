"""
MARKEX XGBoost Forecasting Engine
Fetches historical data via yfinance, engineers features, trains an XGBoost
model and forecasts 1-day, 1-week and 1-month price targets.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# ── Symbol mapping: internal ID → yfinance ticker ─────────────────────────────
SYMBOL_MAP = {
    "BTC":    "BTC-USD",
    "ETH":    "ETH-USD",
    "SOL":    "SOL-USD",
    "BNB":    "BNB-USD",
    "XRP":    "XRP-USD",
    "AAPL":   "AAPL",
    "NVDA":   "NVDA",
    "MSFT":   "MSFT",
    "TSLA":   "TSLA",
    "AMZN":   "AMZN",
    "XAU":    "GC=F",
    "XAG":    "SI=F",
    "OIL":    "CL=F",
    "NG":     "NG=F",
    "SPY":    "SPY",
    "QQQ":    "QQQ",
    "VTI":    "VTI",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
}

# Forecast horizons in trading days
HORIZONS = {"1d": 1, "1w": 5, "1m": 21}


def _fetch_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Download OHLCV data from yfinance."""
    data = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}")
    data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
    data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return data


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer 40+ technical indicators as model features."""
    d = df.copy()
    c = d["Close"]
    h = d["High"]
    lo = d["Low"]
    v = d["Volume"]

    # ── Returns ──────────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 21]:
        d[f"ret_{lag}"] = c.pct_change(lag)

    # ── Moving averages ───────────────────────────────────────────────────────
    for w in [5, 10, 20, 50]:
        d[f"sma_{w}"] = c.rolling(w).mean()
        d[f"sma_ratio_{w}"] = c / d[f"sma_{w}"]

    # ── Exponential MAs ───────────────────────────────────────────────────────
    for span in [12, 26]:
        d[f"ema_{span}"] = c.ewm(span=span, adjust=False).mean()
    d["macd"] = d["ema_12"] - d["ema_26"]
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # ── RSI ───────────────────────────────────────────────────────────────────
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    d["bb_upper"] = bb_mid + 2 * bb_std
    d["bb_lower"] = bb_mid - 2 * bb_std
    d["bb_pct"] = (c - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-10)
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / (bb_mid + 1e-10)

    # ── ATR ───────────────────────────────────────────────────────────────────
    tr = pd.concat([
        h - lo,
        (h - c.shift()).abs(),
        (lo - c.shift()).abs()
    ], axis=1).max(axis=1)
    d["atr_14"] = tr.rolling(14).mean()
    d["atr_pct"] = d["atr_14"] / (c + 1e-10)

    # ── Stochastic ────────────────────────────────────────────────────────────
    low14  = lo.rolling(14).min()
    high14 = h.rolling(14).max()
    d["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-10)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # ── Volume signals ────────────────────────────────────────────────────────
    d["vol_ma20"] = v.rolling(20).mean()
    d["vol_ratio"] = v / (d["vol_ma20"] + 1e-10)
    d["obv"] = (np.sign(c.diff()) * v).cumsum()
    d["obv_norm"] = d["obv"] / (d["obv"].rolling(20).std() + 1e-10)

    # ── Price channel / range ─────────────────────────────────────────────────
    d["high_low_range"] = (h - lo) / (c + 1e-10)
    d["close_position"] = (c - lo) / (h - lo + 1e-10)

    # ── Log price (helps with heteroscedasticity) ─────────────────────────────
    d["log_close"] = np.log(c + 1e-10)

    # ── Day-of-week seasonality ───────────────────────────────────────────────
    d["dow"] = d.index.dayofweek
    d["month"] = d.index.month

    return d


def _build_dataset(df: pd.DataFrame, horizon: int):
    """
    Build X, y for supervised learning.
    Target = forward log return over `horizon` days.
    """
    df = _add_features(df)
    df["target"] = np.log(df["Close"].shift(-horizon) / df["Close"])
    df = df.dropna()

    feature_cols = [c for c in df.columns
                    if c not in ("Open", "High", "Low", "Close", "Volume", "target")]
    X = df[feature_cols].values
    y = df["target"].values
    return X, y, feature_cols, df["Close"].values


def _train_xgb(X_train, y_train) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def forecast(symbol_id: str) -> dict:
    """
    Main entry point.
    Returns a dict with forecast prices, confidence metrics, and signal data.
    """
    ticker = SYMBOL_MAP.get(symbol_id)
    if not ticker:
        raise ValueError(f"Unknown symbol: {symbol_id}")

    try:
        raw = _fetch_ohlcv(ticker)
    except Exception as e:
        logger.error(f"Data fetch failed for {ticker}: {e}")
        raise

    if len(raw) < 100:
        raise ValueError(f"Insufficient data for {symbol_id} ({len(raw)} rows)")

    current_price = float(raw["Close"].iloc[-1])
    results = {"symbol": symbol_id, "current_price": current_price, "forecasts": {}}

    # ── Per-horizon training ──────────────────────────────────────────────────
    horizon_models = {}
    for label, h in HORIZONS.items():
        X, y, feat_cols, closes = _build_dataset(raw, h)

        # Train / val split (last 20% for validation)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)

        model = _train_xgb(X_train_s, y_train)

        # Validation MAPE (on price level)
        pred_log_ret_val = model.predict(X_val_s)
        actual_prices_val   = closes[split + h: split + h + len(y_val)]
        pred_prices_val     = closes[split: split + len(y_val)] * np.exp(pred_log_ret_val)
        min_len = min(len(actual_prices_val), len(pred_prices_val))
        if min_len > 5:
            mape = mean_absolute_percentage_error(
                actual_prices_val[:min_len], pred_prices_val[:min_len]
            )
        else:
            mape = 0.05

        # Predict on latest window
        X_latest = scaler.transform(X[-1:])
        pred_log_ret = float(model.predict(X_latest)[0])
        pred_price   = current_price * np.exp(pred_log_ret)
        pct_change   = (pred_price - current_price) / current_price * 100

        # Confidence: heuristic based on validation error
        confidence = max(40, min(95, int(100 * (1 - mape * 5))))

        results["forecasts"][label] = {
            "price":       round(pred_price, 6 if current_price < 1 else (2 if current_price < 1000 else 0)),
            "pct_change":  round(pct_change, 2),
            "direction":   "up" if pct_change > 0 else "down",
            "confidence":  confidence,
            "mape":        round(mape * 100, 2),
        }
        horizon_models[label] = (model, scaler, feat_cols)

    # ── Aggregate AI panel signals ────────────────────────────────────────────
    df_feat = _add_features(raw)
    latest  = df_feat.iloc[-1]

    rsi_val    = float(latest.get("rsi_14", 50))
    macd_hist  = float(latest.get("macd_hist", 0))
    bb_pct_val = float(latest.get("bb_pct", 0.5))
    vol_ratio  = float(latest.get("vol_ratio", 1.0))
    stoch_k    = float(latest.get("stoch_k", 50))

    # Overall sentiment from 1d forecast
    fc_1d = results["forecasts"]["1d"]
    fc_1w = results["forecasts"]["1w"]
    fc_1m = results["forecasts"]["1m"]

    votes = sum([
        fc_1d["pct_change"] > 0,
        fc_1w["pct_change"] > 0,
        fc_1m["pct_change"] > 0,
    ])
    if votes >= 2 and fc_1w["pct_change"] > 0.5:
        sentiment = "BULLISH"
    elif votes <= 1 and fc_1w["pct_change"] < -0.5:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    avg_conf = int(np.mean([fc_1d["confidence"], fc_1w["confidence"], fc_1m["confidence"]]))

    # Signal bars
    trend_strength = min(95, max(10, int(abs(fc_1w["pct_change"]) * 15 + 40)))
    volatility_score = min(95, max(10, int(float(latest.get("atr_pct", 0.02)) * 1500)))
    volume_signal  = min(95, max(10, int((vol_ratio - 1) * 50 + 50)))
    momentum_score = min(95, max(10, int(rsi_val)))

    # Technical signals
    macd_status   = "DETECTED" if macd_hist > 0 else "BEARISH"
    rsi_status    = "OVERBOUGHT" if rsi_val > 70 else ("OVERSOLD" if rsi_val < 30 else "NORMAL")
    vol_status    = "SPIKE" if vol_ratio > 2 else ("HIGH" if vol_ratio > 1.5 else "INACTIVE")
    support_status = "NEAR" if bb_pct_val < 0.2 else ("AT RESIST" if bb_pct_val > 0.8 else "CLEAR")

    # 24h stats
    high_24 = float(raw["High"].iloc[-1])
    low_24  = float(raw["Low"].iloc[-1])
    open_24 = float(raw["Open"].iloc[-1])

    # Volume string
    vol_raw = float(raw["Volume"].iloc[-1])
    if vol_raw >= 1e12:
        vol_str = f"{vol_raw/1e12:.1f}T"
    elif vol_raw >= 1e9:
        vol_str = f"{vol_raw/1e9:.1f}B"
    elif vol_raw >= 1e6:
        vol_str = f"{vol_raw/1e6:.1f}M"
    else:
        vol_str = f"{vol_raw:.0f}"

    # 7d change
    if len(raw) >= 8:
        chg_7d = (current_price - float(raw["Close"].iloc[-8])) / float(raw["Close"].iloc[-8]) * 100
    else:
        chg_7d = 0.0

    results.update({
        "sentiment":       sentiment,
        "avg_confidence":  avg_conf,
        "signal_bars": {
            "trend_strength": trend_strength,
            "volatility":     volatility_score,
            "volume_signal":  volume_signal,
            "momentum":       momentum_score,
        },
        "technical_signals": {
            "macd":    {"status": macd_status,    "active": macd_hist > 0},
            "rsi":     {"status": rsi_status,     "active": rsi_val > 70 or rsi_val < 30, "value": round(rsi_val, 1)},
            "volume":  {"status": vol_status,     "active": vol_ratio > 1.5},
            "support": {"status": support_status, "active": bb_pct_val < 0.2 or bb_pct_val > 0.8},
        },
        "market_data": {
            "high_24":   round(high_24, 4 if current_price < 1 else 2),
            "low_24":    round(low_24, 4 if current_price < 1 else 2),
            "open_24":   round(open_24, 4 if current_price < 1 else 2),
            "volume":    vol_str,
            "chg_7d":    round(chg_7d, 2),
            "rsi":       round(rsi_val, 1),
            "bb_pct":    round(bb_pct_val * 100, 1),
            "stoch_k":   round(stoch_k, 1),
        },
    })

    return results


# ── Demo / offline fallback ────────────────────────────────────────────────────
import random

def _demo_forecast(symbol_id: str) -> dict:
    """
    Returns a plausible-looking forecast using only the base prices from
    INSTRUMENTS when yfinance cannot be reached (e.g. no internet / rate-limited).
    The XGBoost signal structure is preserved so the UI renders correctly.
    """
    BASE_PRICES = {
        "BTC":67420.50,"ETH":3842.10,"SOL":182.45,"BNB":598.30,"XRP":0.621,
        "AAPL":228.35,"NVDA":875.20,"MSFT":418.90,"TSLA":242.10,"AMZN":196.45,
        "XAU":2312.40,"XAG":27.84,"OIL":78.92,"NG":2.145,"SPY":542.10,
        "QQQ":468.35,"VTI":248.90,"EURUSD":1.0842,"GBPUSD":1.271,"USDJPY":151.84,
    }
    rng = random.Random(symbol_id + str(int(datetime.utcnow().timestamp() // 600)))
    base = BASE_PRICES.get(symbol_id, 100.0)

    def fc(days):
        drift = (rng.random() - 0.45) * 0.008 * days
        pct   = round(drift * 100, 2)
        price = base * (1 + drift)
        dp    = 4 if base < 1 else (2 if base < 1000 else 0)
        return {"price": round(price, dp), "pct_change": pct,
                "direction": "up" if pct > 0 else "down",
                "confidence": rng.randint(55, 88), "mape": round(rng.uniform(1, 6), 2)}

    sentiment_votes = [fc(1)["pct_change"] > 0, fc(5)["pct_change"] > 0, fc(21)["pct_change"] > 0]
    sentiment = "BULLISH" if sum(sentiment_votes) >= 2 else "BEARISH"

    dp = 4 if base < 1 else 2
    return {
        "symbol": symbol_id,
        "current_price": base,
        "demo_mode": True,
        "forecasts": {"1d": fc(1), "1w": fc(5), "1m": fc(21)},
        "sentiment": sentiment,
        "avg_confidence": rng.randint(60, 82),
        "signal_bars": {
            "trend_strength": rng.randint(45, 90),
            "volatility":     rng.randint(20, 65),
            "volume_signal":  rng.randint(30, 75),
            "momentum":       rng.randint(40, 85),
        },
        "technical_signals": {
            "macd":    {"status": "DETECTED", "active": True},
            "rsi":     {"status": "NORMAL",   "active": False, "value": round(rng.uniform(40, 65), 1)},
            "volume":  {"status": "INACTIVE", "active": False},
            "support": {"status": "NEAR",     "active": True},
        },
        "market_data": {
            "high_24":  round(base * 1.018, dp),
            "low_24":   round(base * 0.982, dp),
            "open_24":  round(base * 0.995, dp),
            "volume":   "—",
            "chg_7d":   round((rng.random() - 0.45) * 8, 2),
            "rsi":      round(rng.uniform(35, 70), 1),
            "bb_pct":   round(rng.uniform(20, 80), 1),
            "stoch_k":  round(rng.uniform(20, 80), 1),
        },
    }


def forecast_with_fallback(symbol_id: str) -> dict:
    """Tries real XGBoost forecast; falls back to demo mode on network error."""
    try:
        return forecast(symbol_id)
    except Exception as e:
        logger.warning(f"Live forecast failed for {symbol_id} ({e}); using demo mode")
        result = _demo_forecast(symbol_id)
        return result
