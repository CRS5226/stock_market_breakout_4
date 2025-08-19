# indicator.py

import pandas as pd
import numpy as np


# ---------- helpers ----------
def _bps(curr, prev):
    """Return basis points change between two values (safe)."""
    if prev is None or prev == 0 or pd.isna(prev) or pd.isna(curr):
        return np.nan
    return (float(curr) - float(prev)) / float(prev) * 10000.0


def _rolling_vwap(df: pd.DataFrame, window: int = 60):
    """
    If session VWAP isn't feasible (no session resets), compute a rolling VWAP.
    Uses typical price = (H+L+C)/3. Falls back gracefully if columns missing.
    """
    if not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"].clip(lower=0)
    vwap = pv.rolling(window=window, min_periods=1).sum() / df["Volume"].rolling(
        window=window, min_periods=1
    ).sum().replace(0, np.nan)
    return vwap


def _session_vwap(df: pd.DataFrame, timestamp_col: str = "Timestamp"):
    """
    Session VWAP (resets each day) if you have a timestamp column that includes date.
    Expects 'Timestamp' to be pandas-parsable. If not present, returns NaNs.
    """
    if timestamp_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    try:
        ts = pd.to_datetime(df[timestamp_col])
    except Exception:
        return pd.Series(np.nan, index=df.index)

    if not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)

    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"].clip(lower=0)
    # group by date component
    g = ts.dt.date
    cum_pv = pv.groupby(g).cumsum()
    cum_v = df["Volume"].groupby(g).cumsum().replace(0, np.nan)
    return cum_pv / cum_v


def _atr(df: pd.DataFrame, period: int = 14):
    """ATR using Wilder smoothing (EMA with alpha=1/period)."""
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)
    high_low = (df["High"] - df["Low"]).abs()
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


# ---------- your existing functions, now extended ----------
def add_indicators(df: pd.DataFrame, config: dict):
    """
    Extends your current indicators with extra, LLM-friendly features:
      hh20, ll20, dist_hh20_bps, bb_width_bps, bb_squeeze,
      ema20_slope_bps, ema50_slope_bps, adx14 (alias of ADX),
      macd_hist_delta, vwap, vwap_diff_bps, atr_pct, vol_z
    Keeps all your original columns intact.
    """
    try:
        # --- Moving Averages (original) ---
        ma_fast = config.get("moving_averages", {}).get("ma_fast", 5)
        ma_slow = config.get("moving_averages", {}).get("ma_slow", 20)
        df["MA_Fast"] = df["Close"].rolling(window=ma_fast, min_periods=1).mean()
        df["MA_Slow"] = df["Close"].rolling(window=ma_slow, min_periods=1).mean()

        # --- Bollinger Bands (original) ---
        bb_cfg = config.get("bollinger", {})
        period = int(bb_cfg.get("period", 20))
        std_dev = float(bb_cfg.get("std_dev", 2))
        df["BB_Mid"] = df["Close"].rolling(window=period, min_periods=1).mean()
        df["BB_Std"] = df["Close"].rolling(window=period, min_periods=1).std(ddof=0)
        df["BB_Upper"] = df["BB_Mid"] + (df["BB_Std"] * std_dev)
        df["BB_Lower"] = df["BB_Mid"] - (df["BB_Std"] * std_dev)

        # --- MACD (original) ---
        macd_cfg = config.get("macd", {})
        fast = int(macd_cfg.get("fast_period", 12))
        slow = int(macd_cfg.get("slow_period", 26))
        signal_period = int(macd_cfg.get("signal_period", 9))
        ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
        df["MACD"] = ema_fast - ema_slow
        df["MACD_Signal"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # --- ADX (original) ---
        adx_period = int(config.get("adx", {}).get("period", 14))
        df = compute_adx(df, adx_period)

        # =========================
        #    ADDITIONAL FEATURES
        # =========================

        # 1) 20-bar highs/lows
        lookback = 20
        df["HH20"] = df["High"].rolling(window=lookback, min_periods=1).max()
        df["LL20"] = df["Low"].rolling(window=lookback, min_periods=1).min()

        # 2) Distance to HH20 in bps (positive when below HH20)
        #    dist_hh20_bps = (HH20 - Close)/Close * 10000
        df["dist_hh20_bps"] = ((df["HH20"] - df["Close"]) / df["Close"]).replace(
            [np.inf, -np.inf], np.nan
        ) * 10000.0

        # 3) BB width in bps relative to mid
        #    bb_width_bps = (BB_Upper - BB_Lower) / BB_Mid * 10000
        bb_mid_safe = df["BB_Mid"].replace(0, np.nan)
        df["bb_width_bps"] = ((df["BB_Upper"] - df["BB_Lower"]) / bb_mid_safe) * 10000.0

        # 4) BB squeeze flag: current width < 1.1 * rolling-min(width, 20)
        roll_min_width = (
            df["bb_width_bps"].rolling(window=lookback, min_periods=1).min()
        )
        df["bb_squeeze"] = (df["bb_width_bps"] < (roll_min_width * 1.10)).astype(int)

        # 5) EMA20/EMA50 slopes in bps per bar
        ema20 = df["Close"].ewm(span=20, adjust=False).mean()
        ema50 = df["Close"].ewm(span=50, adjust=False).mean()
        df["ema20_slope_bps"] = ema20.pct_change() * 10000.0
        df["ema50_slope_bps"] = ema50.pct_change() * 10000.0

        # 6) adx14 alias (your ADX is Wilder EMA of DX with 'period')
        df["adx14"] = df["ADX"]

        # 7) MACD histogram delta
        df["macd_hist_delta"] = df["MACD_Hist"].diff()

        # 8) VWAP and difference in bps (Close vs VWAP)
        #    Prefer session VWAP if you have Timestamp; else rolling 60 bars VWAP.
        vwap_session = _session_vwap(df, timestamp_col="Timestamp")
        vwap_fallback = _rolling_vwap(df, window=60)
        df["VWAP"] = vwap_session.fillna(vwap_fallback)
        vwap_safe = df["VWAP"].replace(0, np.nan)
        df["vwap_diff_bps"] = ((df["Close"] - vwap_safe) / vwap_safe) * 10000.0

        # 9) ATR% (ATR(14) / Close * 100)
        atr14 = _atr(df, period=14)
        df["ATR14"] = atr14
        df["atr_pct"] = (atr14 / df["Close"].replace(0, np.nan)) * 100.0

        # 10) Volume z-score (rolling 20)
        vol_roll = df["Volume"].rolling(window=20, min_periods=5)
        df["vol_z"] = (df["Volume"] - vol_roll.mean()) / vol_roll.std(ddof=0)

        # (Optional) convenience flags that your LLM can use easily
        df["near_hh20_flag"] = (df["dist_hh20_bps"] >= 0) & (
            df["dist_hh20_bps"] < 10
        )  # within 10 bps of HH20
        df["above_upper_bb_flag"] = (df["Close"] > df["BB_Upper"]).astype(int)
        df["below_lower_bb_flag"] = (df["Close"] < df["BB_Lower"]).astype(int)

        return df

    except Exception as e:
        print(f"[Indicator Error] Failed to compute indicators: {e}")
        raise e


def compute_adx(df: pd.DataFrame, period: int):
    df["UpMove"] = df["High"].diff()
    df["DownMove"] = df["Low"].diff()
    df["+DM"] = np.where(
        (df["UpMove"] > df["DownMove"]) & (df["UpMove"] > 0), df["UpMove"], 0.0
    )
    df["-DM"] = np.where(
        (df["DownMove"] > df["UpMove"]) & (df["DownMove"] > 0), df["DownMove"], 0.0
    )

    df["TR_tmp1"] = (df["High"] - df["Low"]).abs()
    df["TR_tmp2"] = (df["High"] - df["Close"].shift(1)).abs()
    df["TR_tmp3"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = pd.concat([df["TR_tmp1"], df["TR_tmp2"], df["TR_tmp3"]], axis=1).max(
        axis=1
    )

    # Wilder smoothing via EMA(alpha=1/period)
    tr_ema = df["TR"].ewm(alpha=1 / period, adjust=False).mean()
    pdm_ema = df["+DM"].ewm(alpha=1 / period, adjust=False).mean()
    ndm_ema = df["-DM"].ewm(alpha=1 / period, adjust=False).mean()

    df["+DI"] = 100.0 * (pdm_ema / tr_ema.replace(0, np.nan))
    df["-DI"] = 100.0 * (ndm_ema / tr_ema.replace(0, np.nan))
    df["DX"] = 100.0 * (
        abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"]).replace(0, np.nan)
    )
    df["ADX"] = df["DX"].ewm(alpha=1 / period, adjust=False).mean()

    return df


def is_breakout(df, resistance, support, config):
    if len(df) < 5:
        return None, None, None, "Insufficient data"

    current = df.iloc[-1]

    close = current["Close"]
    volume = current["Volume"]
    vol_thresh = config.get("volume_threshold", 0)
    action = config.get("signal", "No Action")

    # Breakout logic
    if close > resistance:
        reason = f"📈 Breakout: Close ({close:.2f}) > Resistance ({resistance:.2f})"

        if volume < vol_thresh:
            reason += " | Low Volume"
        else:
            reason += " | Volume OK"

        if close > current["BB_Upper"]:
            reason += " | BB confirms"
        else:
            reason += " | BB no confirm"

        if current["ADX"] < config["adx"]["threshold"]:
            reason += f" | ADX={round(current['ADX'], 1)} < threshold"
        else:
            reason += f" | ADX={round(current['ADX'], 1)} confirms"

        return "breakout", close, resistance, reason

    # Breakdown logic
    elif close < support:
        reason = f"📉 Breakdown: Close ({close:.2f}) < Support ({support:.2f})"

        if volume < vol_thresh:
            reason += " | Low Volume"
        else:
            reason += " | Volume OK"

        if close < current["BB_Lower"]:
            reason += " | BB confirms"
        else:
            reason += " | BB no confirm"

        if current["ADX"] < config["adx"]["threshold"]:
            reason += f" | ADX={round(current['ADX'], 1)} < threshold"
        else:
            reason += f" | ADX={round(current['ADX'], 1)} confirms"

        return "breakdown", close, support, reason

    return None, None, None, "No breakout/breakdown"
