import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException, KiteException
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = "config30.json"
SAVE_FOLDER = "historical_data"
INTERVAL = "day"

# ===================== helpers =====================


def _atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR using Wilder smoothing."""
    high_low = (df["high"] - df["low"]).abs()
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def _adx_wilder(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """+DI, -DI, ADX with Wilder smoothing."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    tr_ema = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_ema = (
        pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
    )
    minus_dm_ema = (
        pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
    )

    plus_di = 100.0 * (plus_dm_ema / tr_ema.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm_ema / tr_ema.replace(0, np.nan))

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    out = pd.DataFrame(index=df.index)
    out["+DI"] = plus_di
    out["-DI"] = minus_di
    out["ADX"] = adx
    return out


def _rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Rolling (multi-day) VWAP for daily bars.
    Typical price * Volume rolling sum / Volume rolling sum.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"].clip(lower=0)
    num = pv.rolling(window=window, min_periods=1).sum()
    den = df["volume"].rolling(window=window, min_periods=1).sum().replace(0, np.nan)
    return num / den


# ===================== indicators =====================


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetime and sorting; Kite returns 'date','open','high','low','close','volume'
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --- Moving Averages (your original) ---
    df["MA_Fast"] = df["close"].rolling(window=9, min_periods=1).mean()
    df["MA_Slow"] = df["close"].rolling(window=20, min_periods=1).mean()

    # --- Bollinger Bands (original, period=20, std=2) ---
    bb_period = 20
    bb_mid = df["close"].rolling(window=bb_period, min_periods=1).mean()
    bb_std = df["close"].rolling(window=bb_period, min_periods=1).std(ddof=0)
    df["BB_Mid"] = bb_mid
    df["BB_Upper"] = bb_mid + (bb_std * 2.0)
    df["BB_Lower"] = bb_mid - (bb_std * 2.0)

    # --- MACD (original 12/26/9) ---
    ema_fast = df["close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # --- ADX family (upgraded to Wilder smoothing) ---
    adx_df = _adx_wilder(df, period=14)
    df["+DI"] = adx_df["+DI"]
    df["-DI"] = adx_df["-DI"]
    df["ADX"] = adx_df["ADX"]

    # ======== NEW FIELDS you wanted ========

    # 1) HH20 / LL20 (20-bar highs/lows)
    lookback = 20
    df["HH20"] = df["high"].rolling(window=lookback, min_periods=1).max()
    df["LL20"] = df["low"].rolling(window=lookback, min_periods=1).min()

    # 2) dist_hh20_bps = (HH20 - close)/close * 10000
    df["dist_hh20_bps"] = (
        (df["HH20"] - df["close"]) / df["close"].replace(0, np.nan)
    ) * 10000.0

    # 3) bb_width_bps = (BB_Upper - BB_Lower)/BB_Mid * 10000
    df["bb_width_bps"] = (
        (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"].replace(0, np.nan)
    ) * 10000.0

    # 4) bb_squeeze: current width < 1.10 * rolling-min(width, 20)
    roll_min_width = df["bb_width_bps"].rolling(window=lookback, min_periods=1).min()
    df["bb_squeeze"] = (df["bb_width_bps"] < (roll_min_width * 1.10)).astype(int)

    # 5) EMA20/EMA50 slope in bps per bar
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    df["ema20_slope_bps"] = ema20.pct_change() * 10000.0
    df["ema50_slope_bps"] = ema50.pct_change() * 10000.0

    # 6) adx14 alias
    df["adx14"] = df["ADX"]

    # 7) macd_hist_delta
    df["macd_hist_delta"] = df["MACD_Hist"].diff()

    # 8) VWAP (rolling 20-day) + vwap_diff_bps
    df["VWAP"] = _rolling_vwap(df, window=20)
    df["vwap_diff_bps"] = (
        (df["close"] - df["VWAP"]) / df["VWAP"].replace(0, np.nan)
    ) * 10000.0

    # 9) ATR14 + atr_pct
    df["ATR14"] = _atr_wilder(df, period=14)
    df["atr_pct"] = (df["ATR14"] / df["close"].replace(0, np.nan)) * 100.0

    # 10) vol_z (rolling 20)
    vol_roll = df["volume"].rolling(window=20, min_periods=5)
    df["vol_z"] = (df["volume"] - vol_roll.mean()) / vol_roll.std(ddof=0)

    return df


# ===================== KiteConnect =====================


def make_kite():
    API_KEY = os.getenv("KITE_API_KEY")
    ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")
    if not API_KEY or not ACCESS_TOKEN:
        raise RuntimeError("Missing creds: set KITE_API_KEY + KITE_ACCESS_TOKEN")
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite


def fetch_historical_from_config(config_path: str, interval: str = "day"):
    kite = make_kite()
    today = datetime.today()
    to_date = today - timedelta(days=1)
    from_date = to_date.replace(year=to_date.year - 1)

    print(f"📅 Fetching data from {from_date.date()} to {to_date.date()} [{interval}]")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read {config_path}: {e}")
        return

    os.makedirs(SAVE_FOLDER, exist_ok=True)

    for stock in cfg.get("stocks", []):
        stock_code = stock.get("stock_code")
        token = stock.get("instrument_token")
        if not stock_code or not token:
            print(f"⚠️ Skipping invalid entry: {stock}")
            continue

        print(f"📥 Fetching {stock_code} ({token})...")
        try:
            candles = kite.historical_data(
                instrument_token=token,
                from_date=from_date.strftime("%Y-%m-%d"),
                to_date=to_date.strftime("%Y-%m-%d"),
                interval=interval,
                continuous=False,
                oi=False,
            )
        except TokenException as te:
            print(f"❌ Auth error for {stock_code}: {te}")
            continue
        except KiteException as ke:
            print(f"❌ Kite error for {stock_code}: {ke}")
            continue
        except Exception as e:
            print(f"❌ Unexpected error for {stock_code}: {e}")
            continue

        if not candles:
            print(f"⚠️ No data returned for {stock_code}")
            continue

        df = pd.DataFrame(candles)
        # Ensure expected column names (lowercase) exist
        expected = {"date", "open", "high", "low", "close", "volume"}
        missing = expected - set(df.columns)
        if missing:
            print(f"⚠️ Missing columns for {stock_code}: {missing}")
            continue

        df = calculate_indicators(df)

        df = df.round(4)

        out = os.path.join(
            SAVE_FOLDER,
            f"{stock_code}_historical_{from_date.date()}_to_{to_date.date()}.csv",
        )
        df.to_csv(out, index=False)
        print(f"✅ Saved {len(df)} candles with indicators → {out}")


if __name__ == "__main__":
    fetch_historical_from_config(CONFIG_PATH, INTERVAL)
