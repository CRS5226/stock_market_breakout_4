import redis
import json
import pandas as pd


# --- Redis connection ---
def get_redis():
    return redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


def fetch_candles(stock_code: str, n: int = 2):
    """Fetch last N candles from Redis (chronological order)."""
    r = get_redis()
    key = f"candles:{stock_code.upper()}"
    items = r.lrange(key, 0, n - 1)  # newest-first
    if not items:
        print(f"[⚠️] No candles found in Redis for {stock_code}")
        return None
    rows = [json.loads(x) for x in items]
    df = pd.DataFrame(rows)
    df["minute"] = pd.to_datetime(df["minute"])
    return df.sort_values("minute").reset_index(drop=True)


def fetch_indicators(stock_code: str, n: int = 2):
    """Fetch last N indicators from Redis (newest first)."""
    r = get_redis()
    key = f"indicators:{stock_code.upper()}"
    items = r.lrange(key, 0, n - 1)
    if not items:
        print(f"[⚠️] No indicators found in Redis for {stock_code}")
        return None
    rows = [json.loads(x) for x in items]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    stock = "GESHIP"

    df_candles = fetch_candles(stock, n=5)
    if df_candles is not None:
        print(f"[📊] Last {len(df_candles)} candles for {stock}:")
        print(df_candles)

    df_indicators = fetch_indicators(stock, n=5)
    if df_indicators is not None:
        print(f"\n[📈] Last {len(df_indicators)} indicator snapshots for {stock}:")
        print(df_indicators)
