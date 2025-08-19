# server.py

import os
import time
import json
import pandas as pd
from multiprocessing import Process, Manager
from indicator import is_breakout, add_indicators
from collector import start_collector
from telegram_alert import (
    send_trade_alert,
    send_pipeline_status,
    send_error_alert,
    send_config_update,
)
from llm_forecast import forecast_config_update, route_model
from basic_algo_forecaster import basic_forecast_update
from throughput_monitor import ThroughputMonitor
from redis_utils import get_redis, get_recent_candles

r = get_redis()

CONFIG_PATH = "config400.json"
DATA_FOLDER = "data"
STATS_FILE = "monitor_stats.json"

os.makedirs(DATA_FOLDER, exist_ok=True)

BREAKOUT_STATE = {}
LAST_CONFIG = {}
last_forecast_time = {}


def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Config Error] Failed to read config: {e}")
        return {"stocks": []}


def fetch_latest_config_for_stock(stock_code):
    config = load_config()
    for stock in config.get("stocks", []):
        if stock["stock_code"] == stock_code:
            return stock
    return None


def print_config_changes(stock_code, new_config):
    last_config = LAST_CONFIG.get(stock_code, {})
    messages = []
    support_changed = False
    resistance_changed = False

    for key in new_config:
        new_val = new_config.get(key)
        old_val = last_config.get(key)

        if isinstance(new_val, dict):
            for sub_key in new_val:
                old_sub = old_val.get(sub_key) if old_val else None
                new_sub = new_val[sub_key]
                if old_sub != new_sub:
                    messages.append(f"{key}.{sub_key}: {old_sub} → {new_sub}")
        else:
            if old_val != new_val:
                messages.append(f"{key}: {old_val} → {new_val}")
                if key == "support":
                    support_changed = True
                if key == "resistance":
                    resistance_changed = True

    if messages:
        print(f"[🔁 CONFIG CHANGE] {stock_code} → " + ", ".join(messages))
        # if not last_config:
        #     send_config_update(
        #         f"🆕 Stock Added: {stock_code}\n" + "\n".join(messages), stock_code
        #     )
        # else:
        #     send_config_update(
        #         f"⚙️ Config Updated: {stock_code}\n" + "\n".join(messages), stock_code
        #     )

        # ✅ Update LAST_CONFIG immediately so it won't send duplicates
        LAST_CONFIG[stock_code] = new_config.copy()

    return support_changed or resistance_changed


def save_indicators(df, stock_code, max_rows=100):
    os.makedirs("stock_indicators", exist_ok=True)

    # --- Core columns (same as before) ---
    cols_to_save = [
        "Timestamp",
        "Close",
        "High",
        "Low",
        "Volume",
        "MA_Fast",
        "MA_Slow",
        "BB_Mid",
        "BB_Upper",
        "BB_Lower",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "ADX",
        "+DI",
        "-DI",
    ]

    # --- Extended features from add_indicators ---
    extended_cols = [
        "HH20",
        "LL20",
        "dist_hh20_bps",
        "bb_width_bps",
        "bb_squeeze",
        "ema20_slope_bps",
        "ema50_slope_bps",
        "adx14",
        "macd_hist_delta",
        "VWAP",
        "vwap_diff_bps",
        "ATR14",
        "atr_pct",
        "vol_z",
        "near_hh20_flag",
        "above_upper_bb_flag",
        "below_lower_bb_flag",
    ]

    # Keep only what exists in df (so it doesn’t error if missing)
    all_cols = [c for c in cols_to_save + extended_cols if c in df.columns]

    df_to_save = df[all_cols].copy()
    df_to_save = df_to_save.tail(max_rows)

    file_path = os.path.join("stock_indicators", f"{stock_code}_indicators.csv")
    df_to_save.to_csv(file_path, index=False)


def detect_removed_stocks(existing_codes):
    removed = []
    for stock_code in list(LAST_CONFIG.keys()):
        if stock_code not in existing_codes:
            removed.append(stock_code)
            send_pipeline_status(f"❌ Stock Removed: {stock_code}", stock_code)
            del LAST_CONFIG[stock_code]
            del BREAKOUT_STATE[stock_code]
    return removed


def forecast_manager():
    print("🧠 Forecast Manager started.")

    monitor = ThroughputMonitor(window_sec=60, csv_path="forecast/throughput.csv")
    last_print = 0.0
    last_csv = 0.0

    while True:
        try:
            config_data = load_config()
            stocks = config_data.get("stocks", [])

            for i, stock_cfg in enumerate(stocks):
                stock_code = stock_cfg["stock_code"]

                # --- Redis + historical lookup handled in forecast_config_update ---
                model_choice = route_model(stock_code, i)

                t0 = time.time()
                print(f"[🔮 Forecast] {stock_code} via {model_choice} ...")
                updated_cfg, reasons, err = forecast_config_update(
                    stock_cfg,
                    historical_folder="historical_data",  # ✅ only pass folder now
                    model=model_choice,
                    temperature=0.2,
                    escalate_on_signal=True,
                )
                latency_ms = (time.time() - t0) * 1000.0
                monitor.record(stock_code, model_choice, latency_ms)

                if err:
                    print(f"[❌ LLM Forecast Error - {stock_code}] {err}")
                else:
                    if updated_cfg != stock_cfg:
                        stocks[i] = updated_cfg
                        with open(CONFIG_PATH, "w") as f:
                            json.dump(config_data, f, indent=2)
                        print(f"[🧠 LLM Forecast - {stock_code}] Updated config")
                        if reasons:
                            print(f"[ℹ️ Reasons] {reasons}")
                    else:
                        print(f"[🧠 LLM Forecast - {stock_code}] No changes detected")

                # pacing delay (tune for RPM/TPM)
                time.sleep(5)

                # --- dashboard every 10s
                now = time.time()
                if now - last_print >= 10:
                    snap = monitor.snapshot()
                    # print(
                    #     f"[⏱ Last {snap['window_sec']}s] "
                    #     f"Unique stocks: {snap['unique_symbols_last_window']} "
                    #     f"({snap['pct_of_400']}% of 400) | "
                    #     f"Events: {snap['events_last_window']} | "
                    #     f"Avg {snap['avg_latency_ms']} ms | "
                    #     f"P50 {snap['p50_latency_ms']} ms | "
                    #     f"P90 {snap['p90_latency_ms']} ms | "
                    #     f"P99 {snap['p99_latency_ms']} ms | "
                    #     f"Per-model {snap['per_model_counts']}"
                    # )
                    last_print = now

                # --- CSV snapshot every 60s
                if now - last_csv >= 60:
                    monitor.log_snapshot_csv()
                    last_csv = now

        except Exception as e:
            # send_error_alert(f"[Forecast Manager Error] {type(e).__name__}: {e}")
            time.sleep(30)


def monitor_shard(stock_list, stats, shard_id, total_shards, lookback_candles=200):
    # Divide work among shards
    stocks_for_this_shard = [
        stock for i, stock in enumerate(stock_list) if i % total_shards == shard_id
    ]
    print(f"🟢 Monitoring Shard {shard_id} with {len(stocks_for_this_shard)} stocks")

    while True:
        start_time = time.time()

        for stock in stocks_for_this_shard:
            code = stock["stock_code"]

            try:
                # ✅ Always refresh config for latest support/resistance/params
                updated_config = fetch_latest_config_for_stock(code)
                if updated_config:
                    stock = updated_config

                    # Detect config changes → reset breakout state
                    config_changed = print_config_changes(code, updated_config)
                    if config_changed:
                        BREAKOUT_STATE[code] = {
                            "above_resistance": False,
                            "below_support": False,
                        }

                # ✅ Fetch last N candles
                rows = get_recent_candles(r, code, n=lookback_candles)
                if not rows or len(rows) < 10:
                    continue

                # Chronological order
                df = pd.DataFrame(reversed(rows))
                df["Timestamp"] = pd.to_datetime(df["minute"])
                df.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                        "minute": "Timestamp",
                    },
                    inplace=True,
                )

                # ✅ Add indicators
                df = add_indicators(df, stock)

                # ✅ Save latest candle+indicators
                latest_row = df.iloc[-1].to_dict()
                latest_row["Timestamp"] = str(latest_row["Timestamp"])
                r.lpush(f"indicators:{code}", json.dumps(latest_row))
                r.ltrim(f"indicators:{code}", 0, 200)
                stats["redis_writes"] = stats.get("redis_writes", 0) + 1

                # ✅ Breakout check
                signal, price, levels, reason = is_breakout(
                    df, stock.get("resistance", 0), stock.get("support", 0), stock
                )

                state = BREAKOUT_STATE.setdefault(
                    code, {"above_resistance": False, "below_support": False}
                )
                if signal == "breakout" and not state["above_resistance"]:
                    send_trade_alert(
                        code,
                        f"📈 Breakout Above {stock.get('resistance', 0)}\n🧠 {reason}",
                        price,
                        df["Timestamp"].iloc[-1],
                    )
                    state["above_resistance"], state["below_support"] = True, False

                elif signal == "breakdown" and not state["below_support"]:
                    send_trade_alert(
                        code,
                        f"📉 Breakdown Below {stock.get('support', 0)}\n🧠 {reason}",
                        price,
                        df["Timestamp"].iloc[-1],
                    )
                    state["below_support"], state["above_resistance"] = True, False

            except Exception as e:
                print(e)
                # send_error_alert(f"[{code}] Monitor Error: {type(e).__name__}: {e}")

        # ✅ Update monitoring stats
        stats["monitor_cycles"] += 1
        stats["monitor_time"] += time.time() - start_time
        time.sleep(2)


def stats_writer(stats):
    start_time = time.time()
    while True:
        try:
            elapsed = time.time() - start_time
            per_sec = {
                "csv_writes_per_sec": stats["csv_writes"] / elapsed if elapsed else 0,
                "monitor_cycles_per_sec": (
                    stats["monitor_cycles"] / elapsed if elapsed else 0
                ),
                "ticks_per_sec": stats["tick_count"] / elapsed if elapsed else 0,
            }
            full_stats = dict(stats)
            full_stats.update(per_sec)
            with open(STATS_FILE, "w") as f:
                json.dump(full_stats, f, indent=2)
        except Exception as e:
            print(f"[Stats Error] {e}")
        time.sleep(15)  # 15 seconds interval


def run():
    processes = {}
    manager = Manager()
    stats = manager.dict(
        {"csv_writes": 0, "monitor_cycles": 0, "monitor_time": 0.0, "tick_count": 0}
    )

    stat_proc = Process(target=stats_writer, args=(stats,))
    stat_proc.start()

    print("🚀 Real-Time Stock Monitor started. Watching for changes...")

    NUM_SHARDS = 2  # collectors
    MONITOR_SHARDS = 8  # monitors

    while True:
        try:
            config = load_config()
            stock_list = config.get("stocks", [])

            # Collectors
            for i in range(NUM_SHARDS):
                proc_name = f"collector_{i}"
                if proc_name not in processes:
                    collector_proc = Process(
                        target=start_collector, args=(stock_list, stats, i, NUM_SHARDS)
                    )
                    collector_proc.start()
                    processes[proc_name] = collector_proc

            # Forecaster
            if "forecaster" not in processes:
                forecast_proc = Process(target=forecast_manager)
                forecast_proc.start()
                processes["forecaster"] = forecast_proc

            # Monitors (sharded)
            for i in range(MONITOR_SHARDS):
                proc_name = f"monitor_{i}"
                if proc_name not in processes:
                    monitor_proc = Process(
                        target=monitor_shard,
                        args=(stock_list, stats, i, MONITOR_SHARDS),
                    )
                    monitor_proc.start()
                    processes[proc_name] = monitor_proc
                    print(f"✅ Started Monitor Shard {i}")

            time.sleep(5)

        except KeyboardInterrupt:
            print("⛔️ Stopped by user.")
            break
        except Exception as e:
            send_error_alert(f"[Server Error] {type(e).__name__}: {e}")
            time.sleep(5)


if __name__ == "__main__":
    run()
