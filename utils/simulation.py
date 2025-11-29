import os
import csv
from utils.logger import log_message

HISTORICAL_DATA_CSV = os.path.join("data", "historical_data.csv")


def simulate_coin_trades(historical_data, stop_loss=0.9, take_profit=1.4):
    """
    Simulates multiple coin trades based on historical data.
    """
    entry_price = None
    successful_trades = 0
    failed_trades = 0
    total_trades = 0
    trade_results = []

    try:
        for data_point in historical_data:
            price = data_point["price"]
            timestamp = data_point.get("timestamp", "")

            if entry_price is None:
                entry_price = price
                stop_loss_price = entry_price * stop_loss
                take_profit_price = entry_price * take_profit

            elif price >= take_profit_price:
                successful_trades += 1
                trade_results.append({
                    "timestamp": timestamp,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "profit": price - entry_price,
                    "result": "success",
                })
                entry_price = None

            elif price <= stop_loss_price:
                failed_trades += 1
                trade_results.append({
                    "timestamp": timestamp,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "profit": price - entry_price,
                    "result": "failure",
                })
                entry_price = None

            total_trades += 1

        success_rate = (successful_trades / total_trades) * 100 if total_trades else 0
        log_message(f"Simulation results: {success_rate:.2f}% success rate.")

        append_to_historical_csv(trade_results)
        return {
            "success_rate": success_rate,
            "total_trades": total_trades,
        }
    except Exception as e:
        log_message(f"Error in simulate_coin_trades: {e}", level="error")
        return {"success_rate": 0, "total_trades": 0}


def append_to_historical_csv(trade_results):
    """
    Appends simulation results to the historical_data.csv file.
    """
    if not trade_results:
        log_message("No trade results to append.", level="warning")
        return

    os.makedirs(os.path.dirname(HISTORICAL_DATA_CSV), exist_ok=True)
    is_new_file = not os.path.isfile(HISTORICAL_DATA_CSV)

    try:
        with open(HISTORICAL_DATA_CSV, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["timestamp", "entry_price", "exit_price", "profit", "result"])
            if is_new_file:
                writer.writeheader()
            writer.writerows(trade_results)

        log_message(f"Appended {len(trade_results)} trade results to {HISTORICAL_DATA_CSV}.")
    except Exception as e:
        log_message(f"Error saving to CSV: {e}", level="error")
