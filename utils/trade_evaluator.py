import matplotlib.pyplot as plt


def plot_trade_history(trade_data):
    """
    Plots the trade history showing profit over time.

    Args:
        trade_data (list[dict]): List of trade data dictionaries containing 'timestamp' and 'profit'.
    """
    timestamps = [trade["timestamp"] for trade in trade_data]
    profits = [trade["profit"] for trade in trade_data]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, profits, label="Profit Over Time", marker="o")
    plt.xlabel("Time")
    plt.ylabel("Profit")
    plt.title("Trade History")
    plt.legend()
    plt.grid()
    plt.show()


def evaluate_trade_metrics(trades):
    """
    Evaluates trade metrics including success rate and average profit.

    Args:
        trades (list[dict]): List of trade data dictionaries.

    Returns:
        dict: Dictionary containing success rate and average profit.
    """
    successful_trades = [trade for trade in trades if trade.get("success")]
    total_trades = len(trades)

    success_rate = len(successful_trades) / total_trades * 100 if total_trades else 0
    avg_profit = sum(trade.get("profit", 0) for trade in trades) / total_trades if total_trades else 0

    return {"success_rate": success_rate, "avg_profit": avg_profit}
