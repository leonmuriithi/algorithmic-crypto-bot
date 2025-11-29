def backtest_strategy(historical_data, strategy):
    """
    Backtests a given trading strategy on historical data.

    Args:
        historical_data (list): List of historical data points.
        strategy (function): Trading strategy function to be tested.

    Returns:
        dict: Dictionary containing backtest results including profit, successful trades, total trades, and success rate.
    """
    results = {"profit": 0, "successful_trades": 0, "total_trades": 0}

    for data_point in historical_data:
        trade_result = strategy(data_point)
        results["profit"] += trade_result["profit"]
        results["successful_trades"] += trade_result["success"]
        results["total_trades"] += 1

    success_rate = (results["successful_trades"] / results["total_trades"]) * 100 if results["total_trades"] else 0
    results["success_rate"] = success_rate
    return results

def simulate_coin_trades(data_point):
    """
    Simulates a coin trade based on a given data point.

    Args:
        data_point (dict): Data point containing information for the trade.

    Returns:
        dict: Dictionary containing the result of the trade including profit and success.
    """
    # Example strategy: Buy if price increased, sell if price decreased
    if data_point["price_change"] > 0:
        profit = data_point["price_change"] * data_point["volume"]
        success = 1
    else:
        profit = 0
        success = 0

    return {"profit": profit, "success": success}
