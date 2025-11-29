def analyze_trade_errors(trades):
    """
    Analyzes errors from unsuccessful trades.

    Args:
        trades (list[dict]): List of trade dictionaries containing error details.

    Returns:
        dict: A summary of error reasons and their occurrences.
    """
    error_reasons = {}
    for trade in trades:
        if not trade["success"]:
            reason = trade.get("failure_reason", "Unknown error")
            error_reasons[reason] = error_reasons.get(reason, 0) + 1
    return error_reasons

def log_error_summary(error_summary, log_file="logs/error_summary.log"):
    """
    Logs the error summary to a log file.

    Args:
        error_summary (dict): A summary of error reasons and their occurrences.
        log_file (str): Path to the log file.
    """
    with open(log_file, "a") as file:
        file.write("Error Summary:\n")
        for reason, count in error_summary.items():
            file.write(f"{reason}: {count}\n")
        file.write("\n")

def generate_error_report(trades, log_file="logs/error_summary.log"):
    """
    Generates an error report from the trades and logs it.

    Args:
        trades (list[dict]): List of trade dictionaries containing error details.
        log_file (str): Path to the log file.
    """
    error_summary = analyze_trade_errors(trades)
    log_error_summary(error_summary, log_file)
