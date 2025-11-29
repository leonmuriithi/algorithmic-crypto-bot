import os
from datetime import datetime

def log_audit_event(event):
    """
    Logs an event to the audit log file with a timestamp.

    Args:
        event (str): The event description to log.
    """
    log_dir = "logs"
    log_file_path = os.path.join(log_dir, "audit.log")

    # Ensure the logs directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Format the event with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_event = f"{timestamp} - {event}"

    # Write the event to the log file
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{formatted_event}\n")

def log_trade_action(action, token, amount, price):
    """
    Logs a trade action to the audit log file.

    Args:
        action (str): The trade action (e.g., "BUY", "SELL").
        token (str): The token being traded.
        amount (float): The amount of the token being traded.
        price (float): The price at which the trade was executed.
    """
    event = f"Trade Action: {action} {amount} {token} at ${price}"
    log_audit_event(event)

def log_error(error_message):
    """
    Logs an error message to the audit log file.

    Args:
        error_message (str): The error message to log.
    """
    event = f"Error: {error_message}"
    log_audit_event(event)
