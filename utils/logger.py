import logging
import os

# ✅ Ensure the logs directory exists
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# ✅ Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, "bot.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_message(message, level="info"):
    """
    Logs a message to the bot's log file and console.

    Args:
        message (str): The message to log.
        level (str): The severity level ('info', 'warning', 'error').
    """
    print(message)
    
    if level == "info":
        logging.info(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "error":
        logging.error(message)

    # ✅ Alert for critical failures
    if "CRITICAL" in message.upper():
        with open(os.path.join(log_dir, "critical_alerts.log"), "a") as f:
            f.write(f"{message}\n")
