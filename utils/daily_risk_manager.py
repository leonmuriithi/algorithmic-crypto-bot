import os
import json
from datetime import datetime
from config import CONFIG

DAILY_RISK_FILE = "logs/daily_risk.json"


def load_daily_risk():
    """
    Loads the daily risk data from the JSON file.

    Returns:
        dict: Dictionary containing the date and loss amount.
    """
    if not os.path.exists(DAILY_RISK_FILE):
        return {"date": datetime.now().strftime("%Y-%m-%d"), "loss": 0}

    with open(DAILY_RISK_FILE, "r") as file:
        return json.load(file)


def save_daily_risk(data):
    """
    Saves the daily risk data to the JSON file.

    Args:
        data (dict): Dictionary containing the date and loss amount.
    """
    with open(DAILY_RISK_FILE, "w") as file:
        json.dump(data, file)


def check_daily_loss():
    """
    Checks if the daily loss limit has been reached.

    Returns:
        bool: True if the daily loss limit has been reached, False otherwise.
    """
    daily_risk = load_daily_risk()
    current_date = datetime.now().strftime("%Y-%m-%d")

    if daily_risk["date"] != current_date:
        daily_risk = {"date": current_date, "loss": 0}
        save_daily_risk(daily_risk)

    return daily_risk["loss"] >= CONFIG["DAILY_LOSS_LIMIT"]


def update_daily_loss(loss_amount):
    """
    Updates the daily loss amount and checks if the daily loss limit has been reached.

    Args:
        loss_amount (float): The amount of loss to add to the daily total.

    Returns:
        bool: True if the daily loss limit has been reached, False otherwise.
    """
    daily_risk = load_daily_risk()
    current_date = datetime.now().strftime("%Y-%m-%d")

    if daily_risk["date"] != current_date:
        daily_risk = {"date": current_date, "loss": 0}

    daily_risk["loss"] += loss_amount
    save_daily_risk(daily_risk)

    return daily_risk["loss"] >= CONFIG["DAILY_LOSS_LIMIT"]
