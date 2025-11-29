from config import CONFIG
from utils.logger import log_message
from datetime import datetime


def calculate_token_age_days(creation_date):
    """
    Calculates the age of a token in days from its creation date.

    Args:
        creation_date (str): The creation date of the token.

    Returns:
        int: Age of the token in days.
    """
    try:
        creation_date = datetime.strptime(creation_date, "%Y-%m-%d")
        current_date = datetime.now()
        return (current_date - creation_date).days
    except Exception as e:
        log_message(f"Error calculating token age: {e}", level="error")
        return 0


def has_organic_volume(token_data):
    """
    Checks if the token has organic volume based on volume to liquidity and volume to holder ratios.

    Args:
        token_data (dict): Dictionary containing token data.

    Returns:
        bool: True if the token has organic volume, False otherwise.
    """
    volume = token_data.get("volume", 0)
    liquidity = token_data.get("liquidity", 0)
    holder_count = token_data.get("holder_count", 0)

    if volume == 0 or liquidity == 0 or holder_count == 0:
        return False

    volume_liquidity_ratio = volume / liquidity
    volume_holder_ratio = volume / holder_count

    if volume_liquidity_ratio > CONFIG["MAX_VOLUME_LIQUIDITY_RATIO"]:
        return False
    if volume_holder_ratio > CONFIG["MAX_VOLUME_HOLDER_RATIO"]:
        return False

    return True


def detect_insider_holding(token_data):
    """
    Detects if the token has excessive insider holding.

    Args:
        token_data (dict): Dictionary containing token data.

    Returns:
        bool: True if the token has excessive insider holding, False otherwise.
    """
    insider_holding = token_data.get("insider_holding", 0)
    return insider_holding > CONFIG["MAX_INSIDER_HOLDING"]


def detect_sniper_activity(token_data):
    """
    Detects if the token has sniper activity.

    Args:
        token_data (dict): Dictionary containing token data.

    Returns:
        bool: True if the token has sniper activity, False otherwise.
    """
    sniper_count = token_data.get("sniper_count", 0)
    return sniper_count > 0 and CONFIG.get("EXCLUDE_SNIPERS", True)


def detect_bot_holders(token_data):
    """
    Detects if the token has excessive bot holders.

    Args:
        token_data (dict): Dictionary containing token data.

    Returns:
        bool: True if the token has excessive bot holders, False otherwise.
    """
    bot_holder_count = token_data.get("bot_holders", 0)
    total_holders = token_data.get("holder_count", 0)

    if total_holders == 0:
        return False

    bot_holder_ratio = bot_holder_count / total_holders
    return bot_holder_ratio > CONFIG["MAX_BOT_HOLDER_RATIO"]


def detect_rug_pull(token_data):
    """
    Detects if the token is a potential rug pull based on various checks.

    Args:
        token_data (dict): Dictionary containing token data.

    Returns:
        bool: True if the token is a potential rug pull, False otherwise.
    """
    if not has_organic_volume(token_data):
        log_message(f"Token {token_data.get('name')} failed organic volume check.")
        return True

    if detect_insider_holding(token_data):
        log_message(f"Token {token_data.get('name')} failed insider holding check.")
        return True

    # New: Skip tokens below a certain age threshold
    token_age_days = calculate_token_age_days(token_data.get("creation_date", ""))
    if token_age_days < CONFIG["MIN_TOKEN_AGE_DAYS"]:
        log_message(f"Token {token_data.get('name')} is too new (Age: {token_age_days} days).")
        return True

    if detect_sniper_activity(token_data):
        log_message(f"Token {token_data.get('name')} detected sniper activity.")
        return True

    if detect_bot_holders(token_data):
        log_message(f"Token {token_data.get('name')} has excessive bot holders.")
        return True

    return False
