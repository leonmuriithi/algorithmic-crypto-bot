import time
import os

# ✅ FIXED: Use relative paths for portability
CONTROL_FILE_PATH = os.path.join("control", "pause_flag.txt")

def is_bot_paused(control_file=CONTROL_FILE_PATH):
    """Checks if the bot is paused by reading the control file."""
    try:
        with open(control_file, "r", encoding="utf-8") as f:
            return f.read().strip().upper() == "PAUSE"
    except FileNotFoundError:
        return False

def is_bot_stopped(control_file=CONTROL_FILE_PATH):
    """Checks if the bot is stopped by reading the control file."""
    try:
        with open(control_file, "r", encoding="utf-8") as f:
            return f.read().strip().upper() == "STOP"
    except FileNotFoundError:
        return False

def set_bot_state(state, control_file=CONTROL_FILE_PATH):
    """Sets the bot state in the control file."""
    os.makedirs(os.path.dirname(control_file), exist_ok=True)
    with open(control_file, "w", encoding="utf-8") as f:
        f.write(state.upper())

def wait_for_resume(control_file=CONTROL_FILE_PATH, check_interval=1):
    """Waits until the bot is resumed."""
    while is_bot_paused(control_file):
        time.sleep(check_interval)
        if is_bot_stopped(control_file):
            raise Exception("Bot stopped during pause. Exiting...")