"""
🚀 ALGORTHMIC TRADING BOT - MAIN ENTRY POINT
Architecture: Selenium Web Driver + XGBoost AI + Multi-threading
"""

import os
import time
import threading
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import StaleElementReferenceException

# Internal Modules (Architecture Simulation)
from config import CONFIG
from utils.fetch_data import fetch_new_pairs, process_token_card
from utils.ai_price_predictor import train_price_prediction_model_from_csv
from utils.logger import log_message
from utils.control import is_bot_paused, is_bot_stopped, wait_for_resume
from utils.thread_manager import start_threads

def setup_driver():
    """
    Configures the Selenium WebDriver with anti-detection headers
    and connects to the local debugging port for session persistence.
    """
    options = Options()
    # Dynamic path handling for cross-platform compatibility
    options.binary_location = CONFIG.get("CHROMIUM_PATH", "/usr/bin/chromium")
    
    user_data_dir = os.getenv("CHROME_PROFILE", "./user_data")
    options.add_argument(f"user-data-dir={user_data_dir}")
    options.add_argument("profile-directory=Default")
    
    # Anti-Detection Flags
    options.add_argument("--start-maximized")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-infobars")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    try:
        log_message("Starting browser with the specified profile...")
        driver = webdriver.Chrome(
            options=options,
            service=Service(CONFIG.get("CHROMEDRIVER_PATH", "./chromedriver"))
        )
        # Patch navigator.webdriver to avoid bot detection
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
        })
        log_message("Browser started successfully.")
        return driver
    except Exception as e:
        log_message(f"Failed to start WebDriver: {e}", level="error")
        raise

def process_new_pairs(driver, trade_amount):
    """
    Fetches tokens continuously, processes them, and ensures AI trading runs in parallel.
    Uses XGBoost model to predict breakout probabilities.
    """
    log_message("Monitoring new pairs...")

    model, scaler = None, None
    if not CONFIG.get("SIMULATION_MODE", False):
        model, scaler = train_price_prediction_model_from_csv()
        if not model or not scaler:
            log_message("AI model could not be loaded. Exiting process_new_pairs.", level="error")
            return

    while True:
        try:
            if is_bot_stopped():
                log_message("Bot is stopped. Exiting process_new_pairs.")
                break
            if is_bot_paused():
                log_message("Bot is paused. Waiting to resume...")
                wait_for_resume()

            # Refresh token list dynamically to prevent StaleElementReferenceException
            try:
                log_message("Fetching fresh token list...")
                tokens = fetch_new_pairs(driver, is_simulation_mode=CONFIG["SIMULATION_MODE"])

                if not tokens:
                    log_message("No new tokens found. Retrying in 15 seconds...")
                    time.sleep(15)
                    continue
                
                log_message(f"Fetched {len(tokens)} tokens for analysis.")

                for token in tokens:
                    if is_bot_stopped(): break
                    
                    try:
                        log_message(f"Processing token: {token['name']}")
                        process_token_card(driver, token["card"], CONFIG["SIMULATION_MODE"], None, model, scaler)
                    except IndexError:
                        log_message(f"Error processing token card: List index out of range", level="error")
                        continue
                    except Exception as e:
                        log_message(f"Unexpected error: {e}", level="error")
                        continue

            except StaleElementReferenceException:
                log_message("DOM Update detected (Stale Element). Refreshing...", level="warning")
                driver.refresh()
                time.sleep(10)
                continue

        except Exception as e:
            log_message(f"Error in process_new_pairs: {e}", level="error")

def main():
    """
    Main entry point for the HFT Bot.
    """
    log_message("Initializing System...")

    # 1. Train/Load AI Model
    model, scaler = train_price_prediction_model_from_csv()
    if not model:
        log_message("CRITICAL: AI Model failed to load.", level="error")
        return

    # 2. Start Multi-Threaded Execution
    log_message("Starting Thread Manager...")
    start_threads()

if __name__ == "__main__":
    main()
