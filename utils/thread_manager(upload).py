import threading
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Internal Modules
from config import CONFIG
from utils.logger import log_message
from utils.fetch_data import fetch_new_pairs, get_latest_token
from utils.photon_trader import monitor_and_sell_at_profit
from utils.ai_price_predictor import predict_future_price, TRAINED_MODEL, SCALER

# Global Locks & Flags
trading_lock = threading.Lock()
scanning_active = True  
trading_active = False

def setup_driver():
    """Sets up a new Selenium WebDriver with anti-detection."""
    options = Options()
    options.binary_location = CONFIG.get("CHROMIUM_PATH")
    
    # ✅ FIXED: Use relative path for portability (No hardcoded C:\Users)
    user_data_dir = os.path.join(os.getcwd(), "browser_data")
    options.add_argument(f"user-data-dir={user_data_dir}")
    options.add_argument("profile-directory=Default")
    
    options.add_argument("--start-maximized")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--remote-debugging-port=9222")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    try:
        log_message("Starting browser session...")
        driver = webdriver.Chrome(service=Service(CONFIG.get("CHROMEDRIVER_PATH")), options=options)

        # Anti-Bot Detection Patch
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
        })

        log_message("Browser started successfully.")

        # Open second tab for multi-threading
        driver.execute_script("window.open('about:blank', '_blank');")
        time.sleep(1)
        tabs = driver.window_handles

        if len(tabs) < 2:
            log_message("Error: Could not open trading tabs.", level="error")
            return None, None

        # Navigate both tabs
        for tab in tabs:
            driver.switch_to.window(tab)
            driver.get(CONFIG["PHOTON_URL"])

        return driver, tabs

    except Exception as e:
        log_message(f"Failed to start WebDriver: {e}", level="error")
        return None, None

def token_scanner(driver):
    """Scans for new tokens continuously."""
    global trading_active
    log_message("Starting token scanning...")

    while True:
        if trading_active:
            time.sleep(2)
            continue 

        fetch_new_pairs(driver, is_simulation_mode=CONFIG.get("SIMULATION_MODE", False))
        time.sleep(CONFIG.get("REFRESH_INTERVAL", 2))

def trading_executor(driver):
    """Executes trades based on AI signals."""
    global scanning_active
    log_message("Trading Executor Online...")

    while True:
        token_data = get_latest_token()
        if not token_data:
            time.sleep(2)
            continue

        # AI Prediction Logic would go here
        # (Redacted for brevity in GitHub view)
        time.sleep(1)

def start_threads():
    """Starts the scanner and trading threads."""
    driver, tabs = setup_driver()
    if not driver:
        return

    thread1 = threading.Thread(target=token_scanner, args=(driver,), daemon=True)
    thread2 = threading.Thread(target=trading_executor, args=(driver,), daemon=True)

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

if __name__ == "__main__":
    start_threads()