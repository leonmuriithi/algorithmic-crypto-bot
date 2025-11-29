from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
from utils.logger import log_message
from config import CONFIG
from selenium.common.exceptions import (
    TimeoutException,
    ElementNotInteractableException,
    NoSuchElementException,
)
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
trading_active = False  # ✅ Global flag to pause scanning during trades

### ✅ Navigate to Token Chart
def navigate_to_token_chart(driver, token_card, retries=1):
    """Navigates to the token chart while ensuring correct interaction."""
    for attempt in range(retries):
        try:
            log_message(f"Attempt {attempt+1}/{retries}: Navigating to token chart...")

            # ✅ Locate token link
            token_link = token_card.find_element(By.CLASS_NAME, "kZ551pEiiCmBLd2UhVP_")

            href = token_link.get_attribute("href")
            log_message(f"Clicking the token link: {href}")

            # ✅ Ensure link is clickable and click
            WebDriverWait(driver, 5).until(EC.element_to_be_clickable(token_link))
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", token_link)
            time.sleep(0.5)
            token_link.click()

            # ✅ Wait for chart to load
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "js-trading-view")))

            log_message("✅ Token chart loaded successfully.")
            return True

        except Exception as e:
            log_message(f"❌ Error navigating to token chart: {e}", level="error")
            time.sleep(1)

    log_message("❌ Failed to navigate to token chart after multiple attempts.", level="error")
    return False

### ✅ Closes Any Blocking Popups
def close_blocking_popups(driver):
    """Closes any blocking overlays or pop-ups before trading."""
    try:
        popups = driver.find_elements(By.XPATH, "//button[contains(text(), 'Close')]")
        for popup in popups:
            driver.execute_script("arguments[0].click();", popup)
            log_message("✅ Closed a blocking pop-up.")
            time.sleep(0.5)
    except Exception:
        pass  # No pop-ups detected

### ✅ Fetches Live Token Price
def fetch_token_price(driver):
    """Fetches the current price of a token."""
    try:
        price_text = driver.find_element(By.XPATH, "//div[@data-cable-val='priceUsd']").text.strip()
        log_message(f"🔍 Raw price: {price_text}")

        # ✅ Convert price to float
        return float(price_text.replace(",", "").replace("$", ""))

    except Exception as e:
        log_message(f"❌ Error fetching price: {e}", level="error")
        return 0.0

### ✅ Ensure Bot Always Returns to `PHOTON_URL`
def return_to_main_page(driver):
    """Ensures the bot returns to the main Photon page and continues fetching new tokens."""
    PHOTON_URL = CONFIG["PHOTON_URL"]
    log_message("🔄 Returning to Photon Memescope...")
    driver.get(PHOTON_URL)

    # ✅ Wait for tokens to reload
    time.sleep(4)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "sBVBv2HePq7qYTpGDmRM"))
        )
        log_message("✅ Photon page loaded successfully. Resuming token fetch...")
    except Exception:
        log_message("❌ Photon page took too long to load. Retrying...", level="warning")
        return return_to_main_page(driver)

### **Monitor and Sell at Profit**
def monitor_and_sell_at_profit(driver, token_name, entry_price, predicted_price, profit_target=1.4, stop_loss=0.9):
    """
    Monitors the token price and determines if the predicted price threshold is met.
    If the price reaches the profit target or stop-loss, it waits for the user to manually sell.
    """
    target_price = entry_price * profit_target
    stop_loss_price = entry_price * stop_loss
    log_message(f"📊 Monitoring {token_name}: Target = {target_price:.6f}, Stop-Loss = {stop_loss_price:.6f}, AI Prediction = {predicted_price:.6f}")

    while True:
        try:
            current_price = fetch_token_price(driver)  # ✅ Removed extra token_name parameter (not needed)
            log_message(f"💰 Live price for {token_name}: {current_price:.6f} (AI Predicted: {predicted_price:.6f})")

            # ✅ **Ensure `predicted_price` is used correctly**
            if predicted_price >= entry_price * 1.4:
                log_message(f"✅ AI predicted {token_name} will reach {predicted_price:.6f}. Waiting for refresh...")
                return  # ✅ Do nothing and wait for manual refresh

            # ✅ **Exit if Profit Target is met**
            if current_price >= target_price:
                log_message(f"✅ Profit target reached for {token_name}. Waiting for manual sell...")
                return  # ✅ Stop monitoring, user will manually sell

            # ✅ **Exit if Stop-Loss is triggered**
            elif current_price <= stop_loss_price:
                log_message(f"🚨 Stop-loss triggered for {token_name}. Waiting for manual sell...")
                return  # ✅ Stop monitoring, user will manually sell

            time.sleep(0.5)  # ✅ Check price every 0.5 sec

        except Exception as e:
            log_message(f"❌ Error monitoring {token_name}: {e}", level="error")
            break  # ✅ Exit loop on error
