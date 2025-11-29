import cloudscraper
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import os
from utils.logger import log_message
from utils.ai_price_predictor import train_price_prediction_model_from_csv, predict_future_price
from config import CONFIG
import threading
import time 
import csv
import numpy as np
import pandas as pd
from utils.control import is_bot_paused, is_bot_stopped,  wait_for_resume
from utils.simulation import simulate_coin_trades 
import datetime
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from selenium.common.exceptions import StaleElementReferenceException  # Import for handling stale elements
import traceback 
from utils.photon_trader import navigate_to_token_chart
from datetime import datetime
import itertools


def bypass_cloudflare(driver, url, retries=3, wait_time=10):
    """
    Detects and attempts to bypass Cloudflare protection by interacting with the 'Verify you are human' checkbox.

    Args:
        driver: Selenium WebDriver instance.
        url: Target URL for bypass.
        retries (int): Maximum number of retries.
        wait_time (int): Time to wait (seconds) between retries.

    Returns:
        bool: True if successful or if Cloudflare is not detected, False otherwise.
    """
    for attempt in range(1, retries + 1):
        try:
            driver.get(url)
            log_message(f"Attempt {attempt}/{retries}: Checking for Cloudflare challenge...")

            # Wait for a few seconds to let the Cloudflare challenge load
            time.sleep(5)  

            # Check for the Cloudflare "Verify you are human" checkbox
            try:
                checkbox = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//input[@type='checkbox']"))
                )
                log_message("Cloudflare challenge detected. Attempting to bypass.")

                # Click the checkbox if detected
                driver.execute_script("arguments[0].click();", checkbox)
                log_message("Clicked 'Verify you are human' checkbox.")

                # Wait to see if the page redirects successfully
                WebDriverWait(driver, 15).until(EC.url_changes(url))
                log_message("Successfully bypassed Cloudflare.")
                return True

            except TimeoutException:
                log_message("No Cloudflare challenge detected. Proceeding normally.")
                return True  # No Cloudflare detected, proceed normally.

        except (NoSuchElementException, ElementClickInterceptedException) as e:
            log_message(f"Error interacting with Cloudflare checkbox: {e}. Retrying in {wait_time} seconds.")
            time.sleep(wait_time)

    log_message("All attempts to bypass Cloudflare failed.", level="error")
    return False


def convert_to_float(value):
    """
    Converts a string with suffixes (e.g., K, M, B) to a float.

    Args:
        value (str): The string value to be converted.

    Returns:
        float: Converted float value, or 0.0 if conversion fails.
    """
    try:
        # Ensure value is a valid string
        if not value or not isinstance(value, str):
            log_message(f"Invalid value for conversion: {value}. Defaulting to 0.0.", level="warning")
            return 0.0

        # Remove dollar signs and commas
        value = value.strip().replace("$", "").replace(",", "")
        multiplier = 1

        # Handle suffixes
        if "K" in value.upper():
            multiplier = 1_000
            value = value.upper().replace("K", "")
        elif "M" in value.upper():
            multiplier = 1_000_000
            value = value.upper().replace("M", "")
        elif "B" in value.upper():
            multiplier = 1_000_000_000
            value = value.upper().replace("B", "")

        # Convert to float
        return float(value) * multiplier

    except ValueError as e:
        log_message(f"Error converting value: '{value}' to float. {e}. Defaulting to 0.0.", level="error")
        return 0.0


def update_historical_data(csv_file, driver):
    """
    Updates the historical data CSV file with entry prices for each token.
    Ensures 'entry_price' is correctly added without duplicate values.
    """
    if not os.path.exists(csv_file):
        log_message(f"CSV file {csv_file} not found. Skipping update.", level="error")
        return False

    try:
        # Read existing CSV data
        with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)  # Convert to list for modification
            fieldnames = reader.fieldnames or []

        # Ensure 'entry_price' is in the fieldnames
        if "entry_price" not in fieldnames:
            fieldnames.append("entry_price")

        # Update entry prices
        for row in rows:
            token_name = row.get("token_name")
            if not token_name:
                continue  # Skip malformed rows

            # Fetch price if not already present
            if "entry_price" not in row or not row["entry_price"]:
                entry_price = fetch_token_price(driver, token_name)
                row["entry_price"] = entry_price if entry_price else "N/A"

        # Write back updated data
        with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        log_message(f"Updated historical data in {csv_file} with entry prices.")
        return True  # Update successful

    except Exception as e:
        log_message(f"Error updating historical data: {e}", level="error")
        return False


processed_contracts = set()
token_index = itertools.cycle([0, 1, 2, 3])  # Fetch top 4 tokens

class PerformanceTracker:
    def __init__(self):
        self.total_trades = 0
        self.successful_trades = 0
        self.start_time = time.time()

    def update_trade_count(self, success):
        """Tracks trades and success rate dynamically."""
        self.total_trades += 1
        if success:
            self.successful_trades += 1

        # ✅ Check if 75% success rate is maintained every hour
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 3600:
            success_rate = self.successful_trades / max(1, self.total_trades)
            if success_rate < 0.75:
                log_message(f"🚨 Success Rate Dropped to {success_rate:.2%}! Adjusting AI threshold...")
                adjust_threshold()
            self.reset_tracker()

    def reset_tracker(self):
        """Resets tracker every hour."""
        self.total_trades = 0
        self.successful_trades = 0
        self.start_time = time.time()

performance_tracker = PerformanceTracker()  # Global instance

def adjust_threshold():
    """Dynamically lowers threshold if success rate drops."""
    global OPTIMAL_THRESHOLD
    if OPTIMAL_THRESHOLD > 0.35:  # Lower but never below 0.35
        OPTIMAL_THRESHOLD -= 0.02
        log_message(f"🔄 New AI threshold: {OPTIMAL_THRESHOLD:.2f}")

def fetch_new_pairs(driver, is_simulation_mode=False, min_age_minutes=5):
    """
    Fetches fresh token data and alternates between processing the top 4 tokens.
    Ensures at least 60 trades per session while maintaining a 75% success rate.
    """
    url = CONFIG["PHOTON_URL"]

    log_message("🚀 Training AI model if necessary...")
    model, scaler = train_price_prediction_model_from_csv()

    while True:
        try:
            driver.get(url)
            log_message("🔍 Fetching fresh tokens from Photon Memescope...")

            # ✅ Scroll to load fresh tokens
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(4)

            # ✅ Fetch tokens
            try:
                token_cards = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "sBVBv2HePq7qYTpGDmRM"))
                )
                token_cards = [card for card in token_cards if card.is_displayed()]
            except Exception:
                log_message("⚠️ No fresh token cards found. Retrying...", level="warning")
                time.sleep(4)
                continue  

            if not token_cards or len(token_cards) < 4:
                log_message("⚠️ Not enough fresh token cards found. Retrying...", level="warning")
                time.sleep(4)
                continue

            # ✅ Pick token based on alternating index
            index = next(token_index)
            card = token_cards[index]
            log_message(f"📌 Processing token at index {index}.")

            # ✅ **Check if the token is below the threshold**
            prediction_below_threshold = process_token_card(driver, card, is_simulation_mode, min_age_minutes, model, scaler, performance_tracker)

            if prediction_below_threshold:
                log_message("🔄 Token prediction was below threshold. Fetching a new token...")
                continue  # ✅ **Only return to Photon for a new token if prediction is too low**

            log_message("✅ Token met criteria. **Waiting for manual refresh before fetching another token.**")
            return  # ✅ **Exits and waits for manual refresh before continuing**

        except Exception as e:
            log_message(f"⚠️ Error fetching live tokens: {e}", level="error")
            time.sleep(4)

MINIMUM_TRADES = 60  # Ensures at least 60 trades per session
OPTIMAL_THRESHOLD = 0.47  # Default value (Will be updated dynamically)

def process_token_card(driver, card, is_simulation_mode, min_age_minutes, model, scaler, performance_tracker):
    """
    Processes a single token card, extracts data, navigates to the chart, 
    performs AI predictions, verifies live data, and logs real trades.
    """
    global OPTIMAL_THRESHOLD

    try:
        log_message("🔄 Processing a token...")

        # ✅ Extract token name and contract address
        try:
            name_element = WebDriverWait(card, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "siDxb5Gcy0nyxGjDtRQj"))
            )
            name = name_element.text.strip()

            token_link = WebDriverWait(card, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "kZ551pEiiCmBLd2UhVP_"))
            )
            href = token_link.get_attribute("href")
            if not href:
                log_message(f"⚠️ Skipping {name}: No token link found.", level="warning")
                return

            contract_address = href.split("/")[-1].split("?")[0]
            log_message(f"📌 Token: {name}, Contract: {contract_address}")

            # ✅ Skip if already processed
            if contract_address in processed_contracts:
                log_message(f"⏩ Skipping {name}: Already processed.")
                return
            processed_contracts.add(contract_address)

        except Exception as e:
            log_message(f"⚠️ Failed to extract name or contract: {e}. Skipping...", level="warning")
            return

        # ✅ Extract Market Data BEFORE Navigating
        try:
            volume_text = WebDriverWait(card, 5).until(
                EC.presence_of_element_located((By.XPATH, ".//span[@data-tooltip-content='Volume']/following-sibling::span"))
            ).text
            market_cap_text = card.find_element(By.XPATH, ".//span[@data-tooltip-content='Mkt Cap']/following-sibling::span").text
            creation_date = card.find_element(By.CLASS_NAME, "PexxssXyjdhtFKu0KhLw").text.strip()

            volume = convert_to_float(volume_text)
            market_cap = convert_to_float(market_cap_text)
            token_age = extract_token_age(creation_date)

            log_message(f"📊 Token Data - Name: {name}, Volume: {volume}, Market Cap: {market_cap}, Age: {token_age} mins")

            # ✅ FILTER: Skip tokens that are too old
            if token_age > min_age_minutes:
                log_message(f"❌ Skipping {name}: Token too old ({token_age} mins).")
                return

        except Exception as e:
            log_message(f"⚠️ Error extracting market data: {e}", level="warning")
            return

        # ✅ Navigate to Token Chart
        if not navigate_to_token_chart(driver, card):
            log_message(f"⏩ Skipping {name}: Navigation failed.")
            return

        # ✅ Fetch Live Data AFTER Navigating to Chart
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "c-chart-box__canvas"))
            )

            price_text = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[@data-cable-val='priceUsd']"))
            ).get_attribute("data-value")

            liquidity_text = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[@data-cable-val='usdLiquidity']"))
            ).get_attribute("data-value")

            price = convert_to_float(price_text)
            liquidity = convert_to_float(liquidity_text)

            log_message(f"📈 Live Data - Name: {name}, Price: {price}, Liquidity: {liquidity}")

        except Exception as e:
            log_message(f"⏩ Skipping {name}: Live data error - {e}", level="warning")
            return

        # ✅ **Simulation Mode Handling**
        if is_simulation_mode:
            log_message(f"🛠️ Simulation mode active. Simulating trade for {name}.")
            simulate_token_data(driver, href, name, price, liquidity, market_cap, volume)
            return

        # ✅ **AI Prediction Mode**
        try:
            token_data = {
                "entry_price": price,  
                "price": price,
                "volume": volume,
                "market_cap": market_cap,
                "liquidity": liquidity
            }

            prediction_prob = predict_future_price(token_data)

            if prediction_prob is None:
                log_message(f"⚠️ Skipping {name}: AI prediction failed.", level="warning")
                return

            if prediction_prob >= OPTIMAL_THRESHOLD:
                log_message(f"✅ {name} meets the threshold (≥ {OPTIMAL_THRESHOLD:.2f}). Monitoring price movement...")

                entry_price = price
                target_price = round(entry_price * 1.4, 8)
                stop_loss_price = round(entry_price * 0.8, 8)
                start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                log_message(f"✅ Target Price: {target_price:.8f}, Stop-Loss: {stop_loss_price:.8f}")

                # ✅ Track the original URL before price monitoring
                current_url = driver.current_url

                while True:
                    time.sleep(1)

                    if driver.current_url != current_url:
                        break  # ✅ Exit loop if manual refresh happens

                    try:
                        price_text = WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((By.XPATH, "//div[@data-cable-val='priceUsd']"))
                        ).get_attribute("data-value")
                        current_price = convert_to_float(price_text)
                        log_message(f"🔍 Updated price for {name}: {current_price}")

                    except Exception as e:
                        log_message(f"⚠️ Error fetching updated price: {e}. Retrying...", level="warning")
                        continue

                    if current_price >= target_price:
                        trade_result = {"success": True, "final_price": current_price}
                        log_message(f"🎯 SUCCESS: {name} reached 1.4x! ✅")
                        performance_tracker.update_trade_count(success=True)  # ✅ Track success
                        break

                    elif current_price <= stop_loss_price:
                        trade_result = {"success": False, "final_price": current_price}
                        log_message(f"🚨 FAILURE: {name} hit stop-loss ❌")
                        performance_tracker.update_trade_count(success=False)  # ✅ Track failure
                        break

                # ✅ Save trade before exiting
                trade_result.update({
                    "token_name": name,
                    "entry_price": entry_price,
                    "target_price": target_price,
                    "liquidity": liquidity,
                    "start_time": start_time,
                    "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "profit": (trade_result["final_price"] - entry_price) / entry_price,
                    "market_cap": market_cap,
                    "volume": volume
                })
                save_simulation_result(trade_result)
                # ✅ **ENSURES BOT STAYS ON THE PAGE & WAITS**
                return False  # ✅ **Prevents auto-returning to Photon**

            log_message(f"⏩ Skipping {name}: Probability {prediction_prob:.2f} below threshold.")
            return True # ✅ **Does NOT go to PHOTON when skipping!**

        except Exception as e:
            log_message(f"⚠️ AI Prediction Error: {e}. Skipping {name}.", level="error")
            return True
        

    except Exception as e:
        log_message(f"🚨 Critical Error in process_token_card: {e}. Skipping...", level="error")
        return True


def return_to_main_page(driver):
    """
    Returns to Photon but **does NOT automatically fetch tokens**.
    Only called when manually triggered.
    """
    log_message("Returning to Photon Memescope... (MANUAL ACTION REQUIRED)")

    driver.get(CONFIG["PHOTON_URL"])
    time.sleep(4)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "sBVBv2HePq7qYTpGDmRM"))
        )
        log_message("Photon page loaded successfully.")
    except Exception:
        log_message("Photon page took too long to load. Retrying...", level="warning")
        return return_to_main_page(driver)  # ✅ Still retries if needed

    log_message("✅ **Waiting for manual refresh to fetch new tokens...**")


latest_token = None
lock = threading.Lock()


def update_latest_token(token_data):
    """Updates the latest token detected by the scanner."""
    global latest_token
    with lock:
        latest_token = token_data


def get_latest_token():
    """Retrieves the latest detected token for trading."""
    with lock:
        return latest_token


def extract_token_age(creation_date):
    """
    Converts the token's creation time from a string into minutes.
    
    Args:
        creation_date (str): A string like '5s', '2m', '30m', '1h 15m', etc.

    Returns:
        int: Age of token in minutes.
    """
    try:
        log_message(f"Raw creation date string: {creation_date}")

        if "s" in creation_date:  # ✅ **NEW: Handle seconds**
            seconds = int(creation_date.replace("s", "").strip())
            return max(1, seconds // 60)  # Convert to minutes (minimum 1 min)

        if "h" in creation_date and "m" in creation_date:
            hours, minutes = creation_date.split("h")
            hours = int(hours.strip()) * 60
            minutes = int(minutes.replace("m", "").strip())
            return hours + minutes
        elif "h" in creation_date:
            return int(creation_date.replace("h", "").strip()) * 60
        elif "m" in creation_date:
            return int(creation_date.replace("m", "").strip())

        log_message(f"Unknown format for token age: {creation_date}. Defaulting to 999 mins.", level="warning")
        return 999  # Default to 999 mins if format is unknown

    except Exception as e:
        log_message(f"Error parsing token age: {e}", level="error")
        return 999  # If parsing fails, default to 999 minutes

def simulate_token_data(driver, href, name, entry_price, liquidity, market_cap, volume):
    """
    Simulates a trade in the same tab by monitoring price changes and saving results to CSV.
    """
    try:
        target_price = entry_price * 1.4 
        log_message(f"Simulation started for {name}: Entry Price: {entry_price}, Target Price: {target_price}")

        start_time = datetime.now()
        max_duration = 120  # 120 seconds

        while (datetime.now() - start_time).seconds < max_duration:
            try:
                current_price_text = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//div[@data-cable-val='priceUsd']"))
                ).get_attribute("data-value")
                current_price = convert_to_float(current_price_text)

                log_message(f"Monitoring {name}: Current Price: {current_price}, Target: {target_price}")

                if current_price >= target_price:
                    log_message(f"Simulation successful for {name}: Target reached at {current_price}")
                    success = True
                    break

            except NoSuchElementException:
                log_message(f"Failed to fetch live price for {name}, retrying...", level="warning")

            time.sleep(2)

        else:
            log_message(f"Simulation timed out for {name}. Exiting trade.")
            success = False
            current_price = entry_price

        # ✅ Save results
        save_simulation_result({
            "token_name": name,
            "entry_price": entry_price,
            "target_price": target_price,
            "liquidity": liquidity,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "final_price": current_price,
            "success": success,
            "profit": current_price - entry_price,
            "market_cap": market_cap,
            "volume": volume,
        })

        # ✅ Return to main page
        driver.get(CONFIG["PHOTON_URL"])
        log_message("Returned to main page after simulation.")

    except Exception as e:
        log_message(f"Error during simulation for {name}: {e}", level="error")

SIMULATION_CSV_PATH = "data/historical_data.csv"  # Path to saved simulations

def is_token_already_simulated(token_name):
    """Checks if the token was already simulated by looking in the saved CSV file."""
    if not os.path.exists(SIMULATION_CSV_PATH):
        return False  # No file exists yet, so no tokens are simulated

    try:
        df = pd.read_csv(SIMULATION_CSV_PATH)
        return token_name in df["token_name"].values  # Check if token exists in CSV
    except Exception as e:
        log_message(f"Error reading simulation CSV: {e}", level="error")
        return False

def fetch_token_price(driver, name="UNKNOWN", retries=3):
    """
    Fetches the current price of a token using the same method as process_token_card().
    Ensures price is retrieved successfully with retries if necessary.
    """
    for attempt in range(retries):
        try:
            price_text = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[@data-cable-val='priceUsd']"))
            ).get_attribute("data-value")

            log_message(f"🔍 Attempt {attempt+1}: Raw price for {name}: {price_text}")

            # ✅ Clean the price (keep only numbers and '.')
            cleaned_price = ''.join(c for c in price_text if c.isdigit() or c == '.')

            if not cleaned_price:
                raise ValueError(f"Invalid price format: {price_text}")

            price = float(cleaned_price)
            log_message(f"✅ Fetched price for {name}: {price}")
            return price  # ✅ Return successfully fetched price

        except Exception as e:
            log_message(f"❌ Attempt {attempt+1} failed to fetch price for {name}: {e}", level="warning")
            time.sleep(1)  # ✅ Wait before retrying

    log_message(f"❌ Failed to fetch price for {name} after {retries} attempts.", level="error")
    return 0.0  # ✅ Return 0 if all retries fail


def fetch_historical_tokens(min_age_days):
    """
    Fetches historical tokens from the CSV file for AI training.

    Args:
        min_age_days (float): Minimum age of tokens to include.

    Returns:
        list[dict]: List of historical tokens with details.
    """
    csv_file = os.path.join("data", "historical_data.csv")

    try:
        if not os.path.exists(csv_file):
            log_message(f"No historical data file found at {csv_file}. Returning empty list.")
            return []

        # Read the CSV file
        data = pd.read_csv(csv_file)

        if data.empty:
            log_message(f"Historical data file is empty. Returning empty list.")
            return []

        # Verify required columns exist
        required_columns = [
            "token_name", "entry_price", "target_price", "liquidity",
            "start_time", "end_time", "final_price", "success", "profit"
        ]
        for col in required_columns:
            if col not in data.columns:
                log_message(f"Historical data file missing required column: {col}", level="error")
                return []

        # Convert start_time to datetime for filtering
        data["start_time"] = pd.to_datetime(data["start_time"], errors="coerce")

        # Filter by minimum age
        current_date = pd.Timestamp.now()
        data["age_days"] = (current_date - data["start_time"]).dt.days
        filtered_data = data[data["age_days"] >= min_age_days]

        log_message(f"Fetched {len(filtered_data)} historical tokens for analysis.")
        return filtered_data.to_dict(orient="records")

    except Exception as e:
        log_message(f"Error fetching historical tokens: {e}", level="error")
        return []


def fetch_token_history(driver, token_card, min_age_days=30):
    """
    Fetches historical data for a specific token by clicking on its card.

    Args:
        driver: Selenium WebDriver instance.
        token_card (WebElement): WebElement representing the token card.
        min_age_days (int): Minimum age of the token (in days) for AI training.

    Returns:
        list[dict]: Historical price data for the token or an empty list if the token is too new.
    """
    try:
        # Click the token card to load details
        token_card.click()
        log_message("Clicked on token card to fetch historical data.")

        # Wait for detailed view to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "p-show__widget"))
        )
        log_message("Detailed view loaded successfully for token.")

        # Extract and process historical data
        historical_data = []
        token_details = driver.find_elements(By.CLASS_NAME, "token-details-class")  # Replace with actual class name
        for detail in token_details:
            try:
                timestamp = detail.find_element(By.CLASS_NAME, "timestamp").text
                price = convert_to_float(detail.find_element(By.CLASS_NAME, "price").text)
                volume = convert_to_float(detail.find_element(By.CLASS_NAME, "volume").text)
                historical_data.append({
                    "timestamp": timestamp,
                    "price": price,
                    "volume": volume,
                })
            except Exception as e:
                log_message(f"Error processing token details: {e}", level="error")
                continue

        log_message(f"Fetched historical data: {len(historical_data)} entries.")
        driver.back()  # Navigate back to the main list
        return historical_data
    except Exception as e:
        log_message(f"Error fetching historical data: {e}", level="error")
        driver.back()
        return []


def calculate_token_age_days(creation_date):
    """
    Calculates the age of a token in days.
    """
    try:
        creation_date = datetime.strptime(creation_date, "%Y-%m-%d")
        current_date = datetime.now()
        return (current_date - creation_date).days
    except Exception as e:
        log_message(f"Error calculating token age: {e}", level="error")
        return 0

def simulate_trading(driver, token_name, price, liquidity, market_cap=None, volume=None):
    """
    Simulates real-time trading by assuming an entry price and monitoring for the target (1.5x).
    Respects pause/stop commands and stops after 1 minute if the target is not reached.

    Args:
        driver: Selenium WebDriver instance.
        token_name (str): Name of the token being traded.
        price (float): Current price of the token.
        liquidity (float): Current liquidity of the token.
        market_cap (float, optional): Market cap of the token.
        volume (float, optional): Trading volume of the token.
    """
    if price is None or price <= 0 or liquidity is None or liquidity <= 0:
        log_message(f"Invalid data for {token_name}: Price={price}, Liquidity={liquidity}. Skipping simulation.")
        return

    log_message(f"Starting simulation for {token_name} with entry price {price}.")
    target_price = price * CONFIG["TAKE_PROFIT"]
    simulated_data = {
        "token_name": token_name,
        "entry_price": price,
        "target_price": target_price,
        "liquidity": liquidity,
        "market_cap": market_cap if market_cap else 0,  # Ensure it's recorded
        "volume": volume if volume else 0,  # Ensure it's recorded
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    start_time = time.time()
    success = False
    current_price = price  # Default to entry price initially

    try:
        while True:
            if is_bot_stopped():
                log_message("Bot is stopped. Exiting simulation.")
                return
            if is_bot_paused():
                log_message("Simulation paused. Waiting to resume...")
                wait_for_resume()
                log_message("Simulation resumed.")

            # Timeout after 1 minute
            elapsed_time = time.time() - start_time
            if elapsed_time > 60:
                log_message(f"Simulation timeout reached for {token_name}.")
                break

            # Fetch live price
            try:
                price_text = driver.find_element(By.XPATH, "//div[@data-cable-val='priceUsd']").get_attribute("data-value")
                fetched_price = convert_to_float(price_text)
                if fetched_price is not None:
                    current_price = fetched_price
                    log_message(f"Current price for {token_name}: {current_price}")
                else:
                    log_message(f"Invalid price detected for {token_name}, keeping previous price.", level="warning")
            except Exception as e:
                log_message(f"Error fetching live price for {token_name}: {e}", level="warning")

            # Check if the target price is reached
            if current_price >= target_price:
                log_message(f"Simulation success: {token_name} reached target price {current_price}.")
                simulated_data["success"] = True
                simulated_data["profit"] = current_price - price
                success = True
                break

            time.sleep(5)  # Simulate periodic checking

        # Ensure all data fields are updated
        simulated_data.update({
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "final_price": current_price if current_price is not None else price,  # Ensure valid final price
            "success": success,
            "profit": (current_price - price) if current_price is not None else 0,
        })

        # Store simulation data in CSV
        save_simulation_result(simulated_data)

    except Exception as e:
        log_message(f"Error during simulation for {token_name}: {e}", level="error")

    # Return to Photon Memescope
    driver.get(CONFIG["PHOTON_URL"])
    log_message(f"Returned to Photon Memescope after simulating {token_name}.")


def set_filters(driver):
    """
    Applies pre-defined filters to the token list on the page.

    Args:
        driver: Selenium WebDriver instance.
    """
    try:
        filter_button = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Filters')]"))
        )
        filter_button.click()
        log_message("Clicked on the Filters button.")

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//input[@name='liquidity-min']"))
        ).send_keys(str(CONFIG["MIN_LIQUIDITY"]))
        log_message("Applied minimum liquidity filter.")

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Apply')]"))
        ).click()
        log_message("Filters applied successfully.")
    except Exception as e:
        log_message(f"Error setting filters: {e}", level="error")

simulation_results = []  # Global list to store all simulation results

def save_simulation_result(result):
    """
    Appends the simulation result to historical_data.csv.
    Ensures every trade (win/loss/manual refresh) is saved correctly.
    Retrains AI after every saved trade.
    """
    csv_file = os.path.join("data", "historical_data.csv")
    file_exists = os.path.isfile(csv_file)

    required_columns = [
        "token_name", "entry_price", "target_price", "liquidity",
        "start_time", "end_time", "final_price", "success", "profit",
        "market_cap", "volume"
    ]

    # ✅ Fill missing columns with default values
    for column in required_columns:
        if column not in result:
            result[column] = 0 if column in ["entry_price", "liquidity", "target_price", "final_price", "profit", "market_cap", "volume"] else None

    try:
        previous_size = os.path.getsize(csv_file) if file_exists else 0  # ✅ Track file size before writing

        with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=required_columns)

            if not file_exists:
                writer.writeheader()  # ✅ Write headers if file does not exist

            writer.writerow(result)
            file.flush()  # ✅ Ensure data is written immediately
            os.fsync(file.fileno())  # ✅ Force OS-level write

        # ✅ Confirm the trade was saved
        new_size = os.path.getsize(csv_file)
        if new_size > previous_size:
            log_message(f"✅ Trade saved successfully: {result}")
        else:
            log_message(f"⚠️ Warning: Trade may not have been saved properly! ({result})", level="warning")

        # ✅ **Retrain AI Model After Every Trade**
        log_message("🔄 Retraining AI model after trade save...")
        train_price_prediction_model_from_csv(force_retrain=True)

    except Exception as e:
        log_message(f"❌ Error saving simulation result to CSV: {e}", level="error")

def analyze_simulation_results():
    """Analyze simulation results from historical_data.csv."""
    file_path = "data/historical_data.csv"
    if not os.path.exists(file_path):
        log_message("No historical data file found. Returning empty analysis.")
        return {}

    try:
        data = pd.read_csv(file_path)
        if data.empty:
            log_message("Historical data CSV is empty.")
            return {}

        #  Ensure required columns exist before analysis
        required_columns = {"success", "profit", "token_name", "entry_price", "target_price", "liquidity"}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            log_message(f"Missing columns in historical data: {missing_columns}", level="error")
            return {}

        #  Convert 'success' column to boolean in case it's stored as a string
        data["success"] = data["success"].astype(str).str.lower() == "true"

        #  Filter for successful trades
        successful_trades = data[data["success"]]

        if successful_trades.empty:
            log_message("No successful trades found in the historical data.")
            return {}

        #  Find the trade with the highest profit
        best_trade = successful_trades.loc[successful_trades["profit"].idxmax()]
        log_message(f"Best simulation result: {best_trade.to_dict()}")

        return best_trade.to_dict()

    except Exception as e:
        log_message(f"Error analyzing simulation results: {e}", level="error")
        return {}
