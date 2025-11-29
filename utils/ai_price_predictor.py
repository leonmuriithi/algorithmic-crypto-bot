import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from utils.logger import log_message
import time

# ✅ Global Variables
TRAINED_MODEL = None
SCALER = None
OPTIMAL_THRESHOLD = 0.5  # Fixed threshold for stability
BEST_AUC_SCORE = 0.0  # Track best AUC-PR ever achieved
SESSION_START = time.time()

# ✅ Validate Data Columns
def validate_data(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        log_message(f"❌ ERROR: Missing required columns: {missing_columns}", level='error')
        raise ValueError(f"Missing required columns: {missing_columns}")

# ✅ Load & Process CSV
def load_and_preprocess_csv(csv_file):
    """Loads and processes historical trade data for training."""
    if not os.path.isfile(csv_file):
        log_message(f"❌ File {csv_file} not found.", level='error')
        return None

    column_names = [
        "token_name", "entry_price", "target_price", "liquidity",
        "start_time", "end_time", "final_price", "success", 
        "profit", "market_cap", "volume"
    ]
    
    df = pd.read_csv(csv_file, names=column_names, header=0, on_bad_lines="warn")

    # ✅ Convert required columns to numeric
    for col in ["entry_price", "final_price", "liquidity", "market_cap", "volume", "profit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["entry_price", "final_price"], inplace=True)

    # ✅ Define Success (1.4x entry price)
    df["reaches_1_4x"] = (df["final_price"] >= df["entry_price"] * 1.4).astype(int)

    # ✅ Fix: Oversampling 4K-5K Market Cap Trades
    failed_trades = df[df["profit"] < 0]
    successful_trades = df[df["profit"] >= 0]

    # ✅ Keep 30% of failed trades instead of 10%
    failed_trades_sampled = failed_trades.sample(frac=0.30, random_state=42)
    df = pd.concat([successful_trades, failed_trades_sampled])

    # ✅ Ensure Data is Shuffled Before Training
    df = df.sample(frac=1, random_state=np.random.randint(1, 10000))

    log_message(f"📊 Loaded {len(df)} historical trades.")
    return df

# ✅ Feature Engineering
def process_features(df):
    """Applies feature engineering to the dataset."""
    df = df.copy()

    # ✅ Apply Log Scaling to Prevent Large Feature Dominance
    df["liquidity_log"] = np.log1p(df["liquidity"])
    df["market_cap_log"] = np.log1p(df["market_cap"])
    df["volume_log"] = np.log1p(df["volume"])

    # ✅ Fix: Adjust Market Cap Scaling to Lower Bias
    df["market_cap_scaled"] = np.log1p(df["market_cap"]) / (df["liquidity"] + 1e-9)
    df["price_to_liquidity"] = df["entry_price"] / (df["liquidity"] + 1e-9)

    # ✅ Prevent Overweighting of Market Cap
    df["liq_mc_ratio"] = df["liquidity"] / (df["market_cap"] + 1e-9)
    df["vol_mc_ratio"] = df["volume"] / (df["market_cap"] + 1e-9)

    return df

# ✅ Train Model
def train_price_prediction_model_from_csv(force_retrain=False):
    """Trains the model with corrected feature weighting and prevents market cap dominance."""
    global TRAINED_MODEL, SCALER, BEST_AUC_SCORE

    if TRAINED_MODEL and SCALER and not force_retrain:
        log_message("✅ Model already trained. Skipping retraining.")
        return TRAINED_MODEL, SCALER

    df = load_and_preprocess_csv("data/historical_data.csv")
    if df is None:
        return None, None

    log_message(f"🛠️ Training model on {len(df)} trades...")

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["reaches_1_4x"], random_state=42)

    train_df = process_features(train_df)
    val_df = process_features(val_df)

    # ✅ **Ensure correct feature set**
    feature_columns = ["liquidity_log", "market_cap_log", "volume_log", 
                       "liq_mc_ratio", "vol_mc_ratio", "price_to_liquidity"]

    X_train = train_df[feature_columns].astype(np.float32)
    y_train = train_df["reaches_1_4x"]

    X_val = val_df[feature_columns].astype(np.float32)
    y_val = val_df["reaches_1_4x"]

    # ✅ **Apply Gaussian Noise to Prevent Identical Predictions**
    def add_noise(df, noise_level=0.0001):
        noise = np.random.normal(0, noise_level, df.shape)
        return df + noise

    X_train = add_noise(X_train)
    X_val = add_noise(X_val)

    # ✅ **Fix: Scale Data with StandardScaler**
    SCALER = StandardScaler()
    X_train_scaled = SCALER.fit_transform(X_train)
    X_val_scaled = SCALER.transform(X_val)

    # ✅ **Adjust Sample Weighting to Reduce Market Cap Bias**
    success_ratio = train_df["reaches_1_4x"].mean()
    scale_pos_weight = max((1 - success_ratio) / success_ratio, 6.0)  # Increase weight to lower-market-cap tokens

    # ✅ **Fixed Model Definition** (Removed `monotone_constraints`)
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        tree_method='hist',
        objective='binary:logistic'
    )

    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=50)

    # ✅ **Evaluate model performance**
    new_auc = evaluate_model_performance(model, SCALER)["AUC_Score"]

    if new_auc >= BEST_AUC_SCORE:
        log_message(f"✅ New model is better ({new_auc:.4f} vs {BEST_AUC_SCORE:.4f}). Updating...")
        TRAINED_MODEL = model
        BEST_AUC_SCORE = new_auc
    else:
        log_message(f"⚠️ New model is worse ({new_auc:.4f}). Keeping old model.")

    return TRAINED_MODEL, SCALER

# ✅ Evaluate Model Performance
def evaluate_model_performance(model, scaler):
    """Evaluates model performance and only retrains if needed."""
    global OPTIMAL_THRESHOLD, BEST_AUC_SCORE

    df = load_and_preprocess_csv("data/historical_data.csv")
    if df is None:
        return {"Accuracy": None}

    log_message("🛠️ Evaluating model on validation dataset...")

    df = process_features(df)

    # ✅ **Fix: Ensure feature names match those created in process_features()**
    feature_columns = ["liquidity_log", "market_cap_log", "volume_log", 
                       "liq_mc_ratio", "vol_mc_ratio", "price_to_liquidity"]  # ✅ Corrected feature name

    # ✅ **Check if all required features exist in DataFrame**
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        log_message(f"❌ ERROR: Missing features in evaluation dataset: {missing_features}", level='error')
        return {"AUC_Score": None}

    X_val = df[feature_columns].astype(np.float32)
    y_val = df["reaches_1_4x"]

    X_val_scaled = scaler.transform(X_val)

    # ✅ Compute Predictions
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]

    # ✅ Compute Precision-Recall AUC (Better than Accuracy)
    auc_score = average_precision_score(y_val, y_pred_proba)

    log_message(f"📈 Model AUC-PR Score: {auc_score:.4f}")

    # ✅ Ensure `BEST_AUC_SCORE` updates correctly
    if auc_score > BEST_AUC_SCORE:
        BEST_AUC_SCORE = auc_score
        log_message(f"🔥 New Best AUC-PR Score: {BEST_AUC_SCORE:.4f}")

    # ✅ **Only Retrain if AUC Score Drops Significantly**
    if auc_score < BEST_AUC_SCORE - 0.05:
        log_message("⚠️ Performance dropped! Retraining model...")
        train_price_prediction_model_from_csv(force_retrain=True)
    else:
        log_message("✅ Model performance is stable. No retraining needed.")

    return {"AUC_Score": auc_score, "Best_AUC_Score": BEST_AUC_SCORE}

# ✅ Predict Future Price
def predict_future_price(token_data):
    """Predicts probability that a token will reach 1.4x its entry price."""
    if TRAINED_MODEL is None or SCALER is None:
        log_message("❌ Model or Scaler not available.", level='error')
        return None

    try:
        EPSILON = 1e-7  # ✅ Prevent NaNs & Inf values
        feature_df = pd.DataFrame([{
            "liquidity_log": np.log1p(max(token_data["liquidity"], EPSILON)),
            "market_cap_log": np.log1p(max(token_data["market_cap"], EPSILON)),  # ✅ FIXED: Matches training data
            "volume_log": np.log1p(max(token_data["volume"], EPSILON)),
            "liq_mc_ratio": (token_data["liquidity"] + EPSILON) / (token_data["market_cap"] + EPSILON),
            "vol_mc_ratio": (token_data["volume"] + EPSILON) / (token_data["market_cap"] + EPSILON),
            "price_to_liquidity": (token_data["entry_price"] + EPSILON) / (token_data["liquidity"] + EPSILON)
        }])

        # ✅ Ensure feature names match exactly
        expected_features = ["liquidity_log", "market_cap_log", "volume_log", 
                             "liq_mc_ratio", "vol_mc_ratio", "price_to_liquidity"]
        
        missing_features = [col for col in expected_features if col not in feature_df.columns]
        if missing_features:
            log_message(f"❌ ERROR: Missing features in prediction: {missing_features}", level='error')
            return None

        feature_scaled = SCALER.transform(feature_df)

        prediction_prob = TRAINED_MODEL.predict_proba(feature_scaled)[0][1]

        log_message(f"📈 Probability: {prediction_prob:.4f}")

        return prediction_prob

    except Exception as e:
        log_message(f"❌ Error during AI prediction: {e}", level='error')
        return None
