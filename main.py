import datetime
import itertools
import math
import os
import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from statsmodels.tools.sm_exceptions import ValueWarning

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning)
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# Deep Learning (LSTM and CNN-LSTM)
import tensorflow as tf

# For GARCH modeling
from arch import arch_model

# For regime classification (removed in our feature selection)
#from hmmlearn.hmm import GaussianHMM
from keras.layers import LSTM, Conv1D, Dense, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam

# For option pricing and Greeks
from scipy.stats import norm

# Modeling packages
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

# ARIMA
from statsmodels.tsa.arima.model import ARIMA


# -------------------------------
# ----- Sentiment Fetching -----
# -------------------------------
def fetch_stocktwits_sentiment_now(ticker, timestamp=None):
    """
    Fetches the current sentiment for a given ticker using the StockTwits API.
    Tries the standard endpoint first; if it fails, retries with 'v2' in the path.

    Parameters:
      ticker (str): The stock ticker symbol.
      timestamp (str, optional): ISO timestamp (e.g., "2025-04-16T14:55:51.057Z").
                                 Defaults to current UTC time.

    Returns:
      dict: Sentiment data dictionary (empty on failure).
    """
    if timestamp is None:
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    base_url = "https://api-gw-prd.stocktwits.com/sentiment-api"
    urls = [
        f"{base_url}/{ticker}/detail?end={timestamp}",
        f"{base_url}/v2/{ticker}/detail?end={timestamp}"
    ]

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "DNT": "1",
        "Origin": "https://stocktwits.com",
        "Referer": "https://stocktwits.com/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0"
        ),
        "sec-ch-ua": '"Microsoft Edge";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"'
    }

    for url in urls:
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                json_data = resp.json()
                return json_data.get("data", {})
            else:
                # try the next URL
                continue
        except Exception as e:
            print(f"Error fetching sentiment from {url}: {e}")
            traceback.print_exc()
            continue

    print(f"All sentiment endpoints failed for {ticker}")
    return {}

# -------------------------------
# ----- Data Fetching Functions -----
# -------------------------------
def fetch_yahoo_data(ticker, period="1y"):
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if data.empty:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        print(f"No data with period='{period}'. Retrying with start={start_date_str} and end={end_date_str}...")
        data = yf.download(ticker, start=start_date_str, end=end_date_str, interval="1d", auto_adjust=False, progress=False)
    if data.empty:
        print("No data found for ticker:", ticker)
        return data

    # Update the last close using intraday data to reflect current price (e.g., premarket).
    ticker_obj = yf.Ticker(ticker)
    try:
        intraday = ticker_obj.history(period='1d', interval='1m')
        if not intraday.empty:
            latest_price = intraday['Close'].iloc[-1]
            data.iloc[-1, data.columns.get_loc("Close")] = latest_price
    except Exception as e:
        print(f"Intraday data error for {ticker}: {e}")

    required = ["Close", "High", "Low", "Volume"]
    for col in required:
        if col not in data.columns:
            raise KeyError(f"Expected column {col} not found in data.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = data.columns.str.strip()
    lower_columns = {col.lower(): col for col in data.columns}
    if 'close' in lower_columns:
        if lower_columns['close'] != 'Close':
            data.rename(columns={lower_columns['close']: 'Close'}, inplace=True)
    elif 'adj close' in lower_columns:
        data.rename(columns={lower_columns['adj close']: 'Close'}, inplace=True)
    else:
        raise KeyError("No 'Close' or 'Adj Close' column found in the data.")
    data.dropna(inplace=True)
    return data

# -------------------------------
# ----- Technical Indicator Functions -----
# -------------------------------
def compute_moving_average(data, window=20):
    return data['Close'].rolling(window=window).mean()

def compute_RSI(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_volatility(data, window=20):
    returns = data['Close'].pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility

def compute_average_volume(data, window=20):
    return data['Volume'].rolling(window=window).mean()

def compute_BollingerBands(data, window=20, num_std=2):
    ma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = ma + num_std * std
    lower_band = ma - num_std * std
    return ma, upper_band, lower_band

# -------------------------------
# ----- Feature Engineering -----
# -------------------------------
def prepare_features(data):
    df = data.copy()
    df['Close_Lag1'] = df['Close'].shift(1)
    df['MA20'] = compute_moving_average(df, window=20)
    df['RSI'] = compute_RSI(df)
    df['Volatility'] = compute_volatility(df, window=20)
    df['Avg_Volume'] = compute_average_volume(df, window=20)
    df['Rel_Vol'] = df['Volume'] / df['Avg_Volume']
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    features = ['Close_Lag1', 'MA20', 'RSI', 'Volatility', 'Rel_Vol']
    X = df[features]
    y = df['Target']
    return X, y, df

# -------------------------------
# ----- Forecasting Models -----
# -------------------------------
def predict_price_trend_lr(data):
    X, y, df_features = prepare_features(data)
    train_size = int(len(X) * 0.8)
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    last_row = data.iloc[-1]
    next_day_features = {
        'Close_Lag1': last_row['Close'],
        'MA20': compute_moving_average(data, 20).iloc[-1],
        'RSI': compute_RSI(data).iloc[-1],
        'Volatility': compute_volatility(data, 20).iloc[-1],
        'Rel_Vol': last_row['Volume'] / compute_average_volume(data, 20).iloc[-1],
    }
    next_day_X = pd.DataFrame([next_day_features])
    predicted_price = model.predict(next_day_X)[0]
    return predicted_price, model, mse, r2, df_features

def predict_with_arima(data, steps=1):
    """
    Forecasts the underlying price 'steps' days ahead using an ARIMA model with a constant trend.
    
    Parameters:
      data (pd.DataFrame): Historical price data.
      steps (int): Number of steps (days) ahead to forecast.
    
    Returns:
      tuple: (forecast_value, best_model, best_order)
             forecast_value is the forecast for the final step (i.e. the steps-th day ahead).
    """
    series = data['Close'].copy()
    if series.index.freq is None:
        series = series.asfreq('B')
    
    # Suppress frequency-related warnings.
    warnings.filterwarnings("ignore", category=ValueWarning)
    
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    best_aic = np.inf
    best_order = None
    best_model = None
    
    for order in pdq:
        try:
            # Include a constant trend so that the model can capture upward drift.
            model = ARIMA(series, order=order, trend='c')
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = order
                best_model = model_fit
        except Exception:
            continue

    if best_model is None:
        raise Exception("No valid ARIMA model found.")
    
    forecast = best_model.forecast(steps=steps)
    # Return the forecast for the final day of the horizon.
    return forecast.iloc[-1], best_model, best_order

def create_lstm_model(units=50, learning_rate=0.001, sequence_length=10):
    model = Sequential()
    model.add(LSTM(units, input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def predict_with_lstm(data, epochs=50, batch_size=16, sequence_length=10):
    series = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(scaled_series)):
        window = scaled_series[i-sequence_length:i, 0]
        target = scaled_series[i, 0]
        if np.isnan(window).any() or np.isnan(target):
            continue
        X_seq.append(window)
        y_seq.append(target)
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
    if X_seq.size == 0:
        raise ValueError("No valid sequences available for LSTM.")
    param_grid = {'units': [30, 50], 'learning_rate': [0.0005, 0.001]}
    best_model = None
    best_loss = np.inf
    best_params = None
    tscv = TimeSeriesSplit(n_splits=3)
    for units in param_grid['units']:
        for lr in param_grid['learning_rate']:
            losses = []
            for train_index, val_index in tscv.split(X_seq):
                X_train, X_val = X_seq[train_index], X_seq[val_index]
                y_train, y_val = y_seq[train_index], y_seq[val_index]
                model = create_lstm_model(units=units, learning_rate=lr, sequence_length=sequence_length)
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                loss = model.evaluate(X_val, y_val, verbose=0)
                losses.append(loss)
            avg_loss = np.mean(losses)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = {'units': units, 'learning_rate': lr}
                best_model = create_lstm_model(units=units, learning_rate=lr, sequence_length=sequence_length)
                best_model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, verbose=0)
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, sequence_length, 1), dtype=tf.float32)], reduce_retracing=True)
    def predict_fn(x):
        return best_model(x)
    last_seq = scaled_series[-sequence_length:].reshape((1, sequence_length, 1)).astype(np.float32)
    pred_scaled = predict_fn(tf.convert_to_tensor(last_seq))
    predicted_price = scaler.inverse_transform(pred_scaled.numpy())[0][0]
    return predicted_price, best_model, best_params

def create_cnn_lstm_model(filters=64, kernel_size=2, units=50, learning_rate=0.001, sequence_length=10):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(sequence_length, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def predict_with_cnn_lstm(data, epochs=50, batch_size=16, sequence_length=10):
    series = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(scaled_series)):
        window = scaled_series[i-sequence_length:i, 0]
        target = scaled_series[i, 0]
        if np.isnan(window).any() or np.isnan(target):
            continue
        X_seq.append(window)
        y_seq.append(target)
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
    if X_seq.size == 0:
        raise ValueError("No valid sequences available for CNN-LSTM.")
    param_grid = {'filters': [32, 64], 'units': [30, 50], 'learning_rate': [0.0005, 0.001]}
    best_model = None
    best_loss = np.inf
    best_params = None
    tscv = TimeSeriesSplit(n_splits=3)
    for filters in param_grid['filters']:
        for units in param_grid['units']:
            for lr in param_grid['learning_rate']:
                losses = []
                for train_index, val_index in tscv.split(X_seq):
                    X_train, X_val = X_seq[train_index], X_seq[val_index]
                    y_train, y_val = y_seq[train_index], y_seq[val_index]
                    model = create_cnn_lstm_model(filters=filters, units=units, learning_rate=lr, sequence_length=sequence_length)
                    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                    loss = model.evaluate(X_val, y_val, verbose=0)
                    losses.append(loss)
                avg_loss = np.mean(losses)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_params = {'filters': filters, 'units': units, 'learning_rate': lr}
                    best_model = create_cnn_lstm_model(filters=filters, units=units, learning_rate=lr, sequence_length=sequence_length)
                    best_model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, verbose=0)
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, sequence_length, 1), dtype=tf.float32)], reduce_retracing=True)
    def predict_fn(x):
        return best_model(x)
    last_seq = scaled_series[-sequence_length:].reshape((1, sequence_length, 1)).astype(np.float32)
    pred_scaled = predict_fn(tf.convert_to_tensor(last_seq))
    predicted_price = scaler.inverse_transform(pred_scaled.numpy())[0][0]
    return predicted_price, best_model, best_params

def ensemble_prediction(data, n_runs=3):
    """
    Runs each forecasting model n_runs times, removes per-model outliers (>2σ from that model's mean),
    averages the cleaned results to get each model's prediction, then removes outlier models (>2σ from
    the ensemble mean) before taking the final average.

    Returns:
      ensemble_pred (float)
      details (dict) with per-model averages and the final refined list
    """
    def clean_and_average(preds):
        """Remove values >2σ from preds, then return mean (or nan if empty)."""
        if not preds:
            return np.nan
        arr = np.array(preds)
        mu, sigma = arr.mean(), arr.std(ddof=0)
        filtered = arr[np.abs(arr - mu) <= 2 * sigma]
        return filtered.mean() if len(filtered) > 0 else np.nan

    # collect raw runs
    runs = {'lr': [], 'arima': [], 'lstm': [], 'cnn_lstm': []}
    for _ in range(n_runs):
        try:
            p, *_ = predict_price_trend_lr(data)
            runs['lr'].append(p)
        except: pass
        try:
            p, *_ = predict_with_arima(data)
            runs['arima'].append(p)
        except: pass
        try:
            p, *_ = predict_with_lstm(data)
            runs['lstm'].append(p)
        except: pass
        try:
            p, *_ = predict_with_cnn_lstm(data)
            runs['cnn_lstm'].append(p)
        except: pass

    # clean per-model, compute averages
    avg = {}
    for m, preds in runs.items():
        avg[m] = clean_and_average(preds)

    # build list of valid model avgs
    model_preds = [v for v in avg.values() if not np.isnan(v) and v > 0 and v < data['Close'].max() * 2]
    if len(model_preds) == 0:
        raise ValueError("All models failed or produced outliers.")

    # remove ensemble-level outliers (>2σ from ensemble mean)
    arr = np.array(model_preds)
    mu, sigma = arr.mean(), arr.std(ddof=0)
    refined = arr[np.abs(arr - mu) <= 2 * sigma]
    if len(refined) == 0:
        raise ValueError("All model averages are outliers at ensemble level.")

    ensemble_pred = refined.mean()
    details = {
        'lr':      {'raw_runs': runs['lr'],      'avg': avg['lr']},
        'arima':   {'raw_runs': runs['arima'],   'avg': avg['arima']},
        'lstm':    {'raw_runs': runs['lstm'],    'avg': avg['lstm']},
        'cnn_lstm':{'raw_runs': runs['cnn_lstm'],'avg': avg['cnn_lstm']},
        'refined': refined.tolist()
    }
    return ensemble_pred, details

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def compute_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}

def estimate_garch_volatility(data):
    returns = 100 * data['Close'].pct_change().dropna()
    am = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal', rescale=False)
    res = am.fit(disp="off")
    cond_vol = res.conditional_volatility.iloc[-1] / 100
    return cond_vol, res

def dynamic_thresholds(data):
    data = data.copy()
    data['Pct_Change'] = data['Close'].pct_change() * 100
    mean_change = data['Pct_Change'].mean()
    std_change = data['Pct_Change'].std()
    base_bullish = mean_change + std_change
    base_bearish = mean_change - std_change
    garch_vol, _ = estimate_garch_volatility(data)
    factor = 1 + garch_vol  
    bullish_threshold = base_bullish * factor
    bearish_threshold = base_bearish * factor
    return bullish_threshold, bearish_threshold

def generate_options_recommendation(ticker, data, predicted_price, risk_free_rate=0.01, aggressive_mode=False):
    """
    Generates a detailed options trading recommendation based on technical and volatility analysis,
    incorporating a modest sentiment adjustment.
    
    It uses key technical signals (MA, RSI, realized volatility, relative volume), advanced metrics (volatility gap, Bollinger Band squeeze),
    and current sentiment from StockTwits to determine whether a bullish or bearish options trade should be taken.
    
    Sentiment is fetched from the StockTwits API. For example, if the "now" sentiment is moderately bullish, 
    it will nudge the recommendation toward calls (vice versa for bearish).
    
    Parameters:
      ticker (str): Stock ticker symbol.
      data (pd.DataFrame): Historical price data (with 'Close', 'High', 'Low', 'Volume').
      predicted_price (float): Forecasted next-day closing price.
      risk_free_rate (float, optional): Risk-free rate for Black-Scholes. Default is 0.01.
      aggressive_mode (bool, optional): If True, relaxes filters and allows short-dated trades on extreme moves.
      
    Returns:
      dict: Dictionary with recommendations, including:
        - "current_price", "predicted_price", "pct_change", "ma20", "ma50", "rsi",
          "volatility", "relative_volume", "recommendation", "option_strategy", "expiration_date",
          "timing", "option_call_price", "greeks", "dynamic_thresholds", "vol_gap", "trade_details",
          "entry_instruction", "exit_instruction", "recommended_strike", "recommended_strike_sell".
    """
    current_price = float(data['Close'].iloc[-1])
    pct_change = (predicted_price - current_price) / current_price * 100

    ma20 = compute_moving_average(data, window=20).iloc[-1]
    ma50 = compute_moving_average(data, window=50).iloc[-1]
    rsi = compute_RSI(data).iloc[-1]
    volatility = compute_volatility(data, window=20).iloc[-1]
    relative_vol = data['Volume'].iloc[-1] / compute_average_volume(data, window=20).iloc[-1]

    bullish_threshold, bearish_threshold = dynamic_thresholds(data)

    vol_limit = 0.5 if not aggressive_mode else 0.65
    rsi_limit_bullish = 70 if not aggressive_mode else 75
    rsi_limit_bearish = 30 if not aggressive_mode else 25
    pct_threshold = 3 if not aggressive_mode else 2.0

    bullish_technical = (current_price > ma20) and (ma20 > ma50) and (rsi < rsi_limit_bullish)
    bearish_technical = (current_price < ma20) and (ma20 < ma50) and (rsi > rsi_limit_bearish)

    overall_bullish = bool((pct_change > bullish_threshold) and bullish_technical and (volatility < vol_limit) and (relative_vol > 1))
    overall_bearish = bool((pct_change < bearish_threshold) and bearish_technical and (volatility < vol_limit) and (relative_vol > 1))

    cond_vol, _ = estimate_garch_volatility(data)
    vol_gap = cond_vol - volatility
    vol_note = "High volatility gap: options appear expensive." if vol_gap > 0.05 else ""

    ma_bb, upper_bb, lower_bb = compute_BollingerBands(data, window=20, num_std=2)
    bb_width = (upper_bb.iloc[-1] - lower_bb.iloc[-1]) / ma_bb.iloc[-1]
    bb_note = "Bollinger Band squeeze detected: breakout likely." if bb_width < 0.05 else ""

    # Fetch sentiment from StockTwits.
    sentiment_data = fetch_stocktwits_sentiment_now(ticker)
    # We'll use the "now" sentiment valueNormalized as our basic sentiment measure.
    sentiment_value = sentiment_data.get("sentiment", {}).get("now", {}).get("valueNormalized", 50)  # default to neutral (50) if not available
    
    # Adjust technical thresholds slightly based on sentiment.
    # For example, if sentiment is > 55, nudge bullish; if < 45, nudge bearish.
    sentiment_adjustment = 0
    if sentiment_value > 55:
        sentiment_adjustment = 0.5  # nudge threshold down for bullish trades
    elif sentiment_value < 45:
        sentiment_adjustment = -0.5  # nudge threshold down for bearish trades

    # Incorporate sentiment adjustment into bullish/bearish checks.
    overall_bullish = overall_bullish or ((pct_change + sentiment_adjustment) > bullish_threshold and bullish_technical)
    overall_bearish = overall_bearish or ((pct_change + sentiment_adjustment) < bearish_threshold and bearish_technical)

    # Decide expiry: default ~45 days; if aggressive_mode is enabled and predicted move is extreme, then 10 days.
    if aggressive_mode and abs(pct_change) > 5:
        T = 10 / 365
        expiry_note = "Strong predicted move; using short-dated options (10 days expiry)."
    else:
        T = 45 / 365
        expiry_note = ""

    option_call_price = black_scholes_price(current_price, current_price, T, risk_free_rate, volatility, option_type="call")
    greeks = compute_greeks(current_price, current_price, T, risk_free_rate, volatility, option_type="call")

    trade_details = ""
    entry_instruction = ""
    exit_instruction = ""
    recommended_strike = None
    recommended_strike_sell = None
    option_trade_type = "None"

    if overall_bullish:
        if pct_change > pct_threshold:
            recommended_strike = current_price * 1.02
            trade_details = f"Buy a call option with strike ~{recommended_strike:.2f} (slightly OTM)."
        else:
            recommended_strike = current_price
            trade_details = f"Buy an ATM call option at strike ~{recommended_strike:.2f}."
        if vol_gap > 0.05:
            recommended_strike_sell = recommended_strike * 1.05
            trade_details += f" Alternatively, consider a bull call spread: buy call at {recommended_strike:.2f} and sell call at {recommended_strike_sell:.2f}."
        entry_instruction = ("Enter the bullish call trade when technical confirmation occurs (e.g., Bollinger breakout or MA crossover) on a low-IV day.")
        exit_instruction = ("Exit when the underlying nears the predicted target or if stop-loss conditions (e.g., a 50% premium loss) are met.")
        option_trade_type = "Call"
    elif overall_bearish:
        if abs(pct_change) > pct_threshold:
            recommended_strike = current_price * 0.98
            trade_details = f"Buy a put option with strike ~{recommended_strike:.2f} (slightly ITM)."
        else:
            recommended_strike = current_price
            trade_details = f"Buy an ATM put option at strike ~{recommended_strike:.2f}."
        if vol_gap > 0.05:
            recommended_strike_sell = recommended_strike * 0.95
            trade_details += f" Alternatively, consider a bear put spread: buy put at {recommended_strike:.2f} and sell put at {recommended_strike_sell:.2f}."
        entry_instruction = ("Enter the bearish put trade when bearish technical confirmation occurs (e.g., RSI reversal) on a low-IV day.")
        exit_instruction = ("Exit when the underlying nears the predicted target or if stop-loss criteria (e.g., a 50% drop in premium) are triggered.")
        option_trade_type = "Put"
    else:
        trade_details = "No clear directional signal; no options trade recommended."
        entry_instruction = "Wait for clearer technical and volatility signals."
        exit_instruction = ""

    combined_notes = " ".join([vol_note, bb_note, expiry_note]).strip()
    if combined_notes:
        entry_instruction += " | Additional Info: " + combined_notes

    results = {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "pct_change": pct_change,
        "ma20": ma20,
        "ma50": ma50,
        "rsi": rsi,
        "volatility": volatility,
        "relative_volume": relative_vol,
        "recommendation": "Trade options immediately" if overall_bullish or overall_bearish else "Wait for clearer signals",
        "option_strategy": option_trade_type,
        "expiration_date": datetime.date.today() + datetime.timedelta(days=int(T * 365)) if (overall_bullish or overall_bearish) else None,
        "timing": entry_instruction,
        "option_call_price": option_call_price,
        "greeks": greeks,
        "dynamic_thresholds": {"bullish": bullish_threshold, "bearish": bearish_threshold},
        "vol_gap": vol_gap,
        "trade_details": trade_details,
        "entry_instruction": entry_instruction,
        "exit_instruction": exit_instruction,
        "recommended_strike": recommended_strike,
        "recommended_strike_sell": recommended_strike_sell,
        "sentiment": sentiment_data  # include raw sentiment data for reference
    }
    return results

def plot_price_and_indicators(data, ticker):
    plt.figure(figsize=(12,6))
    plt.plot(data.index, data['Close'], label="Close Price")
    plt.plot(data.index, compute_moving_average(data, 20), label="20-day MA")
    plt.plot(data.index, compute_moving_average(data, 50), label="50-day MA")
    plt.title(f"{ticker} Price and Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_RSI(data, ticker):
    plt.figure(figsize=(12,4))
    plt.plot(data.index, compute_RSI(data), label="RSI")
    plt.axhline(70, color='red', linestyle='--', label="Overbought (70)")
    plt.axhline(30, color='green', linestyle='--', label="Oversold (30)")
    plt.title(f"{ticker} RSI")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_volatility_garch(data):
    cond_vol, _ = estimate_garch_volatility(data)
    plt.figure(figsize=(12,4))
    vol_series = compute_volatility(data, 20)
    plt.plot(data.index, vol_series, label="20-day Volatility")
    plt.title("Historical Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    ticker = input("Enter the stock ticker symbol (e.g., AAPL): ").strip().upper()
    
    print("Fetching latest historical data from Yahoo Finance...")
    data = fetch_yahoo_data(ticker)
    if data.empty:
        print("No data found for ticker:", ticker)
        return
    data.index = pd.to_datetime(data.index)
    data = data.asfreq('B')
    data = data.ffill()
    
    print("Performing ensemble prediction across multiple models...")
    ensemble_pred, details = ensemble_prediction(data, n_runs=3)
    print("\nEnsemble Model Predictions:")
    for model_name, info in details.items():
        if model_name != "refined":
            print(f"  {model_name.upper()}: {info['pred']:.2f}")
    print(f"\nEnsemble Predicted Next-Day Price (average): {ensemble_pred:.2f}")
    
    print("Generating options recommendation based on ensemble forecast and sentiment...")
    recommendation = generate_options_recommendation(ticker, data, ensemble_pred, risk_free_rate=0.01, aggressive_mode=True)
    
    print("\n===== Analysis Report for", ticker, "=====")
    print(f"Current Price: ${recommendation['current_price']:.2f}")
    print(f"Ensemble Predicted Next-Day Price: ${recommendation['predicted_price']:.2f}")
    print(f"Predicted Percentage Change: {recommendation['pct_change']:.2f}%")
    print(f"20-day MA: ${recommendation['ma20']:.2f}   50-day MA: ${recommendation['ma50']:.2f}")
    print(f"RSI: {recommendation['rsi']:.2f}")
    print(f"Volatility (annualized, 20-day): {recommendation['volatility']:.2f}")
    print(f"Relative Volume: {recommendation['relative_volume']:.2f}")
    print(f"Dynamic Thresholds (Bullish): {recommendation['dynamic_thresholds']['bullish']:.2f}, (Bearish): {recommendation['dynamic_thresholds']['bearish']:.2f}")
    print(f"Volatility Gap (GARCH - Hist): {recommendation['vol_gap']:.2f}")
    print("\nRecommendation:", recommendation['recommendation'])
    if recommendation['option_strategy'] != "None":
        print("Optimal Options Strategy:", recommendation['option_strategy'])
        print("Recommended Expiration Date:", recommendation['expiration_date'])
        print("Entry Instruction:", recommendation['entry_instruction'])
        print("Exit Instruction:", recommendation['exit_instruction'])
        print(f"Estimated Option Price: ${recommendation['option_call_price']:.2f}")
        print("Option Greeks:", recommendation['greeks'])
        print("Trade Details:", recommendation['trade_details'])
        print(f"Recommended Strike: {recommendation['recommended_strike']:.2f}")
        if recommendation['recommended_strike_sell']:
            print(f"Recommended Strike for Spread Leg: {recommendation['recommended_strike_sell']:.2f}")
    else:
        print("No options strategy recommended based on current signals.")
    print("\nSentiment Data (from StockTwits):")
    print(recommendation['sentiment'])
    
    plot_price_and_indicators(data, ticker)
    plot_RSI(data, ticker)
    plot_volatility_garch(data)

if __name__ == "__main__":
    main()