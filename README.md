```markdown
# Stock Options Ensemble Analyzer

A Python-based toolkit for forecasting next‑day stock prices using multiple models and generating options trading recommendations based on technical indicators, volatility, and sentiment analysis. The project consists of two scripts:

- **`main.py`** – Core library and interactive analysis for a single ticker  
- **`batch_run.py`** – Wrapper to run analysis in batch over trending tickers and export results

---

## Features

- **Data Fetching**  
  - Historical daily prices from Yahoo Finance  
  - Intraday update of the latest close price  
  - Real‑time sentiment from StockTwits API (with fallback endpoint)

- **Technical Indicators**  
  - Moving averages (20‑day, 50‑day)  
  - Relative Strength Index (RSI)  
  - Annualized volatility  
  - Bollinger Bands  
  - Relative volume

- **Forecasting Models**  
  - Linear Regression  
  - ARIMA (auto AIC selection)  
  - LSTM and CNN‑LSTM (with hyperparameter grid search and cross‑validation)  
  - Ensemble prediction with per‑model and cross‑model outlier removal

- **Volatility Modeling**  
  - GARCH(1,1) conditional volatility estimate  
  - Dynamic bullish/bearish thresholds adjusted by realized vs. GARCH volatility

- **Options Recommendation**  
  - Black–Scholes pricing and Greeks  
  - Strike/expiry selection (standard 45 days or short‑dated 10 days in aggressive mode)  
  - Trade entry/exit guidance and bull/bear spread suggestions  
  - Sentiment‑driven threshold nudges

- **Visualization**  
  - Price & moving averages  
  - RSI with overbought/oversold lines  
  - Historical volatility chart

- **Batch Processing**  
  - Scrapes Yahoo Finance “Trending Stocks”  
  - Runs one‑day ensemble + recommendation per ticker  
  - Saves results to `batch_analysis_results.csv`

---

## Requirements

- Python 3.7+  
- Install dependencies via pip:
  ```bash
  pip install pandas numpy matplotlib yfinance requests statsmodels arch tensorflow keras scikit‑learn scipy beautifulsoup4 tqdm
  ```

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone <your‑repo‑url>
   cd <your‑repo‑dir>
   ```

2. **(Optional) Create a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Single‑Ticker Interactive Analysis

```bash
python main.py
```

- **Prompt:** enter a ticker symbol (e.g., `AAPL`).  
- **Output:**  
  - Ensemble forecast and per‑model details printed to console  
  - Options recommendation summary  
  - Three matplotlib plots displayed  

### 2. Batch Analysis of Trending Tickers

```bash
python batch_run.py
```

- **Scrapes** the top 100 trending tickers by default.  
- **Per‑ticker:** runs one‐day ensemble + recommendation (with `n_runs=1`).  
- **CSV Output:** `batch_analysis_results.csv` in project root.  
- **Elapsed time** printed at the end.

You can adjust the batch size by editing the call in `if __name__ == "__main__":` or by passing a different `limit` to `batch_run(limit)`.

---

## File Structure

```
.
├── main.py           # Core functions & interactive CLI
├── batch_run.py      # Batch wrapper & CSV export
├── requirements.txt  # pip dependencies
└── README.md         # This file
```

### `main.py`

- **Data Fetching:**  
  - `fetch_yahoo_data()`  
  - `fetch_stocktwits_sentiment_now()`

- **Indicators & Features:**  
  - `compute_*` (MA, RSI, volatility, Bollinger)  
  - `prepare_features()`

- **Forecasting:**  
  - `predict_price_trend_lr()`  
  - `predict_with_arima()`  
  - `predict_with_lstm()`  
  - `predict_with_cnn_lstm()`  
  - `ensemble_prediction()`

- **Volatility & Thresholds:**  
  - `estimate_garch_volatility()`  
  - `dynamic_thresholds()`

- **Options Logic:**  
  - `black_scholes_price()`  
  - `compute_greeks()`  
  - `generate_options_recommendation()`

- **Plotting:**  
  - `plot_price_and_indicators()`  
  - `plot_RSI()`  
  - `plot_volatility_garch()`

- **Entry Point:**  
  - `main()` for interactive CLI

### `batch_run.py`

- **Web Scraping:**  
  - `fetch_yahoo_trending()` using BeautifulSoup

- **Batch Processor:**  
  - `batch_run(limit=100)` calls into `main.py`’s functions  
  - Builds a pandas DataFrame and writes `batch_analysis_results.csv`  
  - Prints progress with `tqdm`

---

## Configuration & Customization

- **Model Hyperparameters:**  
  - Adjust grid in `predict_with_lstm()` and `predict_with_cnn_lstm()`

- **Sentiment Fallback:**  
  - The StockTwits endpoint retries with `/v2/` if the first call fails

- **Risk‑Free Rate & Aggressiveness:**  
  - Override defaults when calling `generate_options_recommendation()`

- **Batch Size:**  
  - Change the `limit` parameter in `batch_run()` or pass via command‑line args

---

## Contributing

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m "Add feature"`)  
4. Push to branch (`git push origin feature/YourFeature`)  
5. Open a Pull Request

---

## License

This project is licensed under the [MIT License](LICENSE).  
```
