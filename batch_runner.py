import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Import updated functions from main.py
from main import (
    ensemble_prediction,
    fetch_yahoo_data,
    generate_options_recommendation,
)


def fetch_yahoo_trending(limit=100):
    url = "https://finance.yahoo.com/markets/stocks/trending/"
    # url = "https://finance.yahoo.com/markets/stocks/most-active/"
    # url = "https://finance.yahoo.com/markets/stocks/losers/"
    # url = "https://finance.yahoo.com/markets/stocks/gainers/"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"Error fetching Yahoo Most Active: {resp.status_code}")
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("table tbody tr")[:limit]
    tickers = []
    for row in rows:
        cell = row.select_one("td:nth-child(1) a")
        if cell:
            tickers.append(cell.text.strip())
    return tickers

def batch_run(limit=100):
    start_time = time.perf_counter()
    tickers = []
    tickers = fetch_yahoo_trending(limit)
    # tickers.append("IXHL")
    results = []

    try:
        for ticker in tqdm(tickers, desc="Running analysis"):
            try:
                df = fetch_yahoo_data(ticker)
                if df.empty:
                    continue

                # One-day ensemble prediction.
                ensemble_pred, details = ensemble_prediction(df, n_runs=1)

                # Generate options recommendation (only next-day).
                rec = generate_options_recommendation(
                    ticker, df, ensemble_pred,
                    risk_free_rate=0.01, aggressive_mode=True
                )

                # Extract sentiment summary.
                sentiment_str = "N/A"
                try:
                    sentiment_now = rec.get("sentiment", {}) \
                                      .get("sentiment", {}) \
                                      .get("now", {})
                    sentiment_str = f"{sentiment_now.get('labelNormalized', 'N/A')} ({sentiment_now.get('valueNormalized', 'N/A')})"
                except:
                    pass

                result_entry = {
                    "Ticker": ticker,
                    "Current Price": rec["current_price"],
                    "Next-Day Predicted Price": rec["predicted_price"],
                    "Pct Change": rec["pct_change"],
                    "LR Pred": details.get("lr", {}).get("avg"),
                    "ARIMA Pred": details.get("arima", {}).get("avg"),
                    "LSTM Pred": details.get("lstm", {}).get("avg"),
                    "CNN-LSTM Pred": details.get("cnn_lstm", {}).get("avg"),
                    "Sentiment": sentiment_str,
                    "Recommendation": rec["recommendation"],
                    "Option Strategy": rec["option_strategy"],
                    "Expiry": rec["expiration_date"],
                    "Timing": rec["timing"],
                    "Trade Details": rec["trade_details"]
                }
                results.append(result_entry)
                print(f"{ticker}: {rec['recommendation']} | {rec['option_strategy']} | {rec['pct_change']:.2f}%")
            except KeyboardInterrupt:
                print("\nCtrl+C detected during processing of ticker. Skipping current ticker and finishing batch run.")
                raise
    except KeyboardInterrupt:
        print("Ctrl+C detected. Writing results gathered so far to file...")
    finally:
        df_out = pd.DataFrame(results)
        columns_order = [
            "Ticker", "Current Price", "Next-Day Predicted Price", "Pct Change",
            "LR Pred", "ARIMA Pred", "LSTM Pred", "CNN-LSTM Pred",
            "Sentiment", "Recommendation", "Option Strategy", "Expiry", "Timing", "Trade Details"
        ]
        df_out = df_out[columns_order]

        # Round numeric columns.
        numeric_cols = [
            "Current Price", "Next-Day Predicted Price", "Pct Change",
            "LR Pred", "ARIMA Pred", "LSTM Pred", "CNN-LSTM Pred"
        ]
        for col in numeric_cols:
            if col in df_out.columns:
                df_out[col] = pd.to_numeric(df_out[col], errors="coerce").round(2)

        df_out.to_csv("batch_analysis_results.csv", index=False)

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"\nCompleted batch run — results saved to batch_analysis_results.csv")
        print(f"Total elapsed time: {elapsed:.2f} seconds")

        return df_out

if __name__ == "__main__":
    batch_df = batch_run(10)