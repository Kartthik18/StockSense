# Stock Price Predictor using LSTM & News Sentiment

A multivariate LSTM-based stock price prediction pipeline that combines historical OHLCV data with real-time news sentiment analysis to forecast closing prices for stocks listed on **NSE** and **NASDAQ**.

---

## Features

- Fetches 5 years of historical stock data via **yfinance**
- Pulls recent news articles via **NewsAPI** and filters English-language content
- Runs **financial sentiment analysis** using a fine-tuned DistilRoBERTa model (`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`)
- Generates technical indicators using **stockstats**
- Trains a **multivariate LSTM** model with features: `open`, `high`, `low`, `volume`, `Weighted_Sentiment`
- Evaluates model performance with a comprehensive set of regression and directional metrics

---

## Pipeline Overview

```
yfinance (5y data)
       │
       ▼
  stockstats wrap → OHLCV features
       │
       ▼
  NewsAPI → langdetect (EN filter) → DistilRoBERTa sentiment
       │                                     │
       └──────────────────────────────────────┘
                         │
                         ▼
              Merge → historical_data
                         │
                         ▼
               MinMaxScaler (X & y)
                         │
                         ▼
         create_sequences (window=60)
                         │
                         ▼
              LSTM(64) → Dense(1)
                         │
                         ▼
              Inverse Transform → Evaluate
```

---

## Model Architecture

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                          │ (None, 64)                  │          17,920 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 1)                   │              65 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 17,985 (70.25 KB)
```

- **Input shape**: `(60, 5)` — 60-day window × 5 features
- **Optimizer**: Adam
- **Loss**: MSE
- **Epochs**: 18 | **Batch size**: 32
- **Train/Test split**: 80/20

---

## Training Progress

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1     | 0.0390    | 0.0845   |
| 5     | 0.0023    | 0.0164   |
| 10    | 0.0020    | 0.0074   |
| 15    | 0.0017    | 0.0050   |
| 18    | 0.0016    | 0.0046   |

> Val loss consistently decreased across all 18 epochs — no overfitting observed.

---

## Baseline Comparisons

| Method              | MSE (Actual Price Scale) |
|---------------------|--------------------------|
| Standard Moving Avg | 136.49                   |
| EMA (decay=0.5)     | 6.24                     |

---

## LSTM Test Set Evaluation

| Metric               | Value     |
|----------------------|-----------|
| MAE                  | 7.4089    |
| RMSE                 | 9.1086    |
| MAPE                 | 2.49%     |
| R²                   | 0.9017    |
| Directional Accuracy | 48.95%    |

> **R² of 0.90** indicates the model explains ~90% of the variance in closing prices.  
> **MAPE of 2.49%** reflects strong average prediction accuracy within ~2.5% of actual price.  
> Directional Accuracy (~49%) suggests the model is not reliably predicting next-day movement direction — a common limitation of LSTM on financial data without more features.

---

## Installation

```bash
pip3 install yfinance stockstats newsapi-python langdetect transformers tensorflow scikit-learn matplotlib
```

> **Note**: Requires Python 3.10+. Tested on Python 3.14 with TensorFlow 2.22.0rc0 on macOS (Apple Silicon).

---

## ▶Usage

```bash
cd ~/Downloads
python3 yfinance_predictor.py
```

You will be prompted:
```
This only deals with stocks listed on NSE & NASDAQ

enter a valid stock name : AAPL
Is your stock listed on NSE (y/n) : n
```

---

## Dependencies

| Package         | Purpose                              |
|----------------|--------------------------------------|
| `yfinance`      | Download historical stock data       |
| `stockstats`    | Technical indicators wrapper         |
| `newsapi-python`| Fetch recent news articles           |
| `langdetect`    | Filter English-only articles         |
| `transformers`  | HuggingFace sentiment pipeline       |
| `tensorflow`    | LSTM model training                  |
| `scikit-learn`  | Preprocessing & evaluation metrics   |
| `matplotlib`    | Plotting                             |
| `pandas`        | Data manipulation                    |
| `numpy`         | Numerical operations                 |

---

## Limitations

- News sentiment is fetched at runtime and may not align perfectly with historical trading dates (older dates default to NEUTRAL)
- The model is not financial advice — directional accuracy (~49%) is close to random for short-term trading
- NewsAPI free tier limits article history to 1 month

---

## Possible Improvements

- Add more LSTM layers or use Bidirectional LSTM
- Incorporate more technical indicators (RSI, MACD, Bollinger Bands)
- Use a richer historical news dataset aligned with trading dates
- Try Transformer-based time-series models (e.g., Temporal Fusion Transformer)
- Hyperparameter tuning (learning rate, window size, epochs)
