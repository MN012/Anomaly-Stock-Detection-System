# Anomaly Stock Detection System

Predicting next-day returns for Ford Motor Company stock using Z-scores, Isolation Forest, residual and LSTM neural networks.

## Results

<img width="1489" height="1190" alt="image" src="https://github.com/user-attachments/assets/b2ba82c1-d53f-45c0-840d-dbc240ea4a1d" />

# This above is the plotted result of the Z-score, Isolation Forest and Residual Anomaly Detection


- **Information Coefficient**: 0.072
- **Sharpe Ratio**: 0.37
- **Data**: 39 years of daily prices (1986-2025)

The IC of 0.072 indicates the model has statistically significant predictive power. In quantitative finance, IC > 0.05 is considered good performance.

## What It Does

The model takes 60 days of historical stock data (prices, volume, technical indicators) and predicts whether the stock will go up or down the next day. It uses two LSTM layers to learn temporal patterns in the data.

## Features Used

The model uses 30+ features including:
- Price momentum (RSI, MACD, lagged returns)
- Volatility indicators (Bollinger Bands, ATR, realized volatility)
- Trend indicators (moving averages, ADX)
- Volume signals (volume ratio, OBV)

## Model Architecture

```
Input: 60 days × N features
↓
LSTM: 64 units
Dropout: 0.2
↓
LSTM: 32 units
↓
Dense: 1 output (predicted return)
```

Trained for 30 epochs with Adam optimizer and MSE loss.

## Why LSTM?

Stock prices have temporal dependencies - patterns from previous days affect future prices. LSTM's gating mechanism allows it to remember relevant information across the 60-day sequence while filtering out noise. This makes it better suited than simpler models like linear regression (which can't capture non-linear patterns) or random forest (which ignores temporal ordering).

## Setup

```bash
pip install -r requirements.txt
jupyter notebook project.ipynb
```

## Data Leakage Prevention

- Train/test split is purely temporal (train ends before test begins)
- Features are created separately for train and test sets
- Scaling parameters learned only from training data
- All features at time t use only data up to time t-1

## Improvements

Current focus is on improving the Sharpe ratio through better position sizing and trade filtering:
- Size positions by prediction confidence
- Adjust for volatility
- Only trade in favorable market conditions

## Requirements

- Python 3.8+
- TensorFlow 2.15+
- See requirements.txt for full list

## License

MIT
