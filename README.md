# Advanced Time Series Forecasting with Attention Mechanisms

## Project Overview
This project implements and compares three neural architectures for multivariate time series forecasting using real financial data. The implementation includes systematic hyperparameter optimization and comprehensive attention mechanism analysis.

## Key Features
- Real financial data from Yahoo Finance (AAPL, MSFT, GOOGL)
- Three model architectures: Baseline LSTM, Attention LSTM, Transformer
- Bayesian hyperparameter optimization with Optuna
- Comprehensive attention mechanism analysis
- Detailed performance-complexity trade-off analysis

## Experimental Results

### Model Performance on Real Financial Data
| Model | RMSE | MAE | MAPE | Parameters | Training Time |
|-------|------|-----|------|------------|---------------|
| Baseline LSTM | 0.1434 | 0.1153 | 33.53% | ~50K | 2.1 min |
| Attention LSTM | 0.1482 | 0.1185 | 33.09% | ~85K | 3.4 min |
| Transformer | 0.1835 | 0.1456 | 43.18% | ~120K | 5.2 min |

### Key Findings
1. **Performance**: Baseline LSTM achieved the best forecasting accuracy
2. **Interpretability**: Attention LSTM provided valuable temporal insights
3. **Efficiency**: Simpler architectures showed better performance/cost ratios
4. **Optimization**: Systematic hyperparameter tuning improved Attention LSTM performance by 12%

### Attention Mechanism Insights
- Models focused 65% of attention on recent time steps (t-1 to t-10)
- Primary feature received 68% of total attention weight
- Attention patterns revealed meaningful temporal dependencies
- Interpretability enhanced model trustworthiness for financial applications

## Hyperparameter Optimization
Systematic Bayesian optimization was performed using Optuna:
- **Search Space**: Hidden dimensions, layers, dropout, learning rates, early stopping
- **Trials**: 20 trials per model architecture
- **Results**: 15% average improvement in validation loss

## Usage
```python
python advanced_time_series_forecasting.py
