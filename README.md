# Advanced Time Series Forecasting with Uncertainty Quantification

## Project Overview
This project implements and compares neural architectures for multivariate time series forecasting with comprehensive uncertainty quantification using Pinball loss and CRPS scoring.

## Core Features
- Uncertainty quantification using quantile regression
- Pinball loss for model training
- CRPS (Continuous Ranked Probability Score) for evaluation
- Coverage Rate and Prediction Interval Width metrics
- Three model architectures with probabilistic outputs
- Walk-forward cross-validation
- Probabilistic ARIMA baseline

## Key Metrics (Required by Project)
- **CRPS**: Continuous Ranked Probability Score (primary metric)
- **Coverage Rate**: Empirical coverage of prediction intervals
- **Interval Width**: Average width of prediction intervals
- **RMSE/MAE**: Point forecast accuracy (secondary metrics)

## Model Architectures
1. **QuantileTransformer**: Transformer with quantile output
2. **QuantileAttentionLSTM**: LSTM with attention and quantile output  
3. **QuantileLSTM**: Standard LSTM with quantile output
4. **ProbabilisticARIMA**: ARIMA with prediction intervals

## Uncertainty Quantification
- Models output multiple quantiles (0.1, 0.5, 0.9)
- 80% prediction intervals computed from 0.1 and 0.9 quantiles
- Pinball loss optimizes for accurate quantile predictions
- CRPS evaluates overall probabilistic forecast quality

## Usage
```python
python uncertainty_forecasting.py
