# Typing Performance Forecasting

Predicting typing metrics for day 999 using 999 days of MonkeyType history. 

## Problem

Given 999 days of typing practice data, forecast the next day's values for four metrics:

- **WPM** — words per minute
- **Accuracy** — percentage of correct characters
- **Raw WPM** — speed before error correction
- **Consistency** — steadiness of typing speed throughout a session

## Approach

LightGBM with lag-based feature engineering and Optuna hyperparameter tuning.

Since LightGBM doesn't understand time order natively, past values are encoded as explicit features using a sliding window approach:

- **Lag features** — values at offsets of 1, 2, 3, 5, 7, 14, 21, and 30 days
- **Rolling statistics** — mean and standard deviation over 7, 14, and 30-day windows

One model is trained per metric (4 models total), each with its own tuned hyperparameters. The final 150 days are held out for validation.

## Results

| Metric | Default MAE | Tuned MAE |
|---|---|---|
| WPM | 10.122 | 9.610 |
| Accuracy | 3.070 | 3.039 |
| Raw WPM | 9.747 | 9.356 |
| Consistency | 10.091 | 9.851 |

### Day 999 Predictions

| Metric | Prediction |
|---|---|
| WPM | 104.79 |
| Accuracy | 95.77 |
| Raw WPM | 111.51 |
| Consistency | 74.03 |

## Files

```
├── typing_speed_train.csv       # training data (999 days)
├── Round3_Model.ipynb           # full notebook (EDA, training, predictions)
├── predictions.csv              # final predictions for day 999
└── Round3_Report.pdf            # methodology report
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pandas numpy lightgbm matplotlib scikit-learn optuna jupyter
```

Then open `Round3_Model.ipynb` and run all cells top to bottom.

## Dependencies

- pandas, numpy
- lightgbm
- optuna
- matplotlib
- scikit-learn

## Notes

The scoring function uses cubic penalties for WPM and exponential penalties for Raw WPM, so large prediction errors are heavily punished. The model intentionally predicts near the recent rolling average rather than chasing day-to-day spikes, which minimises catastrophic errors under this penalty scheme.
