# Student Health Risk Prediction

[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20Series%20S6E7-blue)](https://kaggle.com/competitions/playground-series-s6e7)

Predict student health status (fit / at-risk / unhealthy) from lifestyle and physiological indicators using CatBoost.

## Results

| Metric | Value |
|--------|-------|
| Balanced Accuracy (5-Fold CV) | **0.9489** |
| Overall Accuracy | 94% |
| Features | 13 (7 numeric + 6 ordinal) |
| Model | CatBoost (default params) |

Per-class performance:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| at-risk | 0.99 | 0.93 | 0.96 |
| fit | 0.73 | 0.95 | 0.82 |
| unhealthy | 0.69 | 0.96 | 0.80 |

## Problem Overview

Multiclass classification with severe class imbalance:
| Class | Samples | Percentage |
|-------|---------|------------|
| at-risk | 592,561 | 85.87% |
| unhealthy | 57,724 | 8.36% |
| fit | 39,803 | 5.77% |

**13 raw features**: sleep_duration, heart_rate, bmi, calorie_expenditure, step_count, exercise_duration, water_intake, diet_type, stress_level, sleep_quality, physical_activity_level, smoking_alcohol, gender.

## Feature Engineering

Minimal — only ordinal encoding + imputation:

| Step | Description |
|------|-------------|
| Ordinal encoding | 6 categoricals → 0/1/2 (BEFORE imputation, preserves NaN) |
| Imputation | Median for numeric, mode for categorical (ordinals keep NaN) |
| Final | 7 raw numeric + 6 ordinal encoded = **13 features** |

**Why encode before impute?** CatBoost's native NaN handling captures missing signal.
- encode → impute: **0.949** ✅ (CatBoost sees NaN in ordinal columns)
- impute → encode: **0.908** ❌ (all values filled, no missing signal)

Derived features, binning, and OHE were tested and contributed < 0.04% combined.

## Key EDA Insights
- **sleep_duration**: strongest predictor — fit(7.95h) > at-risk(7.09h) > unhealthy(5.37h)
- **stress_level**: near-deterministic — high=97.5% unhealthy, low=20% fit, medium=99.4% at-risk
- **physical_activity**: active=96.8% fit, sedentary=91.7% at-risk
- **bmi**: unhealthy(24.1) > at-risk(23.0) > fit(21.8)

## Files

| File | Description |
|------|-------------|
| `run.py` | Training script |
| `predicting_student_health_risk.ipynb` | Jupyter notebook |
| `input/` | Training/testing data |
| `output/` | Generated submissions |

## Usage

```bash
pip install pandas numpy scikit-learn catboost
python run.py
```

## Model Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| loss_function | MultiClass | 3-class classification |
| auto_class_weights | Balanced | Handle 85.9%/8.4%/5.8% imbalance |
| task_type | GPU | GPU acceleration |
| early_stopping_rounds | 50 | Prevent overfitting |
| use_best_model | True | Revert to best iteration |

All other parameters use CatBoost defaults: `depth=6`, `learning_rate=auto`, `iterations=1000`.

## License

MIT
