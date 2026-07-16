# Predicting Student Health Risk

[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20Series%20S6E7-blue)](https://kaggle.com/competitions/playground-series-s6e7)

Multiclass classification predicting student health status (**fit / at-risk / unhealthy**) from lifestyle and physiological indicators, using **CatBoost** with GPU acceleration.

**Public LB Score: ~0.9494** (Balanced Accuracy)

## Problem

Severe class imbalance (85.9% at-risk, 8.4% unhealthy, 5.8% fit). Evaluation metric: **Balanced Accuracy**.

## Pipeline

```
train.csv/test.csv → missing fill → Ordinal/OneHot encode → CatBoost 5-Fold CV → average → submission.csv
```

| Step | Detail |
|------|--------|
| **Numeric missing** | Median fill (7 features: sleep_duration, heart_rate, bmi, etc.) |
| **Categorical missing** | Fill with `'missing'` as independent category (retains missing signal) |
| **Ordinal encode** | stress_level, sleep_quality, physical_activity_level |
| **OneHot encode** | diet_type, smoking_alcohol, gender |
| **Model** | CatBoost (GPU, `auto_class_weights='Balanced'`, early stopping 100) |
| **CV** | 5-Fold Stratified, average test predictions |
| **Output** | `output/submission.csv` |

## Results

| Seed | CV Balanced Accuracy |
|------|:---:|
| 42 | 0.9494 |
| 999 | 0.9495 |
| 5-seed avg | 0.9493 |
| **Public LB** | **~0.9494** |

### Feature Importance

| Feature | Importance |
|---------|:---------:|
| sleep_duration | 22% |
| stress_level | 18% |
| bmi | 15% |
| exercise_duration, heart_rate, water_intake, step_count, ... | 6-7% each |

## Key Insight

Changing categorical NaN from mode imputation to `'missing'` as a separate category improved Balanced Accuracy from **0.909 → 0.949** (+4%). All other feature engineering attempts (interactions, ratios, polynomials) showed no significant gain.

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training script |
| `input/` | Data files (train.csv, test.csv) |
| `output/` | Submission CSV |

## Usage

```bash
pip install pandas numpy scikit-learn catboost
python train.py
```

## License

MIT
