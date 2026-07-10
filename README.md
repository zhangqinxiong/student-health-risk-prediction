# Student Health Risk Prediction

[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20Series%20S6E7-blue)](https://kaggle.com/competitions/playground-series-s6e7)

Predict student health status (fit / at-risk / unhealthy) from lifestyle and physiological indicators using CatBoost.

## Problem Overview

Multiclass classification task with severe class imbalance:
| Class | Samples | Percentage |
|-------|---------|------------|
| at-risk | 592,561 | 85.87% |
| unhealthy | 57,724 | 8.36% |
| fit | 39,803 | 5.77% |

**Features** (13 raw): sleep_duration, heart_rate, bmi, calorie_expenditure, step_count, exercise_duration, water_intake, diet_type, stress_level, sleep_quality, physical_activity_level, smoking_alcohol, gender.

## Results

| Metric | Value |
|--------|-------|
| Balanced Accuracy (5-Fold CV) | **0.9493** |
| Overall Accuracy | 94% |
| Model | CatBoost (default params) |
| Feature Engineering | 46 features, EDA-driven |

Per-class performance:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| at-risk | 0.99 | 0.93 | 0.96 |
| fit | 0.73 | 0.95 | 0.82 |
| unhealthy | 0.69 | 0.96 | 0.80 |

## Feature Engineering (46 features)

Based on EDA findings about key predictors:

| Category | Count | Description |
|----------|-------|-------------|
| Raw Numeric | 7 | Original values (median-imputed) |
| Ordinal Encoded | 6 | Categories mapped to 0/1/2 preserving order |
| Simple Derived | 5 | Arithmetic combinations (sleep_debt, bmi_x_heartrate, etc.) |
| One-Hot Encoded | 18 | 6 raw categories x 3 values |
| Binned OHE | 10 | BMI categories + heart rate zones |

### Key EDA Insights Used
- **sleep_duration**: strongest predictor — fit(7.95h) > at-risk(7.09h) > unhealthy(5.37h)
- **stress_level**: near-deterministic — high=97.5% unhealthy, low=20% fit, medium=99.4% at-risk
- **physical_activity**: active=96.8% fit, sedentary=91.7% at-risk
- **bmi**: unhealthy(24.1) > at-risk(23.0) > fit(21.8)
- **smoking**: yes=45% unhealthy, no=45% fit

## Files

| File | Description |
|------|-------------|
| `run.py` | Training script (Python) |
| `predicting_student_health_risk.ipynb` | Jupyter notebook with detailed walkthrough |
| `features.md` | Complete feature list with rationale |
| `.gitignore` | Git ignore rules |

## Requirements

- Python 3.11+
- pandas, numpy
- scikit-learn
- catboost
- (Optional) GPU with CUDA for faster training

```bash
pip install pandas numpy scikit-learn catboost
```

## Usage

### Local
```bash
# Prepare data in input/ directory
python run.py
```

### Kaggle Notebooks
Set paths in the first code cell:
```python
DATA_DIR = Path('/kaggle/input/competitions/playground-series-s6e7')
OUTPUT_DIR = Path('/kaggle/working/')
```

### Model Parameters
Only non-default parameters specified:
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
