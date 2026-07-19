# Predicting Student Health Risk

[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20Series%20S6E7-blue)](https://kaggle.com/competitions/playground-series-s6e7)

**Public LB: 0.95016** — Multiclass classification predicting student health status (**fit / at-risk / unhealthy**) from lifestyle and physiological indicators.

## Problem

| Metric | Value |
|--------|-------|
| Classes | fit (5.8%), at-risk (85.9%), unhealthy (8.4%) |
| Train size | 690,088 |
| Test size | 295,753 |
| Evaluation | **Balanced Accuracy** (per-class recall averaged) |

Severe class imbalance — naive majority-class prediction yields 0.333 BA.

## Approach

Three-model ensemble (CatBoost + XGBoost + LightGBM) with **per-fold smoothed target encoding** and **Dirichlet-optimized weighted blending**.

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐
│ Preprocess  │ → │ Target Encoding  │ → │ 3 Models × 5-Fold│ → │ Dirichlet     │
│ Ordinal/OH  │    │ (per-fold, leak- │    │ CatBoost         │    │ Weight Search │
│ Median fill │    │  free, smoothed) │    │ XGBoost          │    │ + Blend       │
│ 19 features │    │ +24 TE features  │    │ LightGBM         │    │ → submission  │
└─────────────┘    └──────────────────┘    └──────────────────┘    └───────────────┘
```

### Feature Engineering

| Type | Features | Count |
|------|----------|-------|
| Numeric (passthrough) | sleep_duration, heart_rate, bmi, calorie_expenditure, step_count, exercise_duration, water_intake | 7 |
| Ordinal encoded | stress_level, sleep_quality, physical_activity_level | 3 |
| OneHot encoded | diet_type, smoking_alcohol, gender | 9 |
| **Target Encoding** | per-class probability (3) + count per category → smoothed with global prior | **+24** |
| **Total** | | **43** |

Target encoding is computed **inside each fold** to prevent data leakage. For each categorical value, smoothed class probabilities and sample count are added as features.

### Models

| Model | Key Params | CV BA | GPU |
|-------|-----------|-------|-----|
| **CatBoost** | `auto_class_weights='Balanced'`, `eval_metric='MultiClass'`, ES=100 | 0.94916 | ✅ |
| **XGBoost** | `lr=0.03`, `sample_weight=balanced`, `eval_metric='mlogloss'`, ES=100 | 0.94900 | ✅ |
| **LightGBM** | `lr=0.03`, `class_weight='balanced'`, `eval_metric='multi_logloss'`, ES=100 | 0.94946 | ✅ |
| **Ensemble** | Dirichlet-optimized weighted blend (2000 trials on OOF) | **0.94965** | — |

### CV Performance

| Run | Public LB |
|-----|-----------|
| CatBoost single | 0.94938 |
| CatBoost + XGBoost (Dirichlet) | 0.94976 |
| + LightGBM | 0.94980 |
| + Target Encoding | **0.95016** |

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training script |
| `input/` | Competition data (train.csv, test.csv, sample_submission.csv) |
| `output/submission.csv` | Generated submission |
| `.gitignore` | Ignores data files, cache, and training logs |

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (for CatBoost/XGBoost/LightGBM GPU training)

Dependencies: `pandas`, `numpy`, `scikit-learn`, `catboost`, `xgboost`, `lightgbm`

## Usage

```bash
# Install dependencies
pip install pandas numpy scikit-learn catboost xgboost lightgbm

# Run training (full pipeline: preprocessing → 5-fold CV → ensemble → submission)
python train.py
```

The script will:
1. Load and preprocess train/test data
2. Compute per-fold target encoding
3. Train CatBoost, XGBoost, LightGBM with 5-fold CV
4. Find optimal ensemble weights via Dirichlet search on OOF
5. Generate `output/submission.csv`

## Key Experiments Log

| Experiment | CV BA | LB | Note |
|-----------|-------|-----|------|
| CatBoost baseline | 0.94942 | 0.94938 | Single model |
| + XGBoost ensemble | 0.94952 | 0.94976 | Dirichlet blend |
| + LightGBM | 0.94963 | 0.94980 | 3-model ensemble |
| + Target Encoding | 0.94965 | **0.95016** | ✅ best |
| + KBinsDiscretizer | 0.94983 | 0.94992 | CV↑ LB↓ |
| + Sleep domain features | 0.94985 | 0.94973 | CV↑ LB↓ |
| + N-gram crosses | 0.94953 | 0.94981 | No gain |

## Insights

1. **Target encoding dominates manual feature engineering** — per-category smoothed class probabilities consistently outperform hand-crafted domain features
2. **LightGBM benefits from logloss-based monitoring** — using `multi_error` caused premature stopping at 4-5 trees; switching to `multi_logloss` enabled full convergence
3. **Per-fold preprocessing is critical** — target encoding must be computed inside each CV fold to prevent leakage
4. **Unified logloss monitoring** across all three models improves ensemble coherence
5. **LB probing dominates this competition's leaderboard** — many top scores (>0.952) rely on test set leakage / hardcoded flips, not generalizable ML

## License

MIT
