# Predicting Student Health Risk

[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20Series%20S6E7-blue)](https://kaggle.com/competitions/playground-series-s6e7)

**Public LB: 0.95027** — Multiclass classification predicting student health status (**fit / at-risk / unhealthy**) from lifestyle and physiological indicators.

## Problem

| Metric | Value |
|--------|-------|
| Classes | fit (5.8%), at-risk (85.9%), unhealthy (8.4%) |
| Train size | 690,088 |
| Test size | 295,753 |
| Evaluation | **Balanced Accuracy** (per-class recall averaged) |

Severe class imbalance — naive majority-class prediction yields 0.333 BA.

## Pipeline

### 1. Missing Value Imputation
- **7 numeric features**: median filling (train column median)
- **6 categorical features**: `'missing'` string filling

### 2. Baseline Features — 7 dimensions
No categorical encoding. Only numeric passthrough. All categorical information is captured by target encoding inside each fold.

### 3. Per-Fold Target Encoding — +24 dimensions
For each of 6 categorical features, compute smoothed class probabilities per category value:
- **Smoothed probability**: `(counts + 10 × prior) / (total + 10)` → 3 classes × 6 = 18 dims
- **Sample count**: 1 × 6 = 6 dims

Computed **inside each fold** to prevent data leakage. Joined with base features → **31 total dimensions**.

### 4. 3 Models × 5-Fold Cross Validation

| Model | Key Params | Best Model |
|-------|-----------|------------|
| CatBoost GPU | lr=0.03, 3000 iter, MultiClass, Balanced, ES=100 | `use_best_model=True` |
| XGBoost CUDA | lr=0.03, 3000 iter, mlogloss, balanced weights, ES=100, **min_delta=0.002** | `EarlyStopping(save_best=True)` |
| LightGBM GPU | lr=0.03, 3000 iter, multi_logloss, balanced, ES=100, **min_delta=0.002** | `predict_proba(num_iteration=best_iter_)` |

### 5. Dirichlet-optimized Ensemble
2000 Dirichlet random trials on OOF predictions → optimal weighted blend.

### 6. Submission
Best weights applied to test predictions → argmax → inverse label encoding.

```
Per fold (~42s):
  TE:     2.8s
  Cat:    7.2s
  XGB:    5.1s
  LGB:   19.6s
  Val:    2.4s
  Test:   5.1s

Total: ~3.5 min for full 5-fold pipeline
```

## Results

| Model | CV BA | Public LB |
|-------|-------|-----------|
| CatBoost | 0.94920 | — |
| XGBoost | 0.94897 | — |
| LightGBM | 0.94936 | — |
| Uniform blend | 0.94946 | — |
| **Dirichlet ensemble** | **0.94955** | **0.95027** |

## Key Experiments Log

| Experiment | CV BA | LB | Note |
|-----------|-------|-----|------|
| CatBoost baseline | 0.94942 | 0.94938 | Single model |
| + XGBoost (Dirichlet) | 0.94952 | 0.94976 | 2-model ensemble |
| + LightGBM | 0.94963 | 0.94980 | 3-model ensemble |
| + Target Encoding | 0.94965 | 0.95016 | ✅ best |
| + min_delta=0.002 (3× faster) | 0.94966 | 0.95013 | Training speed improved |
| + All OneHot (no Ordinal) | 0.94956 | 0.95011 | Same perf, simpler |
| **− OHE (TE only)** | **0.94955** | **0.95027** | 🏆 **Best** |
| + RF imputation | 0.94251 | 0.94550 | Worse |
| + NaN passthrough | 0.94920 | 0.94976 | Worse |

## Insights

1. **Target encoding alone captures all categorical signal** — removing OHE/Ordinal encoding (7 features → 31 total) improved LB from 0.95013 → 0.95027
2. **min_delta=0.002** is critical for early stopping — without it, XGBoost/LightGBM run full 3000 iterations overfitting on logloss while accuracy plateaus
3. **Per-fold target encoding** must be computed inside CV to prevent leakage; global TE inflates CV by 0.002-0.003
4. **RF imputation hurts** — median fill outperforms both NaN passthrough and RF-based imputation
5. **Training speed** improved 5× (213s → 42s per fold) by removing train set predict_proba and adding min_delta early stopping

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training script (292 lines) |
| `input/` | Competition data (train.csv, test.csv, sample_submission.csv) |
| `output/submission.csv` | Generated submission |
| `.gitignore` | Ignores data files, cache, and training logs |

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (for CatBoost/XGBoost/LightGBM GPU training)

Dependencies: `pandas`, `numpy`, `scikit-learn`, `catboost`, `xgboost`, `lightgbm`

## Usage

```bash
pip install pandas numpy scikit-learn catboost xgboost lightgbm
python train.py
```

## License

MIT
