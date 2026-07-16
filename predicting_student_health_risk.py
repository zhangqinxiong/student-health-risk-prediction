# ============================================================
# 0. Imports & Configuration
# ============================================================
import pandas as pd
import numpy as np
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
# DATA_DIR = Path('input')  # local path


# ============================================================
# Path Configuration
# ============================================================
import os
from pathlib import Path

DATA_DIR = Path('input')
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Data path: {DATA_DIR}")
print(f"Output path: {OUTPUT_DIR}")


logger.info("=" * 60)
logger.info("STEP 1: LOADING DATA")
logger.info("=" * 60)

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

logger.info(f"Train shape: {train.shape}")
logger.info(f"Test shape: {test.shape}")
logger.info(f"Train columns: {list(train.columns)}")
logger.info(f"Test columns: {list(test.columns)}")

# Define column groups and convert types
num_cols = ['sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure',
            'step_count', 'exercise_duration', 'water_intake']
cat_cols = ['diet_type', 'stress_level', 'sleep_quality',
            'physical_activity_level', 'smoking_alcohol', 'gender']

for col in num_cols:
    train[col] = pd.to_numeric(train[col], errors='coerce')
    test[col] = pd.to_numeric(test[col], errors='coerce')


# ============================================================
# STEP 2: FEATURE ENGINEERING
#
# Core strategy:
#   1. Ordinal encoding BEFORE imputation -> preserves NaN for CatBoost
#   2. CatBoost handles NaN natively -> missing values carry signal
#   3. Experiment: encode->impute (0.949) >> impute->encode (0.908)
#   4. No derived features, no binning, no OHE (contrib < 0.04%)
#   5. Final: 7 raw numeric + 6 ordinal encoded = 13 features
# ============================================================
logger.info("=" * 60)
logger.info("STEP 3: FEATURE ENGINEERING")
logger.info("=" * 60)

def engineer_features(df, is_train=True):
    df = df.copy()

    # ---- 3a. Ordinal encoding (preserves NaN for CatBoost native handling) ----
    # Encoding before imputation is critical:
    #   encode->impute: 0.9489 (CatBoost sees NaN in ordinal columns)
    #   impute->encode: 0.9084 (all NaN filled, no missing signal)
    # Mapping: low/medium/high -> 0/1/2, preserves ordinal ordering
    ord_maps = {
        'stress_level':           {'low': 0, 'medium': 1, 'high': 2},
        'sleep_quality':          {'poor': 0, 'average': 1, 'good': 2},
        'physical_activity_level': {'sedentary': 0, 'moderate': 1, 'active': 2},
        'smoking_alcohol':        {'no': 0, 'occasional': 1, 'yes': 2},
        'diet_type':              {'veg': 0, 'balanced': 1, 'non-veg': 2},
        'gender':                 {'female': 0, 'male': 1, 'other': 2},
    }
    for col, mapping in ord_maps.items():
        df[f'{col}_ord'] = df[col].map(mapping)  # NaN stays NaN
    logger.info("  [3a] Ordinal encoding: 6 categories -> 0/1/2 (NaN preserved)")

    # ---- 3b. Handle missing values ----
    # Numeric: drop if missing > 50%, else median impute
    # Train stores fill values; test reuses train's values (no leakage)
    if is_train:
        drop_cols = []
        for col in num_cols:
            miss_rate = df[col].isnull().mean()
            if miss_rate > 0.5:
                drop_cols.append(col)
                logger.info(f"  [3b] Dropping '{col}' (missing rate: {miss_rate:.2%})")
        engineer_features._drop_cols = drop_cols
    else:
        drop_cols = getattr(engineer_features, '_drop_cols', [])

    for col in num_cols:
        if col in drop_cols:
            df = df.drop(columns=[col])
            continue
        if is_train:
            fill_val = df[col].median()
            setattr(engineer_features, f'{col}_fill', fill_val)
        else:
            fill_val = getattr(engineer_features, f'{col}_fill', df[col].median())
        df[col] = df[col].fillna(fill_val)

    kept_num = [c for c in num_cols if c not in drop_cols]
    logger.info(f"  [3b] Numeric: dropped {len(drop_cols)}, kept {len(kept_num)} (median filled)")

    # ---- 3c. Drop original categorical strings (replaced by ordinal encodings) ----
    for col in cat_cols:
        df = df.drop(columns=[col])
    df = df.drop(columns=['id'], errors='ignore')

    n_feat = len(df.columns) - (0 if 'health_condition' not in df.columns else 1)
    logger.info(f"  [3c] Final features: {n_feat}")
    return df


# Apply feature engineering to train and test
train_fe = engineer_features(train, is_train=True)
test_fe = engineer_features(test, is_train=False)

# Separate target (only in train) before column alignment
y = train_fe['health_condition'].values
train_fe = train_fe.drop(columns=['health_condition'])
test_fe = test_fe.drop(columns=['health_condition'], errors='ignore')

# Align columns between train and test
train_fe, test_fe = train_fe.align(test_fe, join='inner', axis=1)

logger.info(f"\nFinal features - Train: {train_fe.shape}, Test: {test_fe.shape}")
logger.info(f"Feature list:\n{list(train_fe.columns)}")

X = train_fe.values
X_test = test_fe.values
ids_test = test['id'].values

logger.info(f"\nX shape: {X.shape}")
logger.info(f"X_test shape: {X_test.shape}")
logger.info(f"Target classes: {np.unique(y)}")


# (OHE removed - experiment showed < 0.02% impact with 31 features)


# ============================================================
# 4. LIGHTGBM 5-FOLD CROSS-VALIDATION
# ============================================================
logger.info("=" * 60)
logger.info("STEP 4: LIGHTGBM 5-FOLD CV")
logger.info("=" * 60)

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_enc = le.fit_transform(y)
logger.info(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

cv_scores = []
oof_preds = np.zeros(len(y_enc))
test_preds = np.zeros((len(X_test), len(le.classes_)))

lgb_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'class_weight': 'balanced',
    'random_state': SEED,
    'device': 'gpu',
    'verbose': -1,
}

logger.info(f"LightGBM params: {lgb_params}")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enc)):
    logger.info(f"\n{'='*40}")
    logger.info(f"FOLD {fold + 1}/{n_splits}")
    logger.info(f"{'='*40}")

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

    logger.info(f"Train size: {X_tr.shape[0]}, Val size: {X_val.shape[0]}")
    logger.info(f"Train class dist: {np.bincount(y_tr)}")
    logger.info(f"Val class dist:   {np.bincount(y_val)}")

    model = lgb.LGBMClassifier(**lgb_params)

    start_time = datetime.now()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    best_iter = model.best_iteration_
    logger.info(f"Training time: {elapsed:.1f}s (best iteration: {best_iter})")

    val_prob = model.predict_proba(X_val)
    val_pred = val_prob.argmax(axis=1)
    oof_preds[val_idx] = val_pred

    bal_acc = balanced_accuracy_score(y_val, val_pred)
    cv_scores.append(bal_acc)
    logger.info(f"Fold {fold + 1} Balanced Accuracy: {bal_acc:.6f}")

    test_prob = model.predict_proba(X_test)
    test_preds += test_prob / n_splits


# Aggregate CV results
logger.info(f"\n{'='*60}")
logger.info(f"CV RESULTS - {n_splits}-FOLD STRATIFIED")
logger.info(f"{'='*60}")
logger.info(f"Per-fold balanced accuracies: {[f'{s:.6f}' for s in cv_scores]}")
logger.info(f"Mean balanced accuracy: {np.mean(cv_scores):.6f} +/- {np.std(cv_scores):.6f}")

oof_bal_acc = balanced_accuracy_score(y_enc, oof_preds.astype(int))
logger.info(f"OOF Balanced Accuracy: {oof_bal_acc:.6f}")

cm = confusion_matrix(y_enc, oof_preds.astype(int))
logger.info(f"\nOOF Confusion Matrix:\n{cm}")
logger.info(f"\nOOF Classification Report:\n{classification_report(y_enc, oof_preds.astype(int), target_names=le.classes_)}")


# ============================================================
# 5. SUBMISSION
# ============================================================
logger.info("=" * 60)
logger.info("STEP 5: GENERATING SUBMISSION")
logger.info("=" * 60)

test_labels = le.inverse_transform(test_preds.argmax(axis=1))
sub = pd.DataFrame({'id': ids_test, 'health_condition': test_labels})
sub.to_csv(OUTPUT_DIR / 'submission.csv', index=False)

logger.info(f"Submission saved to {OUTPUT_DIR / 'submission.csv'}")
logger.info(f"Shape: {sub.shape}")
logger.info(f"Submission head:\n{sub.head(10)}")
logger.info(f"Submission distribution:\n{sub['health_condition'].value_counts()}")

logger.info("\n" + "=" * 60)
logger.info("DONE")
logger.info("=" * 60)

