import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_eval
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5
N_DIRICHLET = 2000
SMOOTH_M = 10

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
target = train['health_condition']
test_ids = test['id']

label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)
class_names = label_encoder.classes_
n_classes = len(class_names)

numeric_features = [
    'sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure',
    'step_count', 'exercise_duration', 'water_intake'
]

ordinal_features = ['stress_level', 'sleep_quality', 'physical_activity_level']
ordinal_categories = [
    ['low', 'medium', 'high', 'missing'],
    ['poor', 'average', 'good', 'missing'],
    ['sedentary', 'moderate', 'active', 'missing']
]

onehot_features = ['diet_type', 'smoking_alcohol', 'gender']
cat_features = ordinal_features + onehot_features

for df in [train, test]:
    for col in numeric_features:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_features:
        df[col] = df[col].fillna('missing')

preprocessor = ColumnTransformer(transformers=[
    ('ordinal', OrdinalEncoder(categories=ordinal_categories), ordinal_features),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_features),
    ('num', 'passthrough', numeric_features)
])

X_base = preprocessor.fit_transform(train)
X_test_base = preprocessor.transform(test)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Store OOF probs for each model
oof_catb = np.zeros((len(train), 3))
oof_xgb = np.zeros((len(train), 3))
oof_lgb = np.zeros((len(train), 3))
test_catb = np.zeros((len(test), 3))
test_xgb = np.zeros((len(test), 3))
test_lgb = np.zeros((len(test), 3))

catb_scores = []
xgb_scores = []
lgb_scores = []

global_prior = np.bincount(target_encoded, minlength=n_classes) / len(target_encoded)

def target_encode_fold(train_raw, y_fold, val_raw, test_raw):
    """Per-fold smoothed target encoding for multiclass."""
    prior = np.bincount(y_fold, minlength=n_classes) / len(y_fold)
    categories, inv = np.unique(train_raw, return_inverse=True)
    n_cats = len(categories)
    cat_probas = np.zeros((n_cats, n_classes))
    cat_counts = np.zeros(n_cats)
    for i in range(n_cats):
        mask = inv == i
        counts = np.bincount(y_fold[mask], minlength=n_classes)
        total = counts.sum()
        cat_counts[i] = total
        cat_probas[i] = (counts + SMOOTH_M * prior) / (total + SMOOTH_M)
    proba_map = dict(zip(categories, cat_probas))
    count_map = dict(zip(categories, cat_counts))
    def map_values(values):
        probas = np.array([proba_map.get(v, prior) for v in values])
        counts = np.array([count_map.get(v, 0.0) for v in values])
        return np.column_stack([probas, counts])
    return map_values(val_raw), map_values(test_raw)

# Feature names for target encoding columns (for reference)
te_col_names = []
for col in cat_features:
    for k in range(n_classes):
        te_col_names.append(f'{col}_proba_{class_names[k]}')
    te_col_names.append(f'{col}_count')

for fold, (train_idx, val_idx) in enumerate(skf.split(X_base, target_encoded)):
    print(f'\n{"="*60}')
    print(f'Fold {fold + 1}/{N_FOLDS}')
    print(f'{"="*60}')

    # --- Target Encoding (per-fold, leakage-free) ---
    te_train_list = []
    te_val_list = []
    te_test_list = []
    for col in cat_features:
        train_raw = train[col].iloc[train_idx].values
        val_raw = train[col].iloc[val_idx].values
        test_raw = test[col].values

        prior = np.bincount(target_encoded[train_idx], minlength=n_classes) / len(train_idx)
        categories, inv = np.unique(train_raw, return_inverse=True)
        n_cats = len(categories)

        cat_probas = np.zeros((n_cats, n_classes))
        cat_counts = np.zeros(n_cats)
        for i in range(n_cats):
            mask = inv == i
            counts = np.bincount(target_encoded[train_idx][mask], minlength=n_classes)
            total = counts.sum()
            cat_counts[i] = total
            cat_probas[i] = (counts + SMOOTH_M * prior) / (total + SMOOTH_M)

        proba_map = dict(zip(categories, cat_probas))
        count_map = dict(zip(categories, cat_counts))

        def encode(values):
            probas = np.array([proba_map.get(v, prior) for v in values])
            cnt = np.array([count_map.get(v, 0.0) for v in values])
            return np.column_stack([probas, cnt])

        te_train_list.append(encode(train_raw))
        te_val_list.append(encode(val_raw))
        te_test_list.append(encode(test_raw))

    te_train = np.hstack(te_train_list)
    te_val = np.hstack(te_val_list)
    te_test = np.hstack(te_test_list)

    X_train_fold = np.hstack([X_base[train_idx], te_train])
    X_val_fold = np.hstack([X_base[val_idx], te_val])
    X_test_fold = np.hstack([X_test_base, te_test])

    y_train_fold, y_val_fold = target_encoded[train_idx], target_encoded[val_idx]

    # --- CatBoost ---
    catb = CatBoostClassifier(
        task_type='GPU',
        auto_class_weights='Balanced',
        early_stopping_rounds=100,
        eval_metric='MultiClass',
        random_seed=RANDOM_STATE,
        verbose=0
    )
    catb.fit(Pool(X_train_fold, y_train_fold), eval_set=Pool(X_val_fold, y_val_fold), use_best_model=True)

    catb_val_proba = catb.predict_proba(X_val_fold)
    oof_catb[val_idx] = catb_val_proba
    catb_ba = balanced_accuracy_score(y_val_fold, catb_val_proba.argmax(axis=1))
    catb_f1 = f1_score(y_val_fold, catb_val_proba.argmax(axis=1), average='macro')
    catb_scores.append(catb_ba)
    catb_test_proba = catb.predict_proba(X_test_fold)
    test_catb += catb_test_proba / N_FOLDS
    print(f'  CatBoost  | BA: {catb_ba:.6f} | F1: {catb_f1:.6f} | iter: {catb.get_best_iteration()}')

    # --- XGBoost ---
    xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        eval_metric='mlogloss',
        early_stopping_rounds=100,
        device='cuda',
        random_state=RANDOM_STATE,
        verbosity=0
    )
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_fold)
    xgb.fit(X_train_fold, y_train_fold, sample_weight=sample_weights, eval_set=[(X_val_fold, y_val_fold)], verbose=False)

    xgb_val_proba = xgb.predict_proba(X_val_fold)
    oof_xgb[val_idx] = xgb_val_proba
    xgb_ba = balanced_accuracy_score(y_val_fold, xgb_val_proba.argmax(axis=1))
    xgb_f1 = f1_score(y_val_fold, xgb_val_proba.argmax(axis=1), average='macro')
    xgb_scores.append(xgb_ba)
    xgb_test_proba = xgb.predict_proba(X_test_fold)
    test_xgb += xgb_test_proba / N_FOLDS
    print(f'  XGBoost   | BA: {xgb_ba:.6f} | F1: {xgb_f1:.6f} | iter: {xgb.best_iteration}')

    # --- LightGBM ---
    lgb = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        class_weight='balanced',
        objective='multiclass',
        device='gpu',
        random_state=RANDOM_STATE,
        verbose=-1
    )
    lgb.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], eval_metric='multi_logloss',
            callbacks=[lgb_early_stopping(100), lgb_log_eval(0)])

    lgb_val_proba = lgb.predict_proba(X_val_fold)
    oof_lgb[val_idx] = lgb_val_proba
    lgb_ba = balanced_accuracy_score(y_val_fold, lgb_val_proba.argmax(axis=1))
    lgb_f1 = f1_score(y_val_fold, lgb_val_proba.argmax(axis=1), average='macro')
    lgb_scores.append(lgb_ba)
    lgb_test_proba = lgb.predict_proba(X_test_fold)
    test_lgb += lgb_test_proba / N_FOLDS
    print(f'  LightGBM  | BA: {lgb_ba:.6f} | F1: {lgb_f1:.6f} | iter: {lgb.best_iteration_}')

print(f'\n{"="*60}')
print(f'CatBoost CV: {np.mean(catb_scores):.6f} +/- {np.std(catb_scores):.6f}')
print(f'XGBoost  CV: {np.mean(xgb_scores):.6f} +/- {np.std(xgb_scores):.6f}')
print(f'LightGBM CV: {np.mean(lgb_scores):.6f} +/- {np.std(lgb_scores):.6f}')

# --- Dirichlet weight search ---
print(f'\n{"="*60}')
print('Dirichlet weight search...')
rng = np.random.default_rng(RANDOM_STATE)

uniform = np.ones(3) / 3
blend_uniform = uniform[0] * oof_catb + uniform[1] * oof_xgb + uniform[2] * oof_lgb
ba_uniform = balanced_accuracy_score(target_encoded, blend_uniform.argmax(axis=1))

ba_catb_only = balanced_accuracy_score(target_encoded, oof_catb.argmax(axis=1))
ba_xgb_only = balanced_accuracy_score(target_encoded, oof_xgb.argmax(axis=1))
ba_lgb_only = balanced_accuracy_score(target_encoded, oof_lgb.argmax(axis=1))

best_weight = None
best_ba = 0
results = []

for i in range(N_DIRICHLET):
    w = rng.dirichlet(np.ones(3))
    blended = w[0] * oof_catb + w[1] * oof_xgb + w[2] * oof_lgb
    ba = balanced_accuracy_score(target_encoded, blended.argmax(axis=1))
    results.append((ba, w))
    if ba > best_ba:
        best_ba = ba
        best_weight = w

top5 = sorted(results, key=lambda x: -x[0])[:5]
model_names = ['CatBoost', 'XGBoost', 'LightGBM']

print(f'  CatBoost only:       {ba_catb_only:.6f}')
print(f'  XGBoost only:        {ba_xgb_only:.6f}')
print(f'  LightGBM only:       {ba_lgb_only:.6f}')
print(f'  Uniform (1/3 each):  {ba_uniform:.6f}')
print(f'  Best Dirichlet:      {best_ba:.6f}  (', end='')
for i, name in enumerate(model_names):
    print(f'w_{name}={best_weight[i]:.4f}', end='  ' if i < 2 else '')
print(')')
print(f'  Top-5 weights:')
for ba, w in top5:
    parts = [f'{model_names[i]}={w[i]:.4f}' for i in range(3)]
    print(f'    BA: {ba:.6f}  |  {"  ".join(parts)}')

test_blend = best_weight[0] * test_catb + best_weight[1] * test_xgb + best_weight[2] * test_lgb
test_preds = test_blend.argmax(axis=1)

submission = pd.DataFrame({'id': test_ids, 'health_condition': class_names[test_preds]})
submission.to_csv('output/submission.csv', index=False)
print(f'\nDone! output/submission.csv saved.')
print(f'Target encoding added {te_val.shape[1]} extra features ({len(cat_features)} cats x {n_classes+1})')
